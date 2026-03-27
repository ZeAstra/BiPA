import copy
import einops
import numpy as np
from typing import List, Tuple, Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from mmengine import ConfigDict
from mmengine.model import BaseModule

from mmcv.cnn import build_norm_layer, ConvModule

from mmdet.registry import MODELS
from mmdet.structures.bbox import bbox2roi
from mmdet.models.task_modules import SamplingResult
from mmdet.structures import SampleList, DetDataSample
from mmdet.models.utils import unpack_gt_instances, empty_instances
from mmdet.utils import OptConfigType, MultiConfig, ConfigType, InstanceList
from mmdet.models import MaskRCNN, StandardRoIHead, FCNMaskHead, SinePositionalEncoding

from transformers.models.sam.modeling_sam import SamVisionEncoderOutput

from .common import MultiScaleConv, ColorAttentionAdapter


@MODELS.register_module()
class USISAnchor(MaskRCNN):
    def __init__(
            self,
            shared_image_embedding,
            adapter=None,
            decoder_freeze=True,
            *args,
            **kwargs):
        peft_config = kwargs.get('backbone', {}).get('peft_config', {})
        super().__init__(*args, **kwargs)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)

        self.adapter = False
        if adapter is not None:
            self.adapter = MODELS.build(adapter)

        self.decoder_freeze = decoder_freeze
        self.frozen_modules = []
        if peft_config is None:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.roi_head.mask_head.mask_decoder,
                self.roi_head.mask_head.no_mask_embed,
            ]
        self._set_grad_false(self.frozen_modules)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # 检查 adapter 是否使用
        if self.adapter:
            vit_feature, resnet_features = self.backbone(batch_inputs, adapter=self.adapter)
        else:
            vit_feature, resnet_features = self.backbone(batch_inputs)

        if isinstance(vit_feature, SamVisionEncoderOutput):
            image_embeddings = vit_feature[0]
            vision_hidden_states = vit_feature[1]
        elif isinstance(vit_feature, tuple):
            image_embeddings = vit_feature[-1]
            vision_hidden_states = vit_feature
        else:
            raise NotImplementedError
        # 生成位置嵌入
        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])

        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        # 通过 neck 处理 ResNet 特征
        x = self.neck(vision_hidden_states, resnet_features)

        return x, image_embeddings, image_positional_embeddings

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        losses = dict()
        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal',
                                          self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = \
                torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg)
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.loss(
            x, rpn_results_list, batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        x, image_embeddings, image_positional_embeddings = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


@MODELS.register_module()
class USISFPN(BaseModule):
    def __init__(
            self,
            feature_aggregator=None,
            feature_spliter=None,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, vision_hidden_states, resnet_features):
        if hasattr(self, 'feature_aggregator'):
            x = self.feature_aggregator(vision_hidden_states, resnet_features)
        else:
            x = vision_hidden_states, resnet_features
        if hasattr(self, 'feature_spliter'):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x


@MODELS.register_module()
class USISSimpleFPNHead(BaseModule):
    def __init__(self,
                 hidden_channels: int,
                 out_channels: int,
                 num_ins: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_ins = num_ins
        self.num_outs = num_outs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                hidden_channels,
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, inputs):
        # 1. 通过lateral_convs为每个输入特征图进行1x1卷积
        laterals = []
        for i in range(self.num_ins):
            laterals.append(inputs[i])

        # 2. 自顶向下的融合：从最后一个层开始
        # 先将最后一个特征图的尺寸进行存储
        feature_maps = [laterals[-1]]

        # 从倒数第二个开始，进行自顶向下融合
        for i in range(self.num_ins - 2, -1, -1):
            # 先上采样到当前特征图尺寸
            lateral = laterals[i]
            upsampled = F.interpolate(feature_maps[-1], size=lateral.shape[2:], mode='nearest')

            # 将上采样的特征图与当前特征图进行融合
            fused = lateral + upsampled
            feature_maps.append(fused)

        # print(f"feature_maps[0].shape", feature_maps[0].shape)
        # 3. 反转特征图顺序，使得从小尺寸的特征图开始（与上采样过程一致）
        feature_maps = feature_maps[::-1]

        # 4. 每个特征图通过一个3x3卷积进行进一步处理
        outs = []
        for i in range(self.num_ins):
            outs.append(self.fpn_convs[i](feature_maps[i]))

        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        # print(outs[-1].shape)

        return tuple(outs)


#
class EMA(nn.Module):
    def __init__(self, channels, factor=128):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

@MODELS.register_module()
class USISFeatureAggregator(BaseModule):
    def __init__(
            self,
            in_channels,
            select_layers,
            hidden_channels=64,
            mean_alpha=0.8,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.select_layers = select_layers
        self.alpha = mean_alpha
        self.ca_adapter = ColorAttentionAdapter(in_channels[0], 0.0625, act_layer=nn.ReLU)
        # 调整 ResNet 特征的通道到 hidden_channels
        self.adjust1 = nn.Sequential(
            nn.Conv2d(256, hidden_channels, 1),  # 通道数调整，尺寸保持不变
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.adjust2 = nn.Sequential(
            nn.Conv2d(512, hidden_channels, 1),  # 通道数调整，尺寸保持不变
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.adjust3 = nn.Sequential(
            nn.Conv2d(1024, hidden_channels, 1),  # 通道数调整，尺寸保持不变
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )
        self.adjust4 = nn.Sequential(
            nn.Conv2d(2048, hidden_channels, 1),  # 通道数调整，尺寸保持不变
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # 调整 ViT 特征的通道到 hidden_channels
        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                MultiScaleConv(self.in_channels[i_layer], hidden_channels)
            )

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels,
                               hidden_channels, 2, 2),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels,
                               hidden_channels, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels,
                               hidden_channels, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # 轻量级通道加权融合模块
        self.fusion_weights = nn.ModuleList()
        for _ in self.select_layers:
            self.fusion_weights.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(),
                    nn.Conv2d(hidden_channels, 1, 3, padding=1),
                    nn.Sigmoid(),
                )
            )

        # # 最终将通道数调整到 out_channels
        # self.fusion_conv = nn.Sequential(
        #     nn.Conv2d(hidden_channels, out_channels, 1),  # 将 hidden_channels 调整到 out_channels
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        # )
        self.EMA1 = EMA(256, 32)
        self.EMA2 = EMA(512, 64)
        self.EMA3 = EMA(1024, 128)
        self.EMA4 = EMA(2048, 256)

    def forward(self, vit_feature, resnet_features):
        # 调整 ViT 特征形状为 [B, C, H, W]
        vit_feature = [einops.rearrange(x, 'b h w c -> b c h w') for x in vit_feature]

        # 1. 调整 ResNet 特征的通道到 hidden_channels
        adjusted_resnet_features = []
        for idx, resnet_feature in enumerate(resnet_features):
            if idx == 0:
                resnet_feature = self.EMA1(resnet_feature)
                adjusted_resnet_features.append(self.adjust1(resnet_feature))
            elif idx == 1:
                resnet_feature = self.EMA2(resnet_feature)
                adjusted_resnet_features.append(self.adjust2(resnet_feature))
            elif idx == 2:
                resnet_feature = self.EMA3(resnet_feature)
                adjusted_resnet_features.append(self.adjust3(resnet_feature))
            elif idx == 3:
                resnet_feature = self.EMA4(resnet_feature)
                adjusted_resnet_features.append(self.adjust4(resnet_feature))

        #调整 ViT 特征的通道数到 hidden_channels
        features = []
        for idx, i_layer in enumerate(self.select_layers):
            # 调整 ViT 特征的通道数到 hidden_channels
            vit_feature_adjusted = self.downconvs[idx](vit_feature[i_layer] * self.ca_adapter(vit_feature[i_layer]))
            # 根据索引选择应用不同的 FPN 层
            if idx == 0:
                vit_feature_adjusted = self.fpn1(vit_feature_adjusted)
            elif idx == 1:
                vit_feature_adjusted = self.fpn2(vit_feature_adjusted)
            elif idx == 2:
                vit_feature_adjusted = self.fpn3(vit_feature_adjusted)
            elif idx == 3:
                vit_feature_adjusted = self.fpn4(vit_feature_adjusted)
            features.append(vit_feature_adjusted)

        # 3. 轻量级通道加权融合
        fused_features = []
        for resnet_feature, vit_feature, weight_generator in zip(adjusted_resnet_features, features, self.fusion_weights):
            # # 将 ResNet 和 ViT 特征对齐到相同分辨率
            # if resnet_feature.shape[-2:] != vit_feature.shape[-2:]:
            #     vit_feature = F.interpolate(vit_feature, size=resnet_feature.shape[-2:], mode='bilinear', align_corners=False)
            # 生成门控图
            gate_input = torch.cat([resnet_feature, vit_feature], dim=1)
            # 生成权重
            weight = weight_generator(gate_input)
            # print(weight)
            # 融合特征
            fused_feature = resnet_feature * (1 - weight) + vit_feature * weight
            fused_features.append(fused_feature)

        return fused_features


@MODELS.register_module()
class USISPrompterAnchorRoIPromptHead(StandardRoIHead):
    def __init__(
            self,
            with_extra_pe=False,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

        if with_extra_pe:
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(
                num_feats=out_channels // 2,
                normalize=True,
            )
            self.extra_pe = SinePositionalEncoding(**positional_encoding)

    def _mask_forward(self,
                      x: Tuple[Tensor],
                      rois: Tensor = None,
                      pos_inds: Optional[Tensor] = None,
                      bbox_feats: Optional[Tensor] = None,
                      image_embeddings=None,
                      image_positional_embeddings=None,
                      ) -> dict:
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds, iou_predictions = self.mask_head(
            mask_feats,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            roi_img_ids=rois[:, 0] if rois is not None else None,
        )
        mask_results = dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
        return mask_results

    def mask_loss(self, x: Tuple[Tensor],
                  sampling_results: List[SamplingResult], bbox_feats: Tensor,
                  batch_gt_instances: InstanceList,
                  image_embeddings=None,
                  image_positional_embeddings=None,
                  ) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            if len(pos_rois) == 0:
                print('no pos rois')
                return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
            mask_results = self._mask_forward(
                x, pos_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results['mask_preds'],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)

        mask_results.update(loss_mask=mask_loss_and_target['loss_mask'])
        return mask_results

    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList,
             batch_data_samples: List[DetDataSample],
             # extra inputs
             image_embeddings=None,
             image_positional_embeddings=None,
             ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        if hasattr(self, 'extra_pe'):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop('bboxes')

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i],
                batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(
                x, sampling_results, bbox_results['bbox_feats'], batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
            losses.update(mask_results['loss_mask'])

        return losses

    def predict_mask(
            self,
            x: Tuple[Tensor],
            batch_img_metas: List[dict],
            results_list: InstanceList,
            rescale: bool = False,
            image_embeddings=None,
            image_positional_embeddings=None,
    ) -> InstanceList:

        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type='mask',
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary)
            return results_list

        mask_results = self._mask_forward(
            x, mask_rois,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings)

        mask_preds = mask_results['mask_preds']
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        # TODO: Handle the case where rescale is false
        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale)
        return results_list

    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False,
                # extra inputs
                image_embeddings=None,
                image_positional_embeddings=None,
                ) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        if hasattr(self, 'extra_pe'):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode='bilinear', align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale)

        if self.with_mask:
            results_list = self.predict_mask(
                x, batch_img_metas, results_list, rescale=rescale,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        return results_list


@MODELS.register_module()
class USISPrompterAnchorMaskHead(FCNMaskHead, BaseModule):
    def __init__(
            self,
            mask_decoder,
            in_channels,
            roi_feat_size=14,
            per_pointset_point=5,
            with_sincos=True,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
            class_agnostic=False,
            loss_mask: ConfigType = dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            init_cfg=None,
            *args,
            **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type='USISSamPromptEncoder',
            hf_pretrain_name=copy.deepcopy(mask_decoder.get('hf_pretrain_name')),
            init_cfg=copy.deepcopy(mask_decoder.get('init_cfg')),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()
        self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed

        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels * roi_feat_size ** 2 // 4, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point)
        )

        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def forward(self,
                x,
                image_embeddings,
                image_positional_embeddings,
                roi_img_ids=None,
                ):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        point_embedings = self.point_emb(x)
        point_embedings = einops.rearrange(point_embedings, 'b (n c) -> b n c', n=self.per_pointset_point)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]

        # (B * N_set), N_point, C
        sparse_embeddings = point_embedings.unsqueeze(1)
        num_roi_per_image = torch.bincount(roi_img_ids.long())
        # deal with the case that there is no roi in an image
        num_roi_per_image = torch.cat([num_roi_per_image,
                                       torch.zeros(img_bs - len(num_roi_per_image), device=num_roi_per_image.device,
                                                   dtype=num_roi_per_image.dtype)])

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(roi_bs, -1, image_embedding_size[0],
                                                                                 image_embedding_size[1])
        # get image embeddings with num_roi_per_image
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        # assert False, f"{low_res_masks.shape}"
        return low_res_masks, iou_predictions

    def get_targets(self, sampling_results: List[SamplingResult],
                    batch_gt_instances: InstanceList,
                    rcnn_train_cfg: ConfigDict) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets_list = []
        mask_size = rcnn_train_cfg.mask_size
        device = pos_proposals[0].device
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets

    def loss_and_target(self, mask_preds: Tensor,
                        sampling_results: List[SamplingResult],
                        batch_gt_instances: InstanceList,
                        rcnn_train_cfg: ConfigDict) -> dict:
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg)

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        # resize to mask_targets size
        mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode='bilinear', align_corners=False)

        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets,
                                           pos_labels)
        loss['loss_mask'] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)

    def _predict_by_feat_single(self,
                                mask_preds: Tensor,
                                bboxes: Tensor,
                                labels: Tensor,
                                img_meta: dict,
                                rcnn_test_cfg: ConfigDict,
                                rescale: bool = False,
                                activate_map: bool = False) -> Tensor:
        scale_factor = bboxes.new_tensor(img_meta['scale_factor']).repeat(
            (1, 2))
        img_h, img_w = img_meta['ori_shape'][:2]
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = F.interpolate(mask_preds, size=img_meta['batch_input_shape'], mode='bilinear',
                                align_corners=False).squeeze(1)

        scale_factor_w, scale_factor_h = img_meta['scale_factor']
        ori_rescaled_size = (img_h * scale_factor_h, img_w * scale_factor_w)
        im_mask = im_mask[:, :int(ori_rescaled_size[0]), :int(ori_rescaled_size[1])]

        h, w = img_meta['ori_shape']
        im_mask = F.interpolate(im_mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

        if threshold >= 0:
            im_mask = im_mask >= threshold
        else:
            # for visualization and debugging
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask