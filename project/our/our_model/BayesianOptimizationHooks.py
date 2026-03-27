import torch
import optuna
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from optuna import Trial
import logging
from collections import defaultdict

@HOOKS.register_module()
class BayesianOptimizationHook(Hook):
    def __init__(self, epoch_interval=7442/2*7, optimize_target='no_mask_embed', pbounds=None):
        super().__init__()
        self.epoch_interval = epoch_interval
        self.optimize_target = optimize_target
        self.iter_count = 0

        if pbounds is None:
            self.group_size = 16
            self.num_groups = 256 // self.group_size
            self.pbounds = {f'scaling_group_{i}': (1,3) for i in range(self.num_groups)}
        else:
            self.pbounds = pbounds

    def after_train_iter(self, runner, **kwargs):
        self.iter_count = runner.iter
        if self.iter_count % self.epoch_interval != 0:
            return
        if self.optimize_target == 'no_mask_embed':
            self.optimize_no_mask_embed(runner)

    def before_train_iter(self, runner, **kwargs):
        # 在训练迭代前保存原始参数，以便在优化时使用
        pass

    def optimize_no_mask_embed(self, runner):
        # 保存原始参数
        original_weight = runner.model.roi_head.mask_head.no_mask_embed.weight.data.clone()
        device = original_weight.device

        # 将模型设置为评估模式
        runner.model.eval()

        def objective(trial):
            with torch.no_grad():  # 完全不需要梯度
                # 获取缩放因子
                scaling_factors = [trial.suggest_float(f'scaling_group_{i}', 1, 3) for i in range(self.num_groups)]

                # 应用缩放因子
                modified_weight = original_weight.clone()
                for i, factor in enumerate(scaling_factors):
                    start = i * self.group_size
                    end = start + self.group_size
                    modified_weight[:, start:end] *= factor

                # 临时修改参数
                runner.model.roi_head.mask_head.no_mask_embed.weight.data = modified_weight

                try:
                    # 使用整个验证集计算平均loss
                    total_loss = 0.0
                    valid_batches = 0

                    # 遍历整个验证集
                    for data_batch in val_dataloader:
                        try:
                            # 使用MMDetection的标准数据处理流程
                            if hasattr(runner.model, 'data_preprocessor'):
                                processed_data = runner.model.data_preprocessor(data_batch)
                            else:
                                processed_data = data_batch

                            # 调用模型的loss模式进行前向传播
                            outputs = runner.model(**processed_data, mode='loss')

                            if outputs and isinstance(outputs, dict):
                                # 优先使用mask loss
                                if 'loss_mask' in outputs:
                                    loss_val = outputs['loss_mask']
                                    if isinstance(loss_val, torch.Tensor):
                                        total_loss += loss_val.item()
                                        valid_batches += 1
                                # 备用：使用总损失
                                elif 'loss' in outputs:
                                    loss_val = outputs['loss']
                                    if isinstance(loss_val, torch.Tensor):
                                        total_loss += loss_val.item()
                                        valid_batches += 1

                        except Exception:
                            continue  # 跳过失败的batch

                    # 返回平均损失
                    if valid_batches > 0:
                        avg_loss = total_loss / valid_batches
                        return avg_loss
                    else:
                        return float('inf')

                except Exception:
                    return float('inf')
                finally:
                    # 恢复原始参数
                    runner.model.roi_head.mask_head.no_mask_embed.weight.data = original_weight

        # 运行优化
        val_dataloader = runner.val_dataloader
        print(f"Starting optimization of no_mask_embed with {len(val_dataloader)} validation batches...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        # 应用最佳参数
        best_params = study.best_params
        best_factors = [best_params[f'scaling_group_{i}'] for i in range(self.num_groups)]

        optimized_weight = original_weight.clone()
        for i, factor in enumerate(best_factors):
            start = i * self.group_size
            end = start + self.group_size
            optimized_weight[:, start:end] *= factor

        runner.model.roi_head.mask_head.no_mask_embed.weight.data = optimized_weight

        # 恢复模型为训练模式
        runner.model.train()

        # 保存优化后的权重
        import os
        work_dir = getattr(runner, 'work_dir', 'work_dirs')
        os.makedirs(work_dir, exist_ok=True)
        save_path = os.path.join(work_dir, 'optimized_no_mask_embed.pth')
        torch.save(runner.model.state_dict(), save_path)

        print(f"✓ Optimization completed!")
        print(f"  Best loss: {study.best_value:.4f}")
        print(f"  Weights saved to: {save_path}")
        print(f"  Applied {len(best_factors)} scaling factors")

        # 清空优化器状态以应用新参数
        runner.optim_wrapper.optimizer.zero_grad()