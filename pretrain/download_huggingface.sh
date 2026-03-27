#!/usr/bin/env bash

NAME=$1
SAVEDIR=$2

export HF_ENDPOINT="https://hf-mirror.com"

# Download SAM model from Hugging Face
echo "Downloading SAM model from Hugging Face..."
huggingface-cli download \
--resume-download $NAME \
--local-dir $SAVEDIR \
--exclude *.safetensors *.h5 \
${@:3}

# Download ResNet50 pretrained weights
echo "Downloading ResNet50 pretrained weights..."
RESNET_WEIGHTS_URL="https://download.pytorch.org/models/resnet50-19c8e357.pth"
RESNET_SAVE_PATH="pretrain/resnet50-19c8e357.pth"

# Create pretrain directory if it doesn't exist
mkdir -p pretrain

# Download ResNet50 weights
if command -v wget >/dev/null 2>&1; then
    wget -O "$RESNET_SAVE_PATH" "$RESNET_WEIGHTS_URL"
elif command -v curl >/dev/null 2>&1; then
    curl -L -o "$RESNET_SAVE_PATH" "$RESNET_WEIGHTS_URL"
else
    echo "Error: Neither wget nor curl is installed. Please install one of them to download ResNet50 weights."
    exit 1
fi

echo "Download completed!"
echo "SAM model saved to: $SAVEDIR"
echo "ResNet50 weights saved to: $RESNET_SAVE_PATH"
