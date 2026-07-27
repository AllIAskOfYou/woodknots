import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DINOv3ConvNextBackbone

class DINOv3FPN(nn.Module):
    def __init__(self, out_channels=256, num_classes=1):
        super().__init__()
        
        self.encoder = DINOv3ConvNextBackbone.from_pretrained(
            "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            out_indices=[1, 2, 3, 4]
        )
        
        # ConvNeXt-Tiny channel counts for stages 0, 1, 2, 3
        stage_channels = [96, 192, 384, 768]
        
        # Create lateral layers that map EACH stage to out_channels
        self.lateral_layers = nn.ModuleList([
            nn.Conv2d(ch, out_channels, kernel_size=1) for ch in stage_channels
        ])
        
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in stage_channels
        ])
        
        self.head = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.encoder(x).feature_maps # list of 4 tensors
        
        # 1. Start from the deepest feature map (stage 3, 768 channels)
        # Lateral layer 3 expects 768 channels - Correct.
        
        # for i, feat in enumerate(features):
        #    print(f"Stage {i} shape: {feat.shape}")
        p = self.lateral_layers[3](features[3])
        
        # 2. Iterate backwards from stage 2 down to 0
        for i in range(2, -1, -1):
            # Upsample the path from above
            p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Add lateral connection from current stage 'i'
            # lateral_layers[i] expects stage_channels[i] (e.g., 384 for stage 2)
            p = p + self.lateral_layers[i](features[i])
            
            # Apply smoothing
            p = self.smooth_layers[i](p)
            
        # 3. Final Head
        out = F.interpolate(p, size=input_size, mode='bilinear', align_corners=False)
        return self.head(out)
    

from segmentation_models_pytorch.decoders.manet.model import MAnetDecoder

class DINOv3MAnet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        # 1. Your DINOv3 backbone
        self.encoder = DINOv3ConvNextBackbone.from_pretrained(
            "facebook/dinov3-convnext-large-pretrain-lvd1689m",
            out_indices=[1, 2, 3, 4]
        )
        # SMP needs to know the number of channels for each stage
        # These match ConvNeXt-Tiny output channels: [96, 192, 384, 768]
        # encoder_channels = [96, 192, 384, 768]
        encoder_channels = [192, 384, 768, 1536]  # For ConvNeXt-Large
        
        # 2. Use SMP's internal decoder
        decoder_channels = (256, 128, 64, 32)
        self.decoder = MAnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            n_blocks=4, # Matches our 4 stages
            pab_channels=64,
        )
        
        # 3. Final Head
        self.segmentation_head = nn.Sequential(
            # This 3x3 conv cleans up the interpolation artifacts
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # This 1x1 conv does the final classification
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def get_encoder_parameters(self, stages=[2, 3]):
        params = []
        for name, param in self.encoder.named_parameters():
            if any(f"stages.{stage}" in name for stage in stages):
                params.append(param)
        return params
    
    def unfreeze_encoder_stages(self, stages=[2, 3]):
        params = self.get_encoder_parameters(stages)
        for param in params:
            param.requires_grad = True
    
    def get_decoder_parameters(self):
        params = []
        params.extend(list(self.decoder.parameters()))
        params.extend(list(self.segmentation_head.parameters()))
        return params


    def forward(self, x):
        features = self.encoder(x).feature_maps 
        
        # Pass features through the SMP decoder
        decoder_output = self.decoder(features)

        upsampled_output = F.interpolate(decoder_output, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        return self.segmentation_head(upsampled_output)