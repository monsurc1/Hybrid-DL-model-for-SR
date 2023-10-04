import torch
import torch.nn as nn

class AttentionFusionModule(nn.Module):
    def __init__(self):
        super(AttentionFusionModule, self).__init__()
        self.first_time = True
    
    def runtime_init(self, x, fts):
        self.attention_conv = []

        # Convolution to generate attention maps
        for ft in fts: 
            self.attention_conv.append(nn.Conv2d(ft.shape[-3], x.shape[-3], kernel_size=1))
        
        self.first_time = False

    def forward(self, x, fts):

        if self.first_time: self.runtime_init(x, fts)
        
        attention_maps = []
        # Generate attention maps using the feature tensor
        for i, ft in enumerate(fts):
            attention_maps.append(self.attention_conv[i](ft))
        
        # Upsample the attention maps to match the size of x
        attention_maps_upsampled = torch.zeros(x.shape)
        for i in range(len(fts)):
            tmp = nn.functional.interpolate(attention_maps[i], size=x.shape[-2:], mode='bilinear', align_corners=True)
            
            # Apply softmax to make the attention maps values between 0 and 1
            attention_maps_upsampled += torch.softmax(tmp, dim=1) / len(fts)
        
        # Multiply the original image with the attention maps
        fused = x * attention_maps_upsampled
        
        return fused

class StackFusionModule(nn.Module):
    def __init__(self):
        super(StackFusionModule, self).__init__()
        self.first_time = True
    
    def runtime_init(self, x, fts):
        self.attention_conv = []

        # Convolution to generate attention maps
        for ft in fts: 
            self.attention_conv.append(nn.Conv2d(ft.shape[-3], x.shape[-3], kernel_size=1))
        
        self.conv1x1 = nn.Conv2d(x.shape[-3]*2, x.shape[-3], kernel_size=1)

        self.first_time = False

    def forward(self, x, fts):

        if self.first_time: self.runtime_init(x, fts)
        
        attention_maps = []
        # Generate attention maps using the feature tensor
        for i, ft in enumerate(fts):
            attention_maps.append(self.attention_conv[i](ft))
        
        # Upsample the attention maps to match the size of x
        attention_maps_upsampled = torch.zeros(x.shape)
        for i in range(len(fts)):
            tmp = nn.functional.interpolate(attention_maps[i], size=x.shape[-2:], mode='bilinear', align_corners=True)
            
            # Apply softmax to make the attention maps values between 0 and 1
            attention_maps_upsampled += torch.softmax(tmp, dim=1) / len(fts)
        
        # Stack the original image with the attention maps
        fused = torch.cat([x, attention_maps_upsampled], dim=-3)

        reduced_fused = self.conv1x1(fused)
        
        return reduced_fused
