# Differences with paper
# Padded convolutions to simplify data loading

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv  = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3, stride=1, padding=1,bias=False), #same convolution
            nn.BatchNorm2d(out_channels), # Bias = False due to batchnorm, The mean subtraction cancels out the effect of any bias. 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3, stride=1, padding=1,bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
        )
        
    def forward(self,x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self,
                 in_channels=3, out_channels=1, features=[64,128,256,512]): # in paper out_channels is 2
        super(UNET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels,out_channels=feature))
            in_channels = feature

        # Down part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2, # number of channels by half,  spatial dimensions are doubled
                )
            ) 
            self.ups.append(DoubleConv(feature*2, feature)) # and up double conv

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # 1x1 convolution
    
    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)                # Upsampling (ConvTranspose2d)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) # just width and heigth
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x           = self.ups[idx+1](concat_skip)  # Apply DoubleConv
 
        return self.final_conv(x)
    
