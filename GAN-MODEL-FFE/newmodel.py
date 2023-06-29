import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random

resolution=[256,128, 64, 32, 16, 8]

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.downsample=downsample
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,stride=1, padding="same", bias=False)
        self.conv2=nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,stride=2,bias=False)
        self.bn= nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size,stride=1, padding="same", bias=False)
        
        if downsample:
            self.downsampl=nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        else:
            self.downsampl= None

    def forward(self, x):
        identity = x
        if self.downsample:
          out=self.conv2(x)
          out=torch.nn.functional.pad(out, pad=(0,1,0,1,0,1), mode='constant', value=0)
        else:
          out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsampl is not None:
            identity = self.downsampl(x)
        
        out=torch.add(out,identity)
        out=self.bn(out)
        out = self.relu(out)

        return out


class AdaIN(nn.Module):
    def __init__(self, resolution):
        super(AdaIN, self).__init__()
        self.resolution=resolution
        self.gamma = nn.Sequential(
            nn.Linear(256, resolution),
            nn.ReLU(),
            nn.Linear(resolution, resolution),
            nn.ReLU(),
            nn.Linear(resolution, resolution),
            nn.ReLU(),
            nn.Linear(resolution, resolution),
            nn.ReLU()
        )
        self.beta = nn.Sequential(
            nn.Linear(256, resolution),
            nn.ReLU(),
            nn.Linear(resolution, resolution),
            nn.ReLU(),
            nn.Linear(resolution, resolution),
            nn.ReLU(),
            nn.Linear(resolution, resolution),
            nn.ReLU()
        )

    def forward(self, content, style):
        gamma = self.gamma(style)
        gamma = gamma.view(-1, self.resolution, 1, 1, 1)
        
        beta = self.beta(style)
        beta = beta.view(-1, self.resolution, 1, 1,1)
        
        content_shape=content.shape
        reduction_axes = tuple(range(0, len(content_shape)))
      
        mean = torch.mean(content, dim=reduction_axes, keepdim=True)
        stddev = torch.std(content, dim=reduction_axes, keepdim=True) + 1e-3
        normed = (content - mean) / stddev
        
        return normed * gamma + beta

class Encoder(nn.Module):
    def __init__(self,incha,outcha):
        super(Encoder, self).__init__()
        
        self.conv1=nn.Conv3d(incha,outcha*1, kernel_size=3,stride=1, padding="same", bias=False)
        self.relu1=nn.ReLU()
        self.maxpool1= nn.MaxPool3d(2)

        self.conv2=nn.Conv3d(outcha*1,outcha*2, kernel_size=3,stride=1, padding="same", bias=False)
        self.relu2=nn.ReLU()
        self.maxpool2= nn.MaxPool3d(2)

        self.conv3=nn.Conv3d(outcha*2,outcha*4, kernel_size=3,stride=1, padding="same", bias=False)
        self.relu3=nn.ReLU()
        self.maxpool3= nn.MaxPool3d(2)

        self.conv4=nn.Conv3d(outcha*4,outcha*8, kernel_size=3,stride=1, padding="same", bias=False)
        self.relu4=nn.ReLU()
        self.maxpool4= nn.MaxPool3d(2)

        self.res=ResidualBlock(outcha*8,256,downsample=True)
        
        self.fc1=nn.Linear(2,64)
        self.fc2=nn.Linear(64,256)

        self.res1=ResidualBlock(256,256)
        self.res2=ResidualBlock(256,256)
        self.res3=ResidualBlock(256,256)
        self.res4=ResidualBlock(256,256)
        self.res5=ResidualBlock(256,256)
        
        self.dense1=nn.Linear(256,256)
        self.leaky1=nn.LeakyReLU(negative_slope=0.2)
        self.dense2=nn.Linear(256,256)
        self.leaky2=nn.LeakyReLU(negative_slope=0.2)
        self.dense3=nn.Linear(256,256)
        self.leaky3=nn.LeakyReLU(negative_slope=0.2)
        self.dense4=nn.Linear(256,256)
        self.leaky4=nn.LeakyReLU(negative_slope=0.2)
        self.dense5=nn.Linear(256,256)
        self.leaky5=nn.LeakyReLU(negative_slope=0.2)
        

    def forward(self, inputs,inp_variables):
    
        output=self.conv1(inputs)
        output=self.relu1(output)
        output=self.maxpool1(output)
     
        output=self.conv2(output)
        output=self.relu2(output)
        output=self.maxpool2(output)

        output=self.conv3(output)
        output=self.relu3(output)
        output=self.maxpool3(output)

        output=self.conv4(output)
        output=self.relu4(output)
        output=self.maxpool4(output)

        output=self.res(output)

        carspeed=self.fc1(inp_variables)
        carspeed=self.fc2(carspeed)
        carspeed=torch.tile(carspeed,(1,96))
        carspeed=torch.reshape(carspeed,(1,256,12,4,2))
        
        
        latent_vector=torch.add(carspeed,output)
      
        ress1=self.res1(latent_vector)
        avg1=torch.mean(ress1,(2,3,4))

        ress2=self.res2(ress1)
        avg2=torch.mean(ress2,(2,3,4))

        ress3=self.res3(ress2)
        avg3=torch.mean(ress3,(2,3,4))
       
        ress4=self.res4(ress3)
        avg4=torch.mean(ress4,(2,3,4))
       
        ress5=self.res5(ress4)
        avg5=torch.mean(ress5,(2,3,4))
        
        style=self.dense1(avg1)
        style=self.leaky1(style)
        style=torch.add(style,avg2)

        style=self.dense2(style)
        style=self.leaky2(style)
        style=torch.add(style,avg3)

        style=self.dense3(style)
        style=self.leaky3(style)
        style=torch.add(style,avg4)

        style=self.dense4(style)
        style=self.leaky4(style)
        style=torch.add(style,avg5)

        style=self.dense5(style)
        style=self.leaky5(style)
        
        
        return latent_vector,style




class Decoder(nn.Module):
    def __init__(self,resolution):
        super(Decoder,self).__init__()
        # Style Mapping
        self.ada11 = AdaIN(resolution[0])
        self.conv11 =nn.Conv3d(256, 256, kernel_size=3, padding="same",bias=False)
        self.relu11= nn.ReLU(inplace=True)
        self.ada12=AdaIN(resolution[0])
        
        self.us1=nn.Upsample(scale_factor=2)
        self.conv21=nn.Conv3d(resolution[0],resolution[1], kernel_size=3, padding="same",bias=False)
        self.relu21= nn.ReLU(inplace=True)
        self.ada21= AdaIN(resolution[1])
        self.conv22=nn.Conv3d(resolution[1],resolution[1], kernel_size=3, padding="same",bias=False)
        self.relu22= nn.ReLU(inplace=True)
        self.ada22= AdaIN(resolution[1])

        self.us2=nn.Upsample(scale_factor=2)
        self.conv31=nn.Conv3d(resolution[1],resolution[2], kernel_size=3, padding="same",bias=False)
        self.relu31= nn.ReLU(inplace=True)
        self.ada31= AdaIN(resolution[2])
        self.conv32=nn.Conv3d(resolution[2],resolution[2], kernel_size=3, padding="same",bias=False)
        self.relu32= nn.ReLU(inplace=True)
        self.ada32= AdaIN(resolution[2])

        self.us3=nn.Upsample(scale_factor=2)
        self.conv41=nn.Conv3d(resolution[2],resolution[3], kernel_size=3, padding="same",bias=False)
        self.relu41= nn.ReLU(inplace=True)
        self.ada41= AdaIN(resolution[3])
        self.conv42=nn.Conv3d(resolution[3],resolution[3], kernel_size=3, padding="same",bias=False)
        self.relu42= nn.ReLU(inplace=True)
        self.ada42= AdaIN(resolution[3])

        self.us4=nn.Upsample(scale_factor=2)
        self.conv51=nn.Conv3d(resolution[3],resolution[4], kernel_size=3, padding="same",bias=False)
        self.relu51= nn.ReLU(inplace=True)
        self.ada51= AdaIN(resolution[4])
        self.conv52=nn.Conv3d(resolution[4],resolution[4], kernel_size=3, padding="same",bias=False)
        self.relu52= nn.ReLU(inplace=True)
        self.ada52= AdaIN(resolution[4])

        self.us5=nn.Upsample(scale_factor=2)
        self.conv61=nn.Conv3d(resolution[4],resolution[5], kernel_size=3, padding="same",bias=False)
        self.relu61= nn.ReLU(inplace=True)
        self.ada61= AdaIN(resolution[5])
        self.conv62=nn.Conv3d(resolution[5],resolution[5], kernel_size=3, padding="same",bias=False)
        self.relu62= nn.ReLU(inplace=True)
        self.ada62= AdaIN(resolution[5])

        self.conv7=nn.Conv3d(resolution[5],1, kernel_size=3, padding="same",bias=False)
        self.tan=nn.Tanh()

    def forward(self,latent_vector, style):
        output=self.ada11(latent_vector,style)
        output=self.conv11(output)
        output=self.relu11(output)
        output=self.ada12(output,style)
       
        output=self.us1(output)
        output=self.conv21(output)
        output=self.relu21(output)
        output=self.ada21(output,style)
        
        output=self.conv22(output)
        output=self.relu22(output)
        output=self.ada22(output,style)

        output=self.us2(output)
        output=self.conv31(output)
        output=self.relu31(output)
        output=self.ada31(output,style)
        output=self.conv32(output)
        output=self.relu32(output)
        output=self.ada32(output,style)

        output=self.us3(output)
        output=self.conv41(output)
        output=self.relu41(output)
        output=self.ada41(output,style)
        output=self.conv42(output)
        output=self.relu42(output)
        output=self.ada42(output,style)

        output=self.us4(output)
        output=self.conv51(output)
        output=self.relu51(output)
        output=self.ada51(output,style)
        output=self.conv52(output)
        output=self.relu52(output)
        output=self.ada52(output,style)

        output=self.us5(output)
        output=self.conv61(output)
        output=self.relu61(output)
        output=self.ada61(output,style)
        output=self.conv62(output)
        output=self.relu62(output)
        output=self.ada62(output,style)

        output=self.conv7(output)
        output=self.tan(output)
        return output

