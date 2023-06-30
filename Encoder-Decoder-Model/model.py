import torch
import torch.nn as nn
import torch.nn.functional as F

###ENCODER
class Encoder(nn.Module):
    def __init__(self,batch_size):
        super(Encoder, self).__init__()

        self.relu=nn.ReLU()
        self.maxpool= nn.MaxPool3d(2, stride=2)

        self.conv3d1=nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, padding="same")
        self.batch_norm1=nn.BatchNorm3d(8)
        
        self.conv3d2=nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, padding="same")
        self.batch_norm2=nn.BatchNorm3d(16)

        self.conv3d3=nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding="same")
        self.batch_norm3=nn.BatchNorm3d(32)

        self.conv3d4=nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.batch_norm4=nn.BatchNorm3d(64)

        self.conv3d5=nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
        self.batch_norm5=nn.BatchNorm3d(128)
        
        self.linear1=nn.Linear(batch_size*(12288+2),batch_size*(12288+2))
        self.linear2=nn.Linear(batch_size*(12288+2),batch_size*12288)
        self.regularise=nn.Dropout(p=0.2)
        
    def forward(self, inputs,inp_variables):
   
        outputs=self.conv3d1(inputs)
        outputs=self.relu(outputs)
        outputs=self.batch_norm1(outputs)
        outputs=self.maxpool(outputs)

        outputs=self.conv3d2(outputs)
        outputs=self.relu(outputs)
        outputs=self.batch_norm2(outputs)
        outputs=self.maxpool(outputs)

        outputs=self.conv3d3(outputs)
        outputs=self.relu(outputs)
        outputs=self.batch_norm3(outputs)
        outputs=self.maxpool(outputs)

        outputs=self.conv3d4(outputs)
        outputs=self.relu(outputs)
        outputs=self.batch_norm4(outputs)
        outputs=self.maxpool(outputs)

        outputs=self.conv3d5(outputs)
        outputs=self.relu(outputs)
        outputs=self.batch_norm5(outputs)
        outputs=self.maxpool(outputs)

        trans_shape = outputs.shape
        
        outputs=torch.reshape(outputs,(-1,1))
        inp_variables=torch.reshape(inp_variables,(-1,1))
        outputs=torch.cat((outputs,inp_variables), dim=0)
        outputs=torch.flatten(outputs)
      
        outputs=self.linear1(outputs)
        outputs=self.linear2(outputs)
        outputs=self.relu(outputs)
        outputs=self.regularise(outputs)
        
        latent_shape = outputs.shape
        
        return outputs,trans_shape,latent_shape

###DECODER
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.upsample=nn.Upsample(scale_factor=2)
        self.relu=nn.ReLU()

        self.conv1=nn.Conv3d(128, 128, kernel_size=3, padding="same")
        self.batch_norm1 = nn.BatchNorm3d(128)

        self.conv2=nn.Conv3d(128, 64, kernel_size=3, padding="same")
        self.batch_norm2=nn.BatchNorm3d(64)

        self.conv3=nn.Conv3d(64, 32, kernel_size=3, padding="same")
        self.batch_norm3=nn.BatchNorm3d(32)

        self.conv4=nn.Conv3d(32, 16, kernel_size=3, padding="same")
        self.batch_norm4=nn.BatchNorm3d(16)

        self.conv5=nn.Conv3d(16, 8, kernel_size=3, padding="same")
        self.batch_norm5=nn.BatchNorm3d(8)
        
        self.conv6=nn.Conv3d(8, 1, kernel_size=3, padding="same")
        self.sigmoid=nn.Sigmoid()
    def forward(self,latent,trans_shape):
        
        outputs=latent.view(-1, trans_shape[1], trans_shape[2], trans_shape[3], trans_shape[4])

        outputs=self.upsample(outputs)
        outputs= self.relu(self.conv1(outputs))
        outputs= self.batch_norm1(outputs)
     
        outputs=self.upsample(outputs)
        outputs = self.relu(self.conv2(outputs))
        outputs = self.batch_norm2(outputs)

        outputs=self.upsample(outputs)
        outputs = self.relu(self.conv3(outputs))
        outputs = self.batch_norm3(outputs)

        outputs=self.upsample(outputs)
        outputs =self.relu(self.conv4(outputs))
        outputs = self.batch_norm4(outputs)

        outputs=self.upsample(outputs)
        outputs =self.relu(self.conv5(outputs))
        outputs = self.batch_norm5(outputs)
        
        outputs = self.sigmoid(self.conv6(outputs))
       
        return outputs