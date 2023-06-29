import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure



###LOSS FUNCTION
class CombinedLoss(nn.Module):
    def __init__(self, loss_type='jaccard', smooth=1.):
        super(CombinedLoss, self).__init__()
        self.loss_type = loss_type
        self.smooth = smooth
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

    def forward(self, y_true, y_pred):
        y_true_f=y_true.view(-1).float()
        y_pred_f=y_pred.view(-1).float()
    
        intersection=torch.sum(y_true_f * y_pred_f)

        if self.loss_type=='jaccard':
            union=torch.sum(y_pred_f**2)+torch.sum(y_true_f**2)
        elif self.loss_type=='sorensen':
            union=torch.sum(y_pred_f)+torch.sum(y_true_f)
        else:
            raise ValueError("Unknown `loss_type`: %s" % self.loss_type)

        dice_loss=1-(2.*intersection+self.smooth)/(union+self.smooth)
        ssim_loss=1-torch.mean(self.ssim(y_pred,y_true))
        square_loss=torch.mean(torch.square(y_pred-y_true))

        return dice_loss + ssim_loss + square_loss


class AutoencoderLoss(nn.Module):
    def __init__(self):
        super( AutoencoderLoss, self).__init__()
        self.Loss=nn.MSELoss()
    def forward(self, target,pred):
        vx, vy, vz, p = target[0], target[1], target[2], target[3]
        reconstructed_vx, reconstructed_vy, reconstructed_vz, reconstructed_p = pred[0], pred[1], pred[2], pred[3]
        reconstruction_loss_vx =self.Loss(reconstructed_vx[0], vx)
        reconstruction_loss_vy =self.Loss(reconstructed_vy[0], vy)
        reconstruction_loss_vz =self.Loss(reconstructed_vz[0], vz)
        reconstruction_loss_p =self.Loss(reconstructed_p[0], p)
        total_loss = reconstruction_loss_vx + reconstruction_loss_vy + reconstruction_loss_vz + reconstruction_loss_p
        
        return total_loss

class SliceLoss(nn.Module):
    def __init__(self):
        super(SliceLoss, self).__init__()

    def forward(self, target, pred):
        vx, vy, vz, p ,mask= target[0], target[1], target[2], target[3],target[4]
        reconstructed_vx, reconstructed_vy, reconstructed_vz, reconstructed_p = pred[0], pred[1], pred[2], pred[3]
  
        length,width,height=vx.shape[1],vx.shape[2],vx.shape[3]
      
        slicelossl=0.0
        slicelossw=0.0
        slicelossh=0.0
 
        for i in range(length):
            diffx=torch.sum(torch.sub(reconstructed_vx[0,0,i,:,:],vx[0,i,:,:]))
            diffy=torch.sum(torch.sub(reconstructed_vy[0,0,i,:,:],vy[0,i,:,:]))
            diffz=torch.sum(torch.sub(reconstructed_vz[0,0,i,:,:],vz[0,i,:,:]))
    
            diffpress=torch.sum(torch.sub(reconstructed_p[0,0,i,:,:],p[0,i,:,:]))
            masksum=torch.sum(mask[0,i,:,:])
            slicelossl+=((diffx+diffy+diffz+diffpress)/masksum)
        slicelossl/=length
        for i in range(width):
            diffx=torch.sum(torch.sub(reconstructed_vx[0,0,:,i,:],vx[0,:,i,:]))
            diffy=torch.sum(torch.sub(reconstructed_vy[0,0,:,i,:],vy[0,:,i,:]))
            diffz=torch.sum(torch.sub(reconstructed_vz[0,0,:,i,:],vz[0,:,i,:]))
            diffpress=torch.sum(torch.sub(reconstructed_p[0,0,:,i,:],p[0,:,i,:]))
            masksum=torch.sum(mask[0,:,i,:])
            slicelossw+=((diffx+diffy+diffz+diffpress)/masksum)

        slicelossw/=width
        for i in range(height):
            diffx=torch.sum(torch.sub(reconstructed_vx[0,0,:,:,i],vx[0,:,:,i]))
            diffy=torch.sum(torch.sub(reconstructed_vy[0,0,:,:,i],vy[0,:,:,i]))
            diffz=torch.sum(torch.sub(reconstructed_vz[0,0,:,:,i],vz[0,:,:,i]))
            diffpress=torch.sum(torch.sub(reconstructed_p[0,0,:,:,i],p[0,:,:,i]))
            masksum=torch.sum(mask[0,:,:,i])
     
            slicelossh+=((diffx+diffy+diffz+diffpress)/masksum)
        slicelossh/=height
        totalloss=slicelossl+slicelossw+slicelossh
        
        return totalloss


class CombinedLoss1(nn.Module):
    def __init__(self, loss_type='jaccard', smooth=1.):
        super(CombinedLoss1, self).__init__()
        self.loss_type = loss_type
        self.smooth = smooth
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0)

    def forward(self,target, pred):
        vx, vy, vz, p ,mask= target[0], target[1], target[2], target[3],target[4]
        reconstructed_vx, reconstructed_vy, reconstructed_vz, reconstructed_p = pred[0], pred[1], pred[2], pred[3]
  
        length,width,height=vx.shape[1],vx.shape[2],vx.shape[3]
      
        slicelossl=0.0
        slicelossw=0.0
        slicelossh=0.0
 
        for i in range(length):
            diffx=torch.sum(torch.sub(vx[0,i,:,:],reconstructed_vx[0,0,i,:,:])**2)
            diffy=torch.sum(torch.sub(vy[0,i,:,:],reconstructed_vy[0,0,i,:,:])**2)
            diffz=torch.sum(torch.sub(vz[0,i,:,:],reconstructed_vz[0,0,i,:,:])**2)
    
            diffpress=torch.sum(torch.sub(p[0,i,:,:],reconstructed_p[0,0,i,:,:])**2)
            masksum=torch.sum(mask[0,i,:,:]**2)
            slicelossl+=((diffx+diffy+diffz+diffpress)/masksum)
        slicelossl/=length
        for i in range(width):
            diffx=torch.sum(torch.sub(vx[0,:,i,:],reconstructed_vx[0,0,:,i,:])**2)
            diffy=torch.sum(torch.sub(vy[0,:,i,:],reconstructed_vy[0,0,:,i,:])**2)
            diffz=torch.sum(torch.sub(vz[0,:,i,:],reconstructed_vz[0,0,:,i,:])**2)
            diffpress=torch.sum(torch.sub(p[0,:,i,:],reconstructed_p[0,0,:,i,:])**2)
            masksum=torch.sum(mask[0,:,i,:]**2)
            slicelossw+=((diffx+diffy+diffz+diffpress)/masksum)

        slicelossw/=width
        for i in range(height):
            diffx=torch.sum(torch.sub(vx[0,:,:,i],reconstructed_vx[0,0,:,:,i])**2)
            diffy=torch.sum(torch.sub(vy[0,:,:,i],reconstructed_vy[0,0,:,:,i])**2)
            diffz=torch.sum(torch.sub(vz[0,:,:,i],reconstructed_vz[0,0,:,:,i])**2)
            diffpress=torch.sum(torch.sub(p[0,:,:,i],reconstructed_p[0,0,:,:,i])**2)
            masksum=torch.sum(mask[0,:,:,i]**2)
     
            slicelossh+=((diffx+diffy+diffz+diffpress)/masksum)
        slicelossh/=height
        totalloss=slicelossl+slicelossw+slicelossh
        
        
        y_true=torch.stack([vx, vy, vz, p])
        y_true= y_true.view(-1, 4, 384, 128, 64)
        y_true_f=y_true.view(-1).float()
        y_pred=torch.stack([ reconstructed_vx, reconstructed_vy, reconstructed_vz, reconstructed_p ])
        y_pred= y_pred.view(-1, 4, 384, 128, 64)
        y_pred_f=y_pred.view(-1).float()
    
        intersection=torch.sum(y_true_f * y_pred_f)

        if self.loss_type=='jaccard':
            union=torch.sum(y_pred_f**2)+torch.sum(y_true_f**2)
        elif self.loss_type=='sorensen':
            union=torch.sum(y_pred_f)+torch.sum(y_true_f)
        else:
            raise ValueError("Unknown `loss_type`: %s" % self.loss_type)

        dice_loss=1-(2.*intersection+self.smooth)/(union+self.smooth)
        ssim_loss=1-torch.mean(self.ssim(y_pred,y_true))
        square_loss=torch.mean(torch.square(y_pred-y_true))

        return dice_loss + ssim_loss + square_loss+totalloss

