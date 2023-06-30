from model import*
from hyperparameters import*
from loss_function import*
import torch 
import pytorch_lightning as pl 
from pytorch_lightning.loggers import TensorBoardLogger

###LIGHTNING MODEL
class LoadModel(pl.LightningModule):

    def __init__(self):
        super(LoadModel, self).__init__()
        self.save_hyperparameters()
        self.encoder=Encoder(batch_size)
        self.decoder1=Decoder()
        self.decoder2=Decoder()
        self.decoder3=Decoder()
        self.decoder4=Decoder()
        self.loss_fn=CombinedLoss(loss_type='jaccard', smooth=1.)
        # self.loss_fn=SliceLoss()
  
    def forward(self,batch):
       
        outputs=[]
        targets=[]
        for i, (inputs,inp_variables, target) in enumerate(batch):
           
            latent,trans_shape,latent_shape=self.encoder(inputs,inp_variables)
            
            velocity1=self.decoder1
            velocity2=self.decoder2
            velocity3=self.decoder3
            velocity4=self.decoder4

            velx=velocity1(latent,trans_shape)
            vely=velocity2(latent,trans_shape)
            velz=velocity3(latent,trans_shape)
            press=velocity4(latent,trans_shape)

            velx=torch.squeeze(velx,1)
            vely=torch.squeeze(vely,1)
            velz=torch.squeeze(velz,1)
            press=torch.squeeze(press,1)
           
            output=[velx, vely,velz, press]

            outputs.append(output)
            targets.append(target)

        return outputs,targets
    
    # training_step defines the train loop.
    def training_step(self,batch, batch_idx):
      
        inputs,inp_ops,targets=batch
       
        targets=targets.view(-1, 4, 384, 128, 64)
        latent,trans_shape,latent_shape=self.encoder(inputs,inp_ops)

        velocity1=self.decoder1
        velocity2=self.decoder2
        velocity3=self.decoder3
        velocity4=self.decoder4

        velx=velocity1(latent,trans_shape)
        vely=velocity2(latent,trans_shape)
        velz=velocity3(latent,trans_shape)
        press=velocity4(latent,trans_shape)
        
        output=torch.stack([velx, vely,velz, press])
        output= torch.squeeze(output,1)
        output=output.view(-1, 4, 384, 128, 64)
       
        # output=[velx, vely,velz, press]

        loss=self.loss_fn(targets,output)
        self.log("train_loss",loss)

        return {'loss':loss}

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5,weight_decay=1e-5)
        return optimizer
 
    def validation_step(self,batch, batch_idx):
    
        inputs,inp_ops,targets=batch

        targets=targets.view(-1, 4, 384, 128, 64)
        latent,trans_shape,latent_shape=self.encoder(inputs,inp_ops)

        velocity1=self.decoder1
        velocity2=self.decoder2
        velocity3=self.decoder3
        velocity4=self.decoder4

        velx=velocity1(latent,trans_shape)
        vely=velocity2(latent,trans_shape)
        velz=velocity3(latent,trans_shape)
        press=velocity4(latent,trans_shape)
        
        output=torch.stack([velx, vely,velz, press])
        output= torch.squeeze(output,1)
        output=output.view(-1, 4, 384, 128, 64)
      
        # output=[velx, vely,velz, press]

        loss=self.loss_fn(targets,output)
        self.log("val_loss",loss)

        return {'val_loss':loss}
    
    def validation_epoch_end(self,outputs):
    
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)

        return {'val_loss': avg_loss}

    def test_step(self,batch, batch_idx):
    
        inputs,inp_ops,targets=batch
        targets=targets.view(-1, 4, 384, 128, 64)
        latent,trans_shape,latent_shape=self.encoder(inputs,inp_ops)

        velocity1=self.decoder1
        velocity2=self.decoder2
        velocity3=self.decoder3
        velocity4=self.decoder4

        velx=velocity1(latent,trans_shape)
        vely=velocity2(latent,trans_shape)
        velz=velocity3(latent,trans_shape)
        press=velocity4(latent,trans_shape)
        
        output=torch.stack([velx, vely,velz, press])
        output= torch.squeeze(output,1)
        output=output.view(-1, 4, 384, 128, 64)
    
        # output=[velx, vely,velz, press]

        loss=self.loss_fn(targets,output)
        self.log("test_loss",loss)

        return {'test_loss':loss}