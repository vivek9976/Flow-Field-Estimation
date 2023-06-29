from newmodel import*
from hyperparameters import*
from lightning_model import*
from loss_function import*
from dataloder import*
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger
from sklearn.model_selection import train_test_split


data_path = "/mnt/disk2/jaiswal/flowfieldestimationstylegan/Flow/train_files"
dataset = FluidSimDataset(data_path = data_path)
train_set,val_set= train_test_split(dataset, train_size=0.9,shuffle=False,random_state=42)

train_dataloader=DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
val_dataloader=DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

tb_logger= TensorBoardLogger("newmodeladampreloss200", name="my_model")
mlflow_logger = MLFlowLogger(experiment_name="Baseline4-100e")
logger = [tb_logger, mlflow_logger]

model=LoadModel()
trainer =Trainer(detect_anomaly=False,logger=logger,max_epochs=epochs,accelerator="gpu", devices=[2],log_every_n_steps=30)
trainer.fit(model,train_dataloader,val_dataloader)


