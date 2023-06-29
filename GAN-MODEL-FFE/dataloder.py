import torch
import numpy as np 
import os
import h5py
import glob 
import yaml
import math
from torch.utils.data import Dataset, DataLoader

###DATA LOADER 
class FluidSimDataset(Dataset):
    def __init__(self, data_path:str, transforms = None):
        """data_path folder contains all the HDF5 files"""
        super().__init__()
        self.data_path = data_path
        self.hdf5files = os.listdir(data_path) #list of names of hdf5 files
        self.transforms = transforms
        
    def __len__(self):
        return len(self.hdf5files)
        
    def __getitem__(self, idx):
        filename = self.hdf5files[idx]
        file = h5py.File(f"{self.data_path}/{filename}", "r")

        velocity_x = np.asarray(file['velocity_x'], dtype=np.float32)
        velocity_y = np.asarray(file['velocity_y'], dtype=np.float32)
        velocity_z = np.asarray(file['velocity_z'], dtype=np.float32)
        pressure = np.asarray(file['Pressure'], dtype=np.float32)
        mask=np.asarray(file['binary_mask'],dtype=np.float32)
        
        car_speed = (int(str(filename).split('kmph')[0].split('_')[-1]))
        angle_of_attack = str(filename).split('deg')[0].split('_')[-1]
        if angle_of_attack == 'zero':
            angle_of_attack = int('0')
        angle_of_attack = int(angle_of_attack)

        inp_ops=[car_speed, angle_of_attack]
        inp_ops = torch.tensor(inp_ops, dtype = torch.float32) 
        inp_ops=inp_ops.view(1,2)
        
        x = np.asarray(file['sdf'], dtype=np.float32)
        #extract important dimensions only
        x = x[128:512, 64:192, 0:64]/255
        x = torch.tensor(np.array(x), dtype = torch.float32)
        x=x.view(1, 384,128, 64)
        
        # velocity_x= velocity_x[128:512, 64:192, 0:64]/255
        # velocity_y= velocity_y[128:512, 64:192, 0:64]/255
        # velocity_z= velocity_z[128:512, 64:192, 0:64]/255
        # pressure= pressure[128:512, 64:192, 0:64]/255
        # mask=mask[128:512, 64:192, 0:64]/255

        # velocity_x =torch.tensor(velocity_x,dtype=torch.float32) 
        # velocity_y = torch.tensor(velocity_y,dtype=torch.float32)
        # velocity_z = torch.tensor(velocity_z,dtype=torch.float32)
        # pressure=torch.tensor(pressure,dtype=torch.float32)
        # mask=torch.tensor(mask,dtype=torch.float32)
        y = [velocity_x, velocity_y, velocity_z, pressure]
        y = list(map(lambda arr:arr[128:512, 64:192, 0:64]/255, y))
        y = torch.tensor(np.array(y), dtype = torch.float32)
      
        
        file.close()
     
        return x,inp_ops,y
