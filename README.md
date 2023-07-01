# Flow Field Estimation 

Code for the PyTorch implementation of "Flow Field Estimation" for vehical data which includes the flow fields like velocity and pressure. I have implemented 2 model Encoder-Decoder model and Style GAN model. So, both are giving the output pretty good so its hard to mention which is performing more better because each model have some pros and cons.
I have also provided pretrained weights for each model you can use them to finetune the model incase you requirement to run the model is not satisfied.

# Abstract

Flow fields, including velocity and pressure fields, are typically used as references for vehicle shape design in the automotive industry to ensure steering stability and energy conservation. Generally, flow fields are calculated using computational fluid dynamics (CFD) simulations which is time-consuming and expensive. Therefore, a more efficient and interactive method is desired by designers for advanced shape discussion and design.
To this end, we propose a fast estimation model using 3D convolutional neural networks. We employ a style extractor to obtain sufficient deep features of each vehicle shape and apply them using adaptive instance normalisation to improve the estimation performance.
In addition, a proposed loss function which mainly includes a slice-weighted square and combined loss function is used to train the estimation model. Our proposed method outperforms especially on flow field estimation in wake regions and regions near the vehicle surface. Therefore, the proposed method allows designing vehicle shapes while ensuring desirable aerodynamic performance within a much shorter period than extended CFD simulations.


# Dataset

For the training estimation model, a dataset that includes the 3D shape information of vehicles and corresponding flow fields. In the dataset, inputs are the unsigned distance functions(uSDFs) and outputs are the 3D flow fields around the vehicle.

# Requirement

```bash
  https://pytorch.org/get-started/locally/
  Download pytroch using the link above according 
  to your requirement.
  pip install numpy
  pip install torchmetrics 
  pip install pytorchlightning
  pip install matplotlib
  pip install tensorboard
  Use python version>=3.8
  and other you can install if I forgot to mention.
```
# Training 

To train the particualar model first install all the packages required and then in the ```hyparameter.py``` you can change your parameters according your system requirements and also change the path of training data in the ```training.py``` itself and in the ```dataloder.py``` match the input format of data with your dataset and in ```dataloader.py``` I have used ```x=x[128:512,64:192,0:64]/255``` to extract the important features from the data you can change this according to your need by visualising your own data. You do not need to change anything in the ```model.py``` and ```loss_function.py``` but you have to make some changes in the ```Lightning_model.py training.py dataloader.py``` according to your requirements and also I have implemented 3 loss functions, so you can use any of them and can visualize your ouput on all of them.

# Testing 

To test the model you have to make some changes in the ```dataloader.py``` take input velocity and pressure vector as a array of list and run the ```test.py``` and binary mask of each case is present , where one means the region that allows air to pass through, was added into the dataset. so you can also use that in one of the plots to gain more insights.

# Output on Test Data for the best weights of Style GAN model.

![ffe1](https://github.com/vivek9976/Flow-Field-Estimation/assets/79739934/b78b8f79-819f-45c6-aadc-6a601f9334e6)

# Output on Test Data for the best weights of Encoder-Decoder model.

![ffe2](https://github.com/vivek9976/Flow-Field-Estimation/assets/79739934/ce96b119-9d1c-4ef2-b703-fb287d0f7dbf)
