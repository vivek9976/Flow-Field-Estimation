# Flow Field Estimation 

Code for the PyTorch implementation of "Flow Field Estimation" for vehical data which mainly includes the flow field like velocity and pressure.

# Abstract

Flow fields, including velocity and pressure fields, are typically used as references for
vehicle shape design in the automotive industry to ensure steering stability and energy
conservation. Generally, flow fields are calculated using computational fluid dynamics
(CFD) simulations which is time-consuming and expensive. Therefore, a more efficient
and interactive method is desired by designers for advanced shape discussion and design.
To this end, we propose a fast estimation model using 3D convolutional neural networks.
We employ a style extractor to obtain sufficient deep features of each vehicle shape and
apply them using adaptive instance normalisation to improve the estimation performance.
In addition, a proposed loss function which mainly includes a slice-weighted square and combined loss function is used to train the estimation model. Our proposed method outperforms especially on flow field estimation in wake regions and regions near the vehicle surface. Therefore, the proposed method allows designing vehicle shapes while ensuring desirable aerodynamic performance within a much shorter period than extended CFD simulations.

# Training 

To train the model first install all the packages required and then in the /hyparameter.py/ you can change your parameters according your system requirements and also change the path of training data and in the dataloder.py
# Output on Test Data for the best weights of Style GAN model.

![ffe1](https://github.com/vivek9976/Flow-Field-Estimation/assets/79739934/b78b8f79-819f-45c6-aadc-6a601f9334e6)

# Output on Test Data for the best weights of Encoder-Decoder model.

![ffe2](https://github.com/vivek9976/Flow-Field-Estimation/assets/79739934/ce96b119-9d1c-4ef2-b703-fb287d0f7dbf)
