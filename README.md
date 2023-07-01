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

# Output on Test Data Different for the best weight of Style GAN model.

