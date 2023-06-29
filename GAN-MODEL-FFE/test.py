from training import*
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt


path="/mnt/disk2/jaiswal/flowfieldestimationstylegan/flowfieldestimationstylegan/pytorch_lightning_model/newmodeladampreloss200/my_model/version_0/checkpoints/epoch=199-step=20600.ckpt"
model = LoadModel.load_from_checkpoint(path)
model.eval()
outputs,targets,sdfs=model(val_dataloader)
# for i in range(len(sdfs)):
#     fig,ax=plt.subplots(1,1,figsize=(20,20))
#     arrsdf=targets[i][4]
#     arrsdf=arrsdf[0]*255
#     print(arrsdf)

#Code to plot the image for diiferent velocity vector and pressure 

for i in range(len(outputs)):

    fig, ax = plt.subplots(4, 2, figsize=(40,40))

    inputvalx=targets[i][0]
    predvalx=outputs[i][0].detach().numpy()
    inputvaly=targets[i][1]
    predvaly=outputs[i][1].detach().numpy()
    inputvalz=targets[i][2]
    predvalz=outputs[i][2].detach().numpy()
    inputpress=targets[i][3]
    predpress=outputs[i][3].detach().numpy()

    velx_input= np.array(inputvalx[0]*255.)
    velx_outputs=np.array(predvalx[0]*255.)
    vely_input= np.array(inputvaly[0]*255.)
    vely_outputs=np.array(predvaly[0]*255.)
    velz_input= np.array(inputvalz[0]*255.)
    velz_outputs=np.array(predvalz[0]*255.)
    press_input= np.array(inputpress[0]*255.)
    press_outputs=np.array(predpress[0]*255.)

    ax[0,0].imshow(velx_input[:,:,20], cmap='seismic')
    ax[0,0].set_title("Ground Truth velocity x")
    ax[0,1].imshow(velx_outputs[:,:,20], cmap='seismic')
    ax[0,1].set_title("Predicted velocity x")

    ax[1,0].imshow(vely_input[:,:,20], cmap='seismic')
    ax[1,0].set_title("Ground Truth velocity y")
    ax[1,1].imshow(vely_outputs[:,:,20], cmap='seismic')
    ax[1,1].set_title("Predicted velocity y")

    ax[2,0].imshow(velz_input[:,:,20], cmap='seismic')
    ax[2,0].set_title("Ground Truth velocity  z")
    ax[2,1].imshow(velz_outputs[:,:,20], cmap='seismic')
    ax[2,1].set_title("Predicted velocity z")

    ax[3,0].imshow(press_input[:,:,20], cmap='seismic')
    ax[3,0].set_title("Ground Truth pressure")
    ax[3,1].imshow(press_outputs[:,:,20], cmap='seismic')
    ax[3,1].set_title("Predicted pressure")

    pathdir="plots/style200/"
    leftpath=".png"
    totalpath=str(pathdir+str(i+1)+leftpath)

    fig.savefig(totalpath)
    plt.show()

# #Code to plot the net velocity and pressure image 
# for i in range(len(outputs)):

#     fig, ax = plt.subplots(2, 2, figsize=(40,40))

#     inputvalx=targets[i][0]
#     predvalx=outputs[i][0].detach().numpy()
#     inputvaly=targets[i][1]
#     predvaly=outputs[i][1].detach().numpy()
#     inputvalz=targets[i][2]
#     predvalz=outputs[i][2].detach().numpy()
#     inputpress=targets[i][3]
#     predpress=outputs[i][3].detach().numpy()

#     inputvel=np.sqrt(inputvalx**2+inputvaly**2+inputvalz**2)
#     inputvel= np.array(inputvel[0]*255.)
#     predvel=np.sqrt(predvalx**2+predvaly**2+predvalz**2)
#     predvel= np.array(predvel[0]*255.)
#     press_input= np.array(inputpress[0]*255.)
#     press_outputs=np.array(predpress[0]*255.)

#     ax[0,0].imshow(inputvel[:,:,20], cmap='seismic')
#     ax[0,0].set_title("Ground Truth net velocity ")
#     ax[0,1].imshow(predvel[:,:,20], cmap='seismic')
#     ax[0,1].set_title("Predicted net velocity ")

#     ax[1,0].imshow(press_input[:,:,20], cmap='seismic')
#     ax[1,0].set_title("Ground Truth pressure ")
#     ax[1,1].imshow(press_outputs[:,:,20], cmap='seismic')
#     ax[1,1].set_title("Predicted pressure ")

#     pathdir="plots/ADAMSLICESQUARELOSS/TESTDATA/NETVEL/CMAP=SEISMIC-20/"
#     leftpath=".png"
#     totalpath=str(pathdir+str(i+1)+leftpath)

#     fig.savefig(totalpath)
#     plt.show()

#Code to plot the image of difference between pred velocity and truth velocity for all component  and same for pressure  
# for i in range(len(outputs)):
#     fig, ax = plt.subplots(4, 1, figsize=(40,40))

#     inputvalx=targets[i][0]
#     predvalx=outputs[i][0].detach().numpy()
#     inputvaly=targets[i][1]
#     predvaly=outputs[i][1].detach().numpy()
#     inputvalz=targets[i][2]
#     predvalz=outputs[i][2].detach().numpy()
#     inputpress=targets[i][3]
#     predpress=outputs[i][3].detach().numpy()

#     velx_input= np.array(inputvalx[0])
#     velx_outputs=np.array(predvalx[0])
#     vely_input= np.array(inputvaly[0])
#     vely_outputs=np.array(predvaly[0])
#     velz_input= np.array(inputvalz[0])
#     velz_outputs=np.array(predvalz[0])
#     press_input= np.array(inputpress[0])
#     press_outputs=np.array(predpress[0])
    
    
#     diffx=np.absolute(np.subtract(velx_input,velx_outputs))*255
#     diffy=np.absolute(np.subtract(vely_input,vely_outputs))*255
#     diffz=np.absolute(np.subtract(velz_input,velz_outputs))*255
#     diffpress=np.absolute(np.subtract(press_input,press_outputs))*255
    
    
#     ax[0].imshow(diffx[:,:,20], cmap='seismic')
#     ax[0].set_title("Difference velocity x")

#     ax[1].imshow(diffy[:,:,20], cmap='seismic')
#     ax[1].set_title("Difference Truth velocity y")

#     ax[2].imshow(diffz[:,:,20], cmap='seismic')
#     ax[2].set_title("Difference Truth velocity  z")

#     ax[3].imshow(diffpress[:,:,20], cmap='seismic')
#     ax[3].set_title("Difference Truth pressure")
    

    
#     pathdir="plots/DIFF/ADAMPRELOSS/TESTDATA/CMAP=SEISMIC-20/"
#     leftpath=".png"
#     totalpath=str(pathdir+str(i+1)+leftpath)

#     fig.savefig(totalpath)
#     plt.show()

#Code to plot the difference between velocity truth and pred for all components and same for pressure


# for i in range(len(outputs)):
#     fig, ax = plt.subplots(4, 1, figsize=(40,40))

#     inputvalx=targets[i][0]
#     predvalx=outputs[i][0].detach().numpy()
#     inputvaly=targets[i][1]
#     predvaly=outputs[i][1].detach().numpy()
#     inputvalz=targets[i][2]
#     predvalz=outputs[i][2].detach().numpy()
#     inputpress=targets[i][3]
#     predpress=outputs[i][3].detach().numpy()

#     velx_input= np.array(inputvalx[0])
#     velx_outputs=np.array(predvalx[0])
#     vely_input= np.array(inputvaly[0])
#     vely_outputs=np.array(predvaly[0])
#     velz_input= np.array(inputvalz[0])
#     velz_outputs=np.array(predvalz[0])
#     press_input= np.array(inputpress[0])
#     press_outputs=np.array(predpress[0])
    
    
#     diffx=np.absolute(np.subtract(velx_input,velx_outputs))*255
#     diffy=np.absolute(np.subtract(vely_input,vely_outputs))*255
#     diffz=np.absolute(np.subtract(velz_input,velz_outputs))*255
#     diffpress=np.absolute(np.subtract(press_input,press_outputs))*255
    
#     diffx=np.reshape(diffx,(-1,1))
#     diffy=np.reshape(diffy,(-1,1))
#     diffz=np.reshape(diffz,(-1,1))
#     diffpress=np.reshape(diffpress,(-1,1))
#     iters=[j for j in range(len(diffx))]
#     ax[0].plot(iters,diffx)
#     ax[0].set_title("Difference velocity x")

#     ax[1].plot(iters,diffy)
#     ax[1].set_title("Difference Truth velocity y")

#     ax[2].plot(iters,diffz)
#     ax[2].set_title("Difference Truth velocity  z")

#     ax[3].plot(iters,diffpress)
#     ax[3].set_title("Difference Truth pressure")
    
#     pathdir="plots/DIFF/ADAMMSE/DIFFPLOT/"
#     leftpath=".png"
#     totalpath=str(pathdir+str(i+1)+leftpath)

#     fig.savefig(totalpath)
#     plt.show()



#Code to plot the difference between net velocity truth and pred and same for pressure


# for i in range(len(outputs)):
#     fig, ax = plt.subplots(2, 1, figsize=(40,40))

#     inputvalx=targets[i][0]
#     predvalx=outputs[i][0].detach().numpy()
#     inputvaly=targets[i][1]
#     predvaly=outputs[i][1].detach().numpy()
#     inputvalz=targets[i][2]
#     predvalz=outputs[i][2].detach().numpy()
#     inputpress=targets[i][3]
#     predpress=outputs[i][3].detach().numpy()

#     inputvel=np.sqrt(inputvalx**2+inputvaly**2+inputvalz**2)
#     inputvel= np.array(inputvel[0]*255.)
#     predvel=np.sqrt(predvalx**2+predvaly**2+predvalz**2)
#     predvel= np.array(predvel[0]*255.)
#     press_input= np.array(inputpress[0]*255.)
#     press_outputs=np.array(predpress[0]*255.)
    
#     diff=np.absolute(np.subtract(inputvel,predvel))
#     diffpress=np.absolute(np.subtract(press_input,press_outputs))
    
#     diff=np.reshape(diff,(-1,1))
#     diffpress=np.reshape(diffpress,(-1,1))
#     iters=[j for j in range(len(diff))]
#     ax[0].plot(iters,diff)
#     ax[0].set_title("Difference velocity")

#     ax[1].plot(iters,diffpress)
#     ax[1].set_title("Difference Truth pressure")
    
#     pathdir="plots/DIFF/ADAMPRELOSS/NETDIFFPLOT/"
#     leftpath=".png"
#     totalpath=str(pathdir+str(i+1)+leftpath)

#     fig.savefig(totalpath)
#     plt.show()