import torch
from .basic_functions import *
from tensorflow.keras.utils import array_to_img
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils
import numpy as np
from IPython.display import display

def train(dataloader,netD,netG,optimizerD,optimizerG,num_epochs,device,savenet,pathsavenet,pathsaveimg,fixed_noise):
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.
    nz=netG.nz
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            D_x = torch.mean(netD(real_cpu))
            D_G_z1 = torch.mean(netD(fake))
            errD = -(D_x - D_G_z1)
            errD.backward()
            optimizerD.step()
            D_losses.append(abs(errD.item()))
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01)
            if iters%3==0 :
                netG.zero_grad()
                fake = netG(noise)
                D_G_z2 = torch.mean(netD(fake))
                errG = -D_G_z2
                errG.backward()
                optimizerG.step()
                G_losses.append(abs(errG.item()))



            # Output training stats
            if iters % 30 == 0:
              print('[{}/{}][{}/{}]\tLoss_D: {}\tLoss_G: {}\tD(x): {}\tD(G(z)): {} / {}'.format(
                      epoch, 
                      num_epochs, 
                      i, 
                      len(dataloader),
                      errD.item(), 
                      errG.item(), 
                      D_x, 
                      D_G_z1, 
                      D_G_z2))

            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                      clear_output(wait=True)
                      f_noise = torch.randn(1, nz, 1, 1, device=device)
                      fake = netG(f_noise).detach().cpu()
                    faken=Normalization(fake[0])
                    display(array_to_img(np.transpose(faken,(1,2,0))))

            iters += 1    
        
        if epoch%4==0 :
            with torch.no_grad():
              fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            if (savenet):
                torch.save(netD.state_dict(), os.path.join(pathsavenet,"netD"+str(epoch)+".pth"))
                torch.save(netG.state_dict(), os.path.join(pathsavenet,"netG"+str(epoch)+".pth"))
            img_to_save=np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu().numpy(),(1,2,0))
            plt.imsave(os.path.join(pathsaveimg,"grid_"+str(epoch)+".png"),img_to_save)
            plt.imshow(img_to_save)
            #plt.savefig("grid_"+str(epoch)+".png")

    return img_list,G_losses,D_losses