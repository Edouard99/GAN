import torch
from .basic_functions import *
from tensorflow.keras.utils import array_to_img
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils
import numpy as np
from IPython.display import display

def train(dataloader,netD,netG,optimizerD,optimizerG,num_epochs,device,savenet,pathsavenet,pathsaveimg,fixed_noise,monitor=True,trsh=0.8):
    """ This function trains the network with a classic GAN training + Monitoring of training:
        Discriminant is fed with real and generated images. The goal of the discriminant is to assess the probability of the image being real or not.
        Discriminant's loss is calculated with Binary Cross Entropy Loss with label 0 = fake and label 1 = real
        Generator's Loss is based on Discriminant results on generated images
        Discriminant's and Generator's weights are optimized in order to minimize each loss.

        Monitoring : In order to prevent the discriminant to become too strong, the discriminant is updated only if Discriminant_Loss > treshold*Generator_Loss

        Each epoch this function saves the network weights (optionnal: savenet=True/False) and saves a grid of images generated from fixed noise
        This function returns a list of the images grids generated during training, the Generator Loss and the Discriminant Loss evolution

            dataloader : dataloader object that will load your images and feed it to the network (torch dataloader)
            netD : discriminant neural network (nn Module)
            netG : generator neural network (nn Module)
            optimizerD : discriminant's optimizer (torch Optimizer)
            optimizerG : generator's optimizer (torch Optimizer)
            num_epochs : number of epochs for training (int)
            device : device on which training is done (CPU/GPU) (torch device)
            savenet : True = save the network weights each 4 epochs in pathsavenet location, False = do not save the network weights (boolean)
            pathsavenet : path to the directory where you want to save network weights, "" if savenet=False (str)
            pathsaveimg : path to the directory where you want to save the grid of images generated from fixed noise each 4 epochs (str)
            fixed_noise : noise that will be used to generate the grid of N images for a generator with nz size of input(tensor shape: N, nz, 1, 1)
            monitor : activating or deactivating monitoring, monitor=False is equivalent to training_classic.py train function (boolean) - default value = true
            trsh : treshold used for monitoring (float) - default value = 0.8
    """


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
            ############################
            # (1) Update D network: minimize BCELOSS : -( log(D(x)) + log(1 - D(G(z))) )
            ############################
            ## Train with all-real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = BCEsmooth(output, label,device)
            errD_real.backward()
            D_x = output.mean().item() # Mean prediction on the batch


            ## Train with all-fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = BCEsmooth(output, label,device)
            errD_fake.backward()
            D_G_z1 = output.mean().item() # Mean prediction on the batch

            ## Compute Error of the Discriminant on both batchs
            errD = errD_real + errD_fake
            ## Update D
            ## If monitoring is ON, the discriminant is updated only if Discriminant_Loss > treshold*Generator_Loss
            if monitor==True:
                if iters !=0 :
                    if errD.item() > trsh*errG.item():
                        optimizerD.step()
                else :
                    optimizerD.step()
            else :
                optimizerD.step()


            ############################
            # (2) Update G network: minimize - log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = BCEsmooth(output, label,device)
            errG.backward()
            D_G_z2 = output.mean().item()

            ## Update G
            optimizerG.step()


            # Output training stats
            if i % 25 == 0:
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
        # Saves the network and images
        if epoch%1==0 :
            with torch.no_grad():
              fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            if (savenet):
                torch.save(netD.state_dict(), os.path.join(pathsavenet,"netD"+str(epoch)+".pth"))
                torch.save(netG.state_dict(), os.path.join(pathsavenet,"netG"+str(epoch)+".pth"))
                torch.save(optimizerD.state_dict(), os.path.join(pathsavenet,"optimD"+str(epoch)+".pth"))
                torch.save(optimizerG.state_dict(), os.path.join(pathsavenet,"optimG"+str(epoch)+".pth"))
            img_to_save=np.transpose(vutils.make_grid(fake, padding=2, normalize=True).cpu().numpy(),(1,2,0))
            plt.imsave(os.path.join(pathsaveimg,"grid_"+str(epoch)+".png"),img_to_save)
            plt.imshow(img_to_save)

    return img_list,G_losses,D_losses