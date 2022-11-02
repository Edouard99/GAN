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
    """ This function trains the network with a WGAN training:
        Discriminant is fed with real and generated images. The goal of the discriminant is to give a score assessing the realness of fakeness of an iamges.
        Discriminant tries to maximize the distance between the score given to a real image and a fake image in order to separate the image distribution.
        Discriminant's loss is calculated with Wasserstein Loss.
        Discriminant tries to maximize the distance, in the code then we minimize -distance (-loss).
        Generator tries to reduce the distance between the two distribution by tring to obtain the score of a real image with a fake image from the discriminant.
        Generator's loss is calculated with Wasserstein Loss.
        Generator tries to minimize the distance
        Discriminant's and Generator's weights are optimized in order to minimize each loss.
        Discriminant is trained 3 times more than the generator and a weight clipping is applied to the discriminator's weights.
        For more information on WGAN training check : https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/


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
    """ 

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    nz=netG.nz
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: minimize : -( D(x) - D(G(z)) )
            ############################
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            D_x = torch.mean(netD(real_cpu)) # Prediction on real images
            D_G_z1 = torch.mean(netD(fake)) # Prediction on fake images
            errD = -(D_x - D_G_z1) # -Distance between the distribution to be minimize
            errD.backward()
            optimizerD.step() # Update D
            D_losses.append(abs(errD.item()))
            for p in netD.parameters():
                p.data.clamp_(-0.01, 0.01) # Weight Clipping
            if iters%3==0 : # Discriminant is trained 3 times more than the generator
                netG.zero_grad()
                fake = netG(noise)
                D_G_z2 = torch.mean(netD(fake))
                errG = -D_G_z2 # Distance to be minimized
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