import torch.nn as nn
import torch

class MinibatchDiscrimination2d(nn.Module):
    """ This module performs Mini-BatchDiscrimination for (batch_size x M x N x N) tensors.
        For more information on Mini-BatchDiscrimination check : https://arxiv.org/abs/1606.03498v1
    
            in_size : N size for (batch_size x M x N x N) tensor (int)
            in_flt : M size for (batch_size x M x N x N) tensor (int)
            out_flt : number of filter to add, the more filter you add the more your final results will not be similar in theory
            device : device on which you train the model (GPU/CPU) (torch device)
            reduction_kernel_size : to improve similarity detection of the image a convolution and deconvolution is perfomed, 
                                    the kernel-size refers to the kernel size used in the convolution (int)
            intermediate_features : the more you have the more your Mini-BatchDiscrimination will be precise (int) - default value 16
    
        This module returns a tensor : original input concatenate with Mini-BatchDiscrimination output : a (batch_size x (M+out_flt) x N x N)
    """
    def __init__(self,in_size, in_flt,out_flt,device,reduction_kernel_size,intermediate_features=16):
      super(MinibatchDiscrimination2d, self).__init__()
      self.in_flt = in_flt
      self.out_flt = out_flt
      self.intermediate_features = intermediate_features
      self.kernel_size= reduction_kernel_size
      torch.use_deterministic_algorithms(True)
      self.conv2d= nn.Conv2d(in_flt, 3, self.kernel_size, self.kernel_size, 0, bias=False).to(device) #out of conv -> 3 channels, 4x4 kernel, no pad, stride 4
      self.conv2dt= nn.ConvTranspose2d(out_flt, out_flt, self.kernel_size, self.kernel_size, 0, bias=False).to(device)

      self.t=int((in_size-self.kernel_size)/self.kernel_size)+1
      self.T = nn.Parameter( 
          torch.Tensor(3*self.t*self.t, out_flt*self.t*self.t, self.intermediate_features), requires_grad=True
        )
      torch.nn.init.xavier_uniform_(self.T.data)
        
    def forward(self, x):
        x_r=self.conv2d(x) #in 32x3x32x32 -> 32x3x8x8
        M = torch.mm(x_r.view(-1,3*self.t*self.t), self.T.view(3*self.t*self.t, -1))#32x2048 =32 x 2*8*8*16 
        M = M.view(-1, self.out_flt*self.t*self.t, self.intermediate_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
        out = out.view(-1,self.out_flt,self.t,self.t)
        out_a = self.conv2dt(out)
        return torch.cat([x, out_a], 1)

# custom weights initialization called on netG and netD
def weights_init(m):
    """ Weights initialization function"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
    """ Generator net 
        This is a classic DC-GAN architecture


            nz: size of input of the generator or latent space dimension (int)
            ngf : number of filter used to build the generator (from ngfx8 to nc) (int)
            nc : number of channel of an image, classic image has 3 channel (int)
    """
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.nz=nz
        self.ngf=ngf
        self.nc=nc
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, int(ngf/2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(ngf/2)),
            nn.ReLU(True),
            # state size. (ngf/2) x 64 x 64
            nn.ConvTranspose2d( int(ngf/2), int(ngf/4), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(ngf/4)),
            nn.ReLU(True),
            # state size. (ngf/4) x 128 x 128
            nn.ConvTranspose2d( int(ngf/4), nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)
        
class Discriminator(nn.Module):
    """ Discriminant net 
        This is a classic DC-GAN architecture with a Mini BatchDiscrimination layer added to avoid mode collapse (always generating same image)
        Output layer is depending on mode because wgan does not have the same discriminant's output as dcgan.

            nc : number of channel of an image, classic image has 3 channel (int)
            ndf : number of filter used to build the generator (from nc to ndfx8) (int)
            device : device on which you train the model (GPU/CPU) (torch device)
            mode : choose between "dcgan" and "wgan", which will change the output layer, thus output value (str) - default value "dcgan"
            
    """
    def __init__(self, nc, ndf, device,mode="dcgan"):
        super(Discriminator, self).__init__()
        self.nc=nc
        self.ndf=ndf
        self.device=device
        self.mode=mode
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, int(ndf/4), 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf/4) x 128 x 128
            MinibatchDiscrimination2d(128,int(ndf/4),2,device,4),
            nn.Conv2d(int(ndf/4)+2, int(ndf/2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(ndf/2)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf/2) x 64 x 64
            nn.Conv2d(int(ndf/2), ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        if self.mode=="wgan":
            self.final_layer=nn.Sequential(
                nn.Flatten(),
                nn.Linear(ndf*8*4*4,1)
            )
        else:
            self.final_layer=nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )



    def forward(self, input):
        x=self.main(input)
        y=self.final_layer(x)
        return y