import torch.nn as nn
import torch

class MinibatchDiscrimination2d(nn.Module):
    def __init__(self,in_size, in_flt,out_flt,device,intermediate_features=16):
      super(MinibatchDiscrimination2d, self).__init__()
      self.in_flt = in_flt
      self.out_flt = out_flt
      self.intermediate_features = intermediate_features

      self.conv2d= nn.Conv2d(in_flt, 3, 4, 4, 0, bias=False).to(device) #out of conv -> 3 channels, 4x4 kernel, no pad, stride 4
      self.conv2dt= nn.ConvTranspose2d(out_flt, out_flt, 4, 4, 0, bias=False).to(device)

      self.t=int((in_size-4)/4)+1
      self.T = nn.Parameter( 
          torch.Tensor(3*self.t*self.t, out_flt*self.t*self.t, self.intermediate_features)
        )
    def forward(self, x):
        x_r=self.conv2d(x) #in 32x3x32x32 -> 32x3x8x8
        M = torch.mm(x_r.view(-1,3*self.t*self.t), self.T.view(3*self.t*self.t, -1))#32x2048 =32 x 2*8*8*16 
        M = M.view(-1, self.out_flt*self.t*self.t, self.intermediate_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
        out = out.view(-1,self.out_flt,self.t,self.t)
        out_a = self.conv2dt(out)
        return torch.cat([x, out_a], 1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Code

class Generator(nn.Module):
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
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
        
class Discriminator(nn.Module):
    def __init__(self, nc, ndf, device):
        super(Discriminator, self).__init__()
        self.nc=nc
        self.ndf=ndf
        self.device=device
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            MinibatchDiscrimination2d(32,ndf,2,device),
            nn.Conv2d(ndf+2, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)