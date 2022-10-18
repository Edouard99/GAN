import torch

def BCEsmooth(input,target):
  target_ls= target*(1-0.1)+0.1/2
  criterion = torch.nn.BCELoss().cuda()
  return criterion(input,target_ls)

def Normalization(fake):
  for i in range(0,3):
      fake[i]=(fake[i]-fake[i].min())/(fake[i].max()-fake[i].min())
  return fake