import torch



def Normalization(fake):
  for i in range(0,3):
      fake[i]=(fake[i]-fake[i].min())/(fake[i].max()-fake[i].min())
  return fake