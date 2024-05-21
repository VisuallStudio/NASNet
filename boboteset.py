import torch
import torch.nn.functional as F
# shape=32
# input = torch.randn(1, 1, shape, shape)
# print("{}  {}".format(input,input.shape))
# kernel=6
# padding=(kernel-1)//2
# deconv = torch.nn.ConvTranspose2d(1, 1, kernel, 2, padding, output_padding=0,bias=False)
# deconv.weight.data = torch.ones(1,1,kernel,kernel)
# dout = deconv(input)
# print("{}  {}".format(dout,dout.shape))

shape=32
input = torch.randn(1, 1, shape, shape)
print("{}  {}".format(input,input.shape))
kernel=4
padding=(kernel-1)//2
deconv = torch.nn.ConvTranspose2d(1, 1, kernel, 1, padding, output_padding=0,bias=False)
deconv.weight.data = torch.ones(1,1,kernel,kernel)
dout = deconv(input)
print("{}  {}".format(dout,dout.shape))
