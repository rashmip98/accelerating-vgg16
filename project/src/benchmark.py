import torch
from torch.utils.cpp_extension import load

# Imports
import time
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from pathlib import Path

# Initialization
# Perform any initialization procedures such as 
#   Binding
#   Flashing FPGA
#   Instantiating your FPGA accelerated VGG16 net
#   Any other initialization that shouldn't be timed
host = load(name="host", sources=["q4/src/host.cpp"], extra_cflags=['-Wall', '-O3', '-g', '-std=c++14'],
            extra_ldflags=['-L$XILINX_XRT/lib/', '-lOpenCL', '-lpthread', '-lrt', '-lstdc++'],
            extra_include_paths=['I$XILINX_XRT/include/', 'I$XILINX_VIVADO/include/']
            #build_directory='/home/centos/project/build/', verbose=True
            )

import host

# Get data path
data_dir = Path('~/data').expanduser()

# Create dataset
input_size = 224
test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = datasets.ImageNet(data_dir, split='val', transform=test_transform)

# Create dataloaders
batch1_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=1)
batch16_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16)

class ConvOnFPGA(torch.nn.Module):
   def __init__(self, in_channels, out_channels, weight, bias):
       super(ConvOnFPGA, self).__init__()
       self.in_channels = in_channels
       self.out_channels = out_channels
       self.kernel_size = (3, 3)
       self.stride = 1
       self.padding = 1
       self.dilation = 1
       self.bias = True
       self.weight = weight
       self.bias = bias

   def forward(self, input):
       b = self.bias.reshape(1,-1,1,1)
       return host.convolution_fpga(input, self.weight, self.stride, self.padding) + b
       

# ------------- Replace Conv2D Layers in VGG with our Layers ----------

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
for param in vgg16.features.parameters():
   param.requires_grad = False

#conv2d_layers = [19]
conv2d_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
for i in conv2d_layers:
   old_layer = vgg16.features[i]
   fpga_layer = ConvOnFPGA(old_layer.in_channels, old_layer.out_channels, old_layer.weight, old_layer.bias)
   vgg16.features[i] = fpga_layer


def comparison(t1, t2):
    return torch.isclose(t1, t2, atol=0.0001, rtol=0)



#Run loop that performs 16 inferences of batch size 1

tic = time.perf_counter()
corrects_top1 = 0
corrects_top5 = 0
for i, data in zip(range(1), batch1_loader):
    print("Inference of Batch " + str(i) + ":")
    output = vgg16(data[0])
    _, prediction_top1 = torch.max(output, 1)
    _, prediction_top5 = torch.topk(output, 5)
    corrects_top1 += torch.sum(comparison(prediction_top1, data[1].data))
    corrects_top5 += torch.sum(comparison(prediction_top5, data[1].data))

acc_one_top1 = 100*corrects_top1.double()/16
acc_one_top5 = 100*corrects_top5.double()/16

# end timer
toc = time.perf_counter()

# Print results
runtime_1 = toc - tic
print(f'Runtime(1) = {runtime_1:0.4f} seconds')
print(f'Top-1 Accuracy (Batch Size 1) = {acc_one_top1:0.4f}%')
print(f'Top-5 Accuracy (Batch Size 1) = {acc_one_top5:0.4f}%')

#Run loop that performs 1 inference for batch size 16

tic = time.perf_counter()
corrects_top1 = 0
corrects_top5 = 0
for i, data in zip(range(1), batch16_loader):
    print("Inference of Batch " + str(i) + ":")
    output = vgg16(data[0])
    _, prediction_top1 = torch.max(output, 1)
    _, prediction_top5 = torch.topk(output, 5)
    corrects_top1 += torch.sum(comparison(prediction_top1, data[1].data))
    corrects_top5 += torch.sum(comparison(prediction_top5, data[1].data))

acc_two_top1 = 100*corrects_top1.double()/16
acc_two_top5 = 100*corrects_top5.double()/16

# end timer
toc = time.perf_counter()

# Print results
runtime_16 = toc - tic
print(f'Runtime(2) = {runtime_16:0.4f} seconds')
print(f'Top-1 Accuracy (Batch Size 16) = {acc_two_top1:0.4f}%')
print(f'Top-5 Accuracy (Batch Size 16) = {acc_two_top5:0.4f}%')

########################################################################

# Print score
print(f'FOM = {((runtime_1 + runtime_16) / 2):0.4f}')