import torch
from torch.utils.cpp_extension import load

# Imports
import time
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from pathlib import Path
#import subprocess
#subprocess.call('./vitis.sh')

# Initialization
# Perform any initialization procedures such as 
#   Binding
#   Flashing FPGA
#   Instantiating your FPGA accelerated VGG16 net
#   Any other initialization that shouldn't be timed
host_trial = load(name="host_trial", sources=["q4/src/host_trial.cpp"], extra_cflags=['-Wall', '-O0', '-g', '-std=c++14'],
            extra_ldflags=['-L$XILINX_XRT/lib/', '-lOpenCL', '-lpthread', '-lrt', '-lstdc++'],
            extra_include_paths=['I$XILINX_XRT/include/', 'I$XILINX_VIVADO/include/']
            #build_directory='/home/centos/project/build/', verbose=True
            )

import host_trial

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
#batch32_loader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=32)

class ConvOnFPGA(torch.nn.Module):
   def __init__(self, in_channels, out_channels, weight, bias):
       super(ConvOnFPGA, self).__init__()
       self.in_channels = in_channels
       self.out_channels = out_channels
       self.kernel_size = (3, 3)
       self.stride = 1
       self.padding = 1
       self.dilation = 1
       self.groups = 1
       self.bias = True
       self.padding_mode = 'zeros'
       self.weight = weight
       self.bias = bias

   def forward(self, input):
       return host_trial.convolution_fpga(input, self.weight, self.stride, self.padding, self.bias)

# ------------- Replace Conv2D Layers in VGG with our Layers ----------

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
for param in vgg16.features.parameters():
   param.requires_grad = False

conv2d_layers = [28]
#conv2d_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
for i in conv2d_layers:
   old_layer = vgg16.features[i]
   fpga_layer = ConvOnFPGA(old_layer.in_channels, old_layer.out_channels, old_layer.weight, old_layer.bias)
   vgg16.features[i] = fpga_layer

#input_tensor, kernel_tensor, stride, padding

########################################################################
# Batch size 1 benchmarking
# Get a single (input, label) tuple of batch size 1 from your dataloader
# Perform 16 inferences of batch size 1
runtime_1 = 0
running_corrects_1 = 0
running_five_corrects_1 = 0
print("Batch size 1")
for i, (inputs, labels) in zip(range(1), batch1_loader):
    print("Inference: ")
    print(i)
    tic = time.perf_counter()
    outputs = vgg16(inputs)
    toc = time.perf_counter()
    runtime_1 += toc - tic # Accumulate running time

    _, predicted_1 = torch.max(outputs, 1)
    _, predicted_5 = torch.topk(outputs, 5)
    running_corrects_1 += torch.sum(predicted_1 == labels.data)
    running_five_corrects_1 += torch.sum(predicted_5 == labels.data)

acc_one_1 = 100 * running_corrects_1.double() / 16
acc_five_1 = 100 * running_five_corrects_1.double() / 16

# Print results
print(f'Runtime(1) = {runtime_1:0.4f} seconds')
print(f'Top-1 Accuracy(1) = {acc_one_1:0.4f}%')
print(f'Top-5 Accuracy(1) = {acc_five_1:0.4f}%')


input_tensor = torch.randint(1, 3, (2,5,4,4), dtype = torch.float) #nchw
kernel_tensor = torch.randint(1, 3, (2,5,3,3), dtype = torch.float) #kcrs
#op nkpq
stride = 1
padding = 0 

conv_result = nn.functional.conv2d(input_tensor, kernel_tensor)

final_img = host_trial.convolution_fpga(input_tensor, kernel_tensor, stride, padding)
print("Convolution PyTorch result: ")
print(conv_result)
print(torch.isclose(final_img, conv_result))
#print("Input tensor")
# print(input_tensor)
# print("Kernel tensor")
# print(kernel_tensor)
print("Tensor Product")
print(final_img.size())
print(final_img)