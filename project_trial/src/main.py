import time
import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from torchvision import datasets, models, transforms
from pathlib import Path

# ------------- Define our FPGA Accelerated Conv2D layer, and wrap in PyTorch Classes ----------
host = load(name='host', 
                   sources=['/home/centos/project_trial/src/host.cpp'],
                   extra_cflags=['-Wall', '-O2', '-g', '-std=c++14'],
                   extra_ldflags=['-L$XILINX_XRT/lib/', '-lOpenCL', '-lpthread', '-lrt', '-lstdc++'],
                   extra_include_paths=['$XILINX_XRT/include/', '$XILINX_VIVADO/include/'],
                   #build_directory='/home/centos/project_trial/build/'
                   #verbose=True
                   ) 
import host 
class FPGAConv2D(torch.nn.Module):
   def __init__(self, in_channels, out_channels, weight, bias):
       super(FPGAConv2D, self).__init__()
       self.in_channels = in_channels
       self.out_channels = out_channels
       self.kernel_size = (3, 3)
       self.stride = (1, 1)
       self.padding = (1, 1)
       self.dilation = 1
       self.groups = 1
       self.bias = True
       self.padding_mode = 'zeros'
       self.weight = weight
       self.bias = bias

   def forward(self, input_tensor):
       return host.fpga_conv2d(input_tensor, self.weight, self.bias)

# ------------- Replace Conv2D Layers in VGG with our Layers ----------

vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
for param in vgg16.features.parameters():
   param.requires_grad = False

conv2d_layers = [28]
#conv2d_layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
for i in conv2d_layers:
   old_layer = vgg16.features[i]
   fpga_layer = FPGAConv2D(old_layer.in_channels, old_layer.out_channels, old_layer.weight, old_layer.bias)
   vgg16.features[i] = fpga_layer

# -------------------------- Benchmark -----------------------
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

batch1_loader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=1)

# Equality up to epsilon = 0.0001
def float_eq(t1, t2):
    return torch.isclose(t1, t2, atol=0.0001, rtol=0)

########################################################################
# Batch size 1 benchmarking

# Perform 16 inferences of batch size 1
runtime_1 = 0
running_corrects_1 = 0
running_five_corrects_1 = 0
print("Batch size 1")
for i, (inputs, labels) in zip(range(16), batch1_loader):
    print("Inference: ")
    print(i)
    tic = time.perf_counter()
    outputs = vgg16(inputs)
    toc = time.perf_counter()
    runtime_1 += toc - tic # Accumulate running time

    _, predicted_1 = torch.max(outputs, 1)
    _, predicted_5 = torch.topk(outputs, 5)
    running_corrects_1 += torch.sum(float_eq(predicted_1, labels.data))
    running_five_corrects_1 += torch.sum(float_eq(predicted_5, labels.data))

acc_one_1 = 100 * running_corrects_1.double() / 16
acc_five_1 = 100 * running_five_corrects_1.double() / 16

# Print results
print(f'Runtime(1) = {runtime_1:0.4f} seconds')
print(f'Top-1 Accuracy(1) = {acc_one_1:0.4f}%')
print(f'Top-5 Accuracy(1) = {acc_five_1:0.4f}%')
