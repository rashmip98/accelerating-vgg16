import torch
from torch.utils.cpp_extension import load

import torch.nn as nn
import time
import os
import subprocess
#subprocess.call('./vitis.sh')

from torchvision import datasets, models, transforms
from pathlib import Path

host = load(name="host", sources=["q4/src/host.cpp"], extra_cflags=['-Wall', '-O0', '-g', '-std=c++14'],
            extra_ldflags=['-L$XILINX_XRT/lib/', '-lOpenCL', '-lpthread', '-lrt', '-lstdc++'],
            extra_include_paths=['I$XILINX_XRT/include/', 'I$XILINX_VIVADO/include/']
            #build_directory='/home/centos/project/build/', verbose=True
            )

#import host_trial

#input_tensor = torch.randint(1,3,(1,3,4,4), dtype = torch.float) #nchw
#kernel_tensor = torch.randint(1,3,(1,3,3,3), dtype = torch.float) #kcrs
#op nkpq
input_tensor = torch.rand(2,15,16,16) # 1 512 14 14
kernel_tensor = torch.rand(8,15,3,3) # 512 512 3 3
stride = 1
padding = 1
rtol = 1e-01
atol = 1e-08
conv_result = nn.functional.conv2d(input_tensor, kernel_tensor, padding = 1)

final_img = host.convolution_fpga(input_tensor, kernel_tensor, stride, padding)
print("Convolution PyTorch result: ")
print(conv_result)
print(torch.isclose(final_img, conv_result, rtol))
#print("Input tensor")
# print(input_tensor)
# print("Kernel tensor")
# print(kernel_tensor)
print("Tensor Product")
print(final_img.size())
print(final_img)