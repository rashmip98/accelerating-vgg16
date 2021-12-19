#!/bin/bash
# exit when any command fails
set -e

echo 'This script expects to be ran from its parent directory and is not the most robust'

echo Sourcing vitis-setup...
# Move to aws-fpga temporarily
pushd ~/aws-fpga
source vitis_setup.sh
popd
echo Sourced vitis-setup

echo Fixing localization...
export LC_ALL="C"
echo Fixed localization

mkdir -p q4/build
mkdir -p q4/logs
mkdir -p q4/reports
echo Created relevant directories

CXX=/usr/local/bin/g++
export CXX
echo Using C++ Compiler $CXX

echo Compiling FPGA kernel...
v++ -t sw_emu --config q4/configs/design.cfg \
    --log_dir q4/logs \
    --report_dir q4/reports \
    --platform $AWS_PLATFORM \
    --compile --kernel multadd \
    -I q4/src -o q4/build/multadd.sw_emu.xo q4/src/multadd.cpp
echo Compiled FPGA kernel

echo Moved compile summary to reports directory
mv q4/build/*compile_summary q4/reports/multadd.sw_emu/

echo Linking FPGA kernel with platform...
v++ -t sw_emu --config q4/configs/design.cfg \
    --log_dir q4/logs \
    --report_dir q4/reports \
    --platform $AWS_PLATFORM \
    --link -o q4/build/multadd.sw_emu.xclbin \
    q4/build/multadd.sw_emu.xo
echo Linked FPGA kernel

echo Moved link summary to reports directory
mv q4/build/*link_summary q4/reports/multadd.sw_emu/

# Set software emulation
echo Setting emulation variables...
emconfigutil --platform $AWS_PLATFORM --od q4/build
export XCL_EMULATION_MODE=sw_emu  
echo Set emulation variables

echo Running host code with kernel...
# need to copy xrt.ini file to execution directory in order to profile
cp q4/xrt.ini .
sudo cp q4/build/emconfig.json /usr/bin/.
python3 q4/src/python_trial.py
#python3 q4/src/benchmark.py
echo Finished run

mv profile_summary.csv q4/reports/multadd.sw_emu/ 2>/dev/null || true
mv timeline_trace.csv q4/reports/multadd.sw_emu/ 2>/dev/null  || true
mv *.run_summary q4/reports/multadd.sw_emu/ 2>/dev/null  || true
mv profile_kernels.csv q4/reports/multadd.sw_emu/ 2>/dev/null  || true
mv timeline_kernels.csv q4/reports/multadd.sw_emu/ 2>/dev/null  || true
echo Moved run summaries to report directory