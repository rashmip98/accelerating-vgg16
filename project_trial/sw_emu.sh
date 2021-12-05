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

mkdir -p project_trial/build
mkdir -p project_trial/logs
mkdir -p project_trial/reports
echo Created relevant directories

CXX=/usr/local/bin/g++
export CXX
echo Using C++ Compiler $CXX

echo Compiling FPGA kernel...
v++ -t sw_emu --config project_trial/configs/design.cfg \
    --log_dir project_trial/logs \
    --report_dir project_trial/reports \
    --platform $AWS_PLATFORM \
    --compile --kernel multadd \
    -I project_trial/src -o project_trial/build/multadd.sw_emu.xo project_trial/src/multadd.cpp
echo Compiled FPGA kernel

echo Moved compile summary to reports directory
mv project_trial/build/*compile_summary project_trial/reports/multadd.sw_emu/

echo Linking FPGA kernel with platform...
v++ -t sw_emu --config project_trial/configs/design.cfg \
    --log_dir project_trial/logs \
    --report_dir project_trial/reports \
    --platform $AWS_PLATFORM \
    --link -o project_trial/build/multadd.sw_emu.xclbin \
    project_trial/build/multadd.sw_emu.xo
echo Linked FPGA kernel

echo Moved link summary to reports directory
mv project_trial/build/*link_summary project_trial/reports/multadd.sw_emu/

# Set software emulation
echo Setting emulation variables...
emconfigutil --platform $AWS_PLATFORM --od project_trial/build
export XCL_EMULATION_MODE=sw_emu  
echo Set emulation variables

echo Running host code with kernel...
# need to copy xrt.ini file to execution directory in order to profile
cp project_trial/xrt.ini .
sudo cp project_trial/build/emconfig.json /usr/bin/.
python3 project_trial/src/main.py

echo Finished run

mv profile_summary.csv project_trial/reports/multadd.sw_emu/ 2>/dev/null || true
mv timeline_trace.csv project_trial/reports/multadd.sw_emu/ 2>/dev/null  || true
mv *.run_summary project_trial/reports/multadd.sw_emu/ 2>/dev/null  || true
mv profile_kernels.csv project_trial/reports/multadd.sw_emu/ 2>/dev/null  || true
mv timeline_kernels.csv project_trial/reports/multadd.sw_emu/ 2>/dev/null  || true
echo Moved run summaries to report directory
