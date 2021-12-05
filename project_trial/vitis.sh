#!/bin/bash

set -e

echo 'This script expects to be ran from its parent directory and is not the most robust'

export LC_ALL="C"

echo Sourcing vitis-setup...
# Move to aws-fpga temporarily
pushd ~/aws-fpga
source vitis_setup.sh
popd
echo Sourced vitis-setup

vitis_analyzer
