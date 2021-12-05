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

export TARGET=sw_emu
export NAME=multadd
export PROJECT_DIR=q4
export BUCKET_NAME=ese539.41574243
export DATA_SIZE=32

mkdir -p ${PROJECT_DIR}/build
mkdir -p ${PROJECT_DIR}/logs
mkdir -p ${PROJECT_DIR}/reports

echo Created relevant directories

make -f ${PROJECT_DIR}/Makefile

if [ $TARGET == "hw" ]
then
$VITIS_DIR/tools/create_vitis_afi.sh -xclbin=${PROJECT_DIR}/build/${NAME}.${TARGET}.xclbin \
		-o=${PROJECT_DIR}/${NAME} -s3_bucket=${BUCKET_NAME} \
        -s3_dcp_key=dcp -s3_logs_key=logs
else
    emconfigutil --platform $AWS_PLATFORM --od ${PROJECT_DIR}/build
    export XCL_EMULATION_MODE=${TARGET}  
    echo Set emulation variables
    echo Running host code with kernel...
    # need to copy xrt.ini file to execution directory in order to profile
    cp ${PROJECT_DIR}/xrt.ini .
	echo Running host code with kernel...
	./${PROJECT_DIR}/build/host ./${PROJECT_DIR}/build/${NAME}.${TARGET}.xclbin ${DATA_SIZE}
	echo Finished run
	mv profile_summary.csv ${PROJECT_DIR}/reports/${NAME}.${TARGET}/
	mv timeline_trace.csv ${PROJECT_DIR}/reports/${NAME}.${TARGET}/
	mv *.run_summary ${PROJECT_DIR}/reports/${NAME}.${TARGET}/
fi