#!/bin/bash
# exit when any command fails
set -e

echo 'This script expects to be ran from its parent directory and is not the most robust'

export TARGET=hw
export NAME=multadd
export PROJECT_DIR=q4
export BUCKET_NAME=ese539.41574243

pwd
echo Fixing localization...
export LC_ALL="C"
echo Fixed localization

echo Sourcing vitis-setup...
# Move to aws-fpga temporarily
pushd ~/aws-fpga
source vitis_setup.sh
#source vitis_runtime_setup.sh
popd
echo Sourced vitis-setup

CXX=/usr/local/bin/g++
export CXX
echo Using C++ Compiler $CXX

systemctl is-active --quiet mpd || sudo systemctl start mpd

sleep 180s
# while [[ "$flg" -eq 1 ]]
# do 
    
#     $val=(systemctl -q is-active mpd)
#     echo $val

#     if (!(systemctl is-active mpd))
#     then
#         echo donee
#         flg=0
#     else
#         echo sleeping
#         sleep 1s
#     fi
# done

echo Running host code with kernel...
#need to copy xrt.ini file to execution directory in order to profile
sudo cp ${PROJECT_DIR}/build/emconfig.json /usr/bin/.
python3 ${PROJECT_DIR}/src/benchmark.py
echo Finished run

mv profile_summary.csv ${PROJECT_DIR}/reports/${NAME}.${TARGET}/ 2>/dev/null || true
mv timeline_trace.csv ${PROJECT_DIR}/reports/${NAME}.${TARGET}/ 2>/dev/null  || true
mv *.run_summary ${PROJECT_DIR}/reports/${NAME}.${TARGET}/ 2>/dev/null  || true
mv profile_kernels.csv ${PROJECT_DIR}/reports/${NAME}.${TARGET}/ 2>/dev/null  || true
mv timeline_kernels.csv ${PROJECT_DIR}/reports/${NAME}.${TARGET}/ 2>/dev/null  || true
echo Moved run summaries to report directory