#!/bin/bash
#NAME := 
#PROJECT_DIR := .
BUILD_DIR := ${PROJECT_DIR}/build
SOURCE_DIR := ${PROJECT_DIR}/src
CONFIG_DIR := ${PROJECT_DIR}/configs
LOG_DIR := ${PROJECT_DIR}/logs
REPORT_DIR := ${PROJECT_DIR}/reports

VPP := v++
EMCONFIGUTIL := emconfigutil

# host sources
HOST_SRC := ${SOURCE_DIR}/host.cpp

# host bin
HOST := ${BUILD_DIR}/host

# kernel sources
KERNEL_SRC := ${SOURCE_DIR}/${NAME}.cpp

# Xilinx kernel objects
XOS := ${BUILD_DIR}/${NAME}.${TARGET}.xo

# Kernel bin
XCLBIN := ${BUILD_DIR}/${NAME}.${TARGET}.xclbin
EMCONFIG_FILE := ${BUILD_DIR}/emconfig.json

# Host options
GCC_OPTS := -I${XILINX_XRT}/include/ -I${XILINX_VIVADO}/include/ -Wall -O0 -g -std=c++11 -L${XILINX_XRT}/lib/ -lOpenCL -lpthread -lrt -lstdc++

# VPP Linker options
VPP_INCLUDE_OPTS := -I ${PROJECT_DIR}/src 

# VPP common options
VPP_COMMON_OPTS := --config ${CONFIG_DIR}/design.cfg \
				   --log_dir ${LOG_DIR} \
				   --report_dir ${REPORT_DIR} \
				   --platform ${AWS_PLATFORM} \
				   --compile --kernel ${NAME}	

.PHONY: all
all: emulate

# error check target variable
ifndef TARGET
	$(error TARGET is not defined. It should be 'sw_emu' or 'hw_emu')
endif

ifndef NAME
	$(error NAME is not defined. It should be kernel name')
endif

${HOST}: ${HOST_SRC}
	g++ ${GCC_OPTS} -o $@ $+
	@echo 'Compiled Host Executable: ${HOST_EXE}'

${XOS}: ${KERNEL_SRC}
	@${RM} $@
	${VPP} -t ${TARGET} --config ${CONFIG_DIR}/design.cfg \
		--log_dir ${LOG_DIR} \
		--report_dir ${REPORT_DIR} \
		--platform ${AWS_PLATFORM} \
		--compile --kernel ${NAME} \
		-I ${SOURCE_DIR} -o $@ $+
	mv ${BUILD_DIR}/*compile_summary ${REPORT_DIR}/${NAME}.${TARGET}/

	@echo 'Compiled kernel'

${XCLBIN}: ${XOS}
	${VPP} -t ${TARGET} --config ${CONFIG_DIR}/design.cfg \
		--log_dir ${LOG_DIR} \
		--report_dir ${REPORT_DIR} \
		--platform ${AWS_PLATFORM} \
		--link -o $@ $+
	mv ${BUILD_DIR}/*link_summary ${REPORT_DIR}/${NAME}.${TARGET}/

	@echo 'Linked kernel'

${EMCONFIG_FILE}:
	${EMCONFIGUTIL} --platform ${AWS_PLATFORM} --od ${BUILD_DIR}

emulate: ${HOST} ${XCLBIN} ${EMCONFIG_FILE}

