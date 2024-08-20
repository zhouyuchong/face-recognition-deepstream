################################################################################
# Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

WITH_OPENCV?=0

NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc
CXX:= g++
SRCS:= nvdsinfer_context_impl.cpp  nvdsinfer_context_impl_capi.cpp \
       nvdsinfer_context_impl_output_parsing.cpp \
       nvdsinfer_func_utils.cpp nvdsinfer_model_builder.cpp \
       nvdsinfer_backend.cpp nvdsinfer_conversion.cu

INCS:= $(wildcard *.h)
LIB:=libnvds_infer.so

NVDS_VERSION:=6.1

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

CFLAGS+= -fPIC -Wno-deprecated-declarations -std=c++14 \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I ../../includes -DNDEBUG

ifeq ($(WITH_OPENCV),1)
CFLAGS+= -DWITH_OPENCV
ifeq ($(shell uname -m), aarch64)
CFLAGS+=-I/usr/include/opencv4
endif
endif

LIBS := -shared -Wl,-no-undefined \
	 -lnvinfer -lnvinfer_plugin -lnvonnxparser -lnvparsers -lpthread \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

ifeq ($(WITH_OPENCV),1)
LIBS += -lopencv_objdetect -lopencv_imgproc -lopencv_core
endif

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
       -lnvds_inferlogger -lnvds_inferutils -ldl \
       -Wl,-rpath,$(LIB_INSTALL_DIR)


OBJS:= $(SRCS:.cpp=.o)
OBJS:= $(OBJS:.cu=.o)

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	@echo $(CFLAGS)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(LIB): $(OBJS) $(DEP) Makefile
	@echo $(CFLAGS)
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(LIB_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)
