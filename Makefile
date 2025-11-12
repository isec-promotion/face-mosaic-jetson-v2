################################################################################
# YOLOv8 Custom Parser for DeepStream
################################################################################

CC:= g++

CFLAGS:= -Wall -std=c++11

CFLAGS+= -shared -fPIC

CFLAGS+= -I/opt/nvidia/deepstream/deepstream-7.1/sources/includes \
         -I /usr/local/cuda-$(CUDA_VER)/include

LIBS:= -lnvinfer
LFLAGS:= -Wl,--start-group $(LIBS) -Wl,--end-group

SRCFILES:= nvdsinfer_yolov8_parser.cpp

TARGET_LIB:= libnvdsinfer_custom_impl_yolo.so

all: $(TARGET_LIB)

$(TARGET_LIB) : $(SRCFILES)
	$(CC) -o $@ $^ $(CFLAGS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)

.PHONY: all clean
