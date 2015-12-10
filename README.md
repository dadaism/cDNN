# cDNN

This is a cuDNN like APIs library for CNN on CPUs.

## Supported Functionalities
Convolution
Pooling
Fully-connection
Activation

## Structure of folders

├── example
│   ├── act_layer_example.c
│   ├── conv_layer_example.c
│   ├── convnet_example.c
│   ├── Makefile
│   ├── pool_layer_example.c
│   ├── tensor_example.c
│   └── test.sh
├── lib
│   ├── activation.o
│   ├── cdnn.o
│   ├── convolution.o
│   ├── libcdnn.so
│   ├── pooling.o
│   ├── tensor.o
│   └── util.o
├── LICENCE.md
├── README.md
├── sample
└── src
    ├── cdnn_activation.c
    ├── cdnn.c
    ├── cdnn_convolution.c
    ├── cdnn_core.h
    ├── cdnn.h
    ├── cdnn_pooling.c
    ├── cdnn_tensor.c
    ├── cdnn_util.c
    ├── cdnn_util.h
    └── Makefile


## Under construction..
