#ifndef __CDNN_CORE_H__
#define __CDNN_CORE_H__
#include "cdnn.h"

struct cdnnContext {
	



};

struct cdnnTensorStruct {
	/* data */
    cdnnTensorFormat_t  format;  //
    cdnnDataType_t dataType;     // image data type
    long size;
    int nDimension;
    int *dimA;
    int *strideA;
};

struct cdnnConvolutionStruct {
	/* data */
    cdnnConvolutionMode_t mode;
    int padHeight;
    int padWidth;
    int strideVertical;
    int strideHorizontal;
    int upscaleX;
    int upscaleY;
};


struct cdnnPoolingStruct {
	/* data */
    cdnnPoolingMode_t mode;
    int nDimension;
    int *windowDimA;
    int *paddingA;
    int *strideA;
};

struct cdnnFilterStruct {
	/* data */
    cdnnDataType_t dataType;  // image data type
    int outputs;              // number of output feature maps
    int inputs;               // number of input feature maps
    int filter_height;        // height of each input filter
    int filter_width;         // width of  each input fitler
};

#endif
