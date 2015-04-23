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
    int arrayLength;       /* nbDims-2 size */ 
    int *padA;
    int *filterStrideA;
    int *upscaleA;
};


struct cdnnPoolingStruct {
	/* data */
    cdnnPoolingMode_t mode;
    int nDimension;
    int *windowDimA;
    int *paddingA;
    int *strideA;
};

/* e.g. 
   4D filter
    filterDimA[3]: number of output feature maps
    filterDimA[2]: number of input feature maps
    filterDimA[1]: height of each input filter
    filterDimA[0]: width of  each input fitler
*/
struct cdnnFilterStruct {
	/* data */
    cdnnDataType_t dataType;  // image data type
    int nDimension;           // dimension
    int *filterDimA;         // filter Dims
};

#endif
