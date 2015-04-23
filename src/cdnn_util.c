#include <stdio.h>
#include <stdlib.h>
#include "cdnn.h"
#include "cdnn_core.h"
#include "cdnn_util.h"

bool CDNNWINAPI isContiguous( const cdnnTensorDescriptor_t tensorDesc)
{
    long stride = 1;
    for (int i=0; i<tensorDesc->nDimension; ++i) {
        if ( stride != tensorDesc->strideA[i] )  return false;
        stride = stride * tensorDesc->dimA[i];
    }
    return true;
}

bool CDNNWINAPI isSameSize( const cdnnTensorDescriptor_t srcDesc, 
                            const cdnnTensorDescriptor_t destDesc
                          )
{
    if ( srcDesc->nDimension!=destDesc->nDimension )
        return false;
    int nDim = srcDesc->nDimension;
    for (int i=0; i<nDim; ++i) {
        if ( srcDesc->dimA[i] != destDesc->dimA[i] )  return false;
    }
    return true;
}


void* CDNNWINAPI allocTensor(cdnnTensorDescriptor_t tensorDesc)
{
    long memSize = 0;
    if (tensorDesc->dataType==CDNN_DATA_FLOAT)
        memSize = sizeof(float)*tensorDesc->size;
    else
        memSize = sizeof(double)*tensorDesc->size;
    return malloc(memSize);
}

void CDNNWINAPI printTensorDesc(cdnnTensorDescriptor_t tensorDesc)
{
    fprintf(stdout, "\nTensor Format: %s\n", 
            tensorDesc->format==CDNN_TENSOR_NCHW  ? "CDNN_TENSOR_NCHW" : "CDNN_TENSOR_NHWC");
    fprintf(stdout, "Data Format: %s\n", 
            tensorDesc->dataType==CDNN_DATA_FLOAT ? "CDNN_DATA_FLOAT" : "CDNN_DATA_DOUBLE");
    fprintf(stdout, "Size: %ld\n", tensorDesc->size);
    fprintf(stdout, "Dimension: %d\n", tensorDesc->nDimension);
     for (int i=0; i<tensorDesc->nDimension; ++i) {
        fprintf(stdout, "\tSize of dim %d: %d\n", i, tensorDesc->dimA[i]);
        //fprintf(stdout, "\tStride of dim %d: %d\n", i, tensorDesc->strideA[i]);
    }
}

void CDNNWINAPI printTensor(cdnnTensorDescriptor_t tensorDesc, void *dataArray)
{
    for (int i=0; i<tensorDesc->dimA[3]; ++i) {
        for (int j=0; j<tensorDesc->dimA[2]; ++j) {
            fprintf(stdout, "(%d,%d,:,:)\n", i, j);
            for (int k=0; k<tensorDesc->dimA[1]; ++k) {
                for (int l=0; l<tensorDesc->dimA[0]; ++l) {
                    long idx = i*tensorDesc->strideA[3] + 
                               j*tensorDesc->strideA[2] +
                               k*tensorDesc->strideA[1] +
                               l*tensorDesc->strideA[0];
                    //fprintf(stdout, "%ld\t", idx);
                    if (tensorDesc->dataType==CDNN_DATA_FLOAT)
                        fprintf(stdout, "%f ", ((float*)dataArray)[idx]);
                    else
                        fprintf(stdout, "%f ", ((double*)dataArray)[idx]);
                }
                fprintf(stdout, "\n");
            }
        }
    }
}





