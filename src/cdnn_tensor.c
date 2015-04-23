#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cdnn_core.h"
#include "cdnn.h"
#include "cdnn_util.h"

#ifndef CDNNWINAPI
#ifdef _WIN32
#define CDNNWINAPI __stdcall
#else
#define CDNNWINAPI
#endif
#endif

/**
 * Create an instance of a generic Tensor descriptor 
 */
cdnnStatus_t cdnnCreateTensorDescriptor( cdnnTensorDescriptor_t   *tensorDesc )
{
    *tensorDesc = (cdnnTensorDescriptor_t)malloc(sizeof(struct cdnnTensorStruct));
    if (tensorDesc!=NULL) 
        return CDNN_STATUS_SUCCESS;
    else 
        return CDNN_STATUS_ALLOC_FAILED;
}

/**
 * Set a 4D Tensor. It is similar to torch tensor.
 */
cdnnStatus_t CDNNWINAPI cdnnSetTensor4dDescriptor(   cdnnTensorDescriptor_t   tensorDesc,
                                                     cdnnTensorFormat_t  format,
                                                     cdnnDataType_t dataType, // image data type
                                                     int n,        // number of inputs (batch size)
                                                     int c,        // number of input feature maps
                                                     int h,        // height of input section
                                                     int w         // width of input section
                                                  )
{
    /* Currently not supported */
    assert( format==CDNN_TENSOR_NCHW );

    tensorDesc->nDimension = 4;
    tensorDesc->format     = format;
    tensorDesc->dataType   = dataType;
    tensorDesc->dimA       = (int *)malloc(sizeof(int)*4);
    tensorDesc->strideA    = (int *)malloc(sizeof(int)*4);
    if ( tensorDesc->dimA==NULL || tensorDesc->strideA==NULL )
        return CDNN_STATUS_ALLOC_FAILED;
    tensorDesc->dimA[0] = w; tensorDesc->strideA[0] = 1;
    tensorDesc->dimA[1] = h; tensorDesc->strideA[1] = w;
    tensorDesc->dimA[2] = c; tensorDesc->strideA[2] = h * tensorDesc->strideA[1];
    tensorDesc->dimA[3] = n; tensorDesc->strideA[3] = c * tensorDesc->strideA[2];
    tensorDesc->size = n * c * h * w;
    return CDNN_STATUS_SUCCESS;
}

/**
 * Set a 4D Tensor with explicit stride. It is similar to torch tensor.
 */
cdnnStatus_t CDNNWINAPI cdnnSetTensor4dDescriptorEx( cdnnTensorDescriptor_t tensorDesc,
                                                     cdnnDataType_t dataType, // image data type
                                                     int n,        // number of inputs (batch size)
                                                     int c,        // number of input feature maps
                                                     int h,        // height of input section
                                                     int w,        // width of input section
                                                     int nStride,
                                                     int cStride,
                                                     int hStride,
                                                     int wStride
                                                   )
{
    /* Currently, padding is not supported */
    tensorDesc->nDimension = 4;

    tensorDesc->dimA       = (int *)malloc(sizeof(int)*4);
    tensorDesc->strideA    = (int *)malloc(sizeof(int)*4);
    if ( tensorDesc->dimA==NULL || tensorDesc->strideA==NULL )
        return CDNN_STATUS_ALLOC_FAILED;
    tensorDesc->dimA[0] = w; tensorDesc->strideA[0] = wStride;
    tensorDesc->dimA[1] = h; tensorDesc->strideA[1] = hStride;
    tensorDesc->dimA[2] = c; tensorDesc->strideA[2] = cStride;
    tensorDesc->dimA[3] = n; tensorDesc->strideA[3] = nStride;

    return CDNN_STATUS_SUCCESS;
}

/**
 * Get a 4D tensor.
 */
cdnnStatus_t CDNNWINAPI cdnnGetTensor4dDescriptor(   const cdnnTensorDescriptor_t tensorDesc,
                                                     cdnnDataType_t *dataType, // image data type
                                                     int *n,        // number of inputs (batch size)
                                                     int *c,        // number of input feature maps
                                                     int *h,        // height of input section
                                                     int *w,        // width of input section
                                                     int *nStride,
                                                     int *cStride,
                                                     int *hStride,
                                                     int *wStride
                                                 )
{
    *dataType = tensorDesc->dataType;          
    *n = tensorDesc->dimA[3];      
    *c = tensorDesc->dimA[2];     
    *h = tensorDesc->dimA[1];      
    *w = tensorDesc->dimA[0];
    *nStride = tensorDesc->strideA[3];
    *cStride = tensorDesc->strideA[2];
    *hStride = tensorDesc->strideA[1];
    *wStride = tensorDesc->strideA[0];

    return CDNN_STATUS_SUCCESS;

}

/**
 * Set a ND tensor descriptor.
 */
cdnnStatus_t CDNNWINAPI cdnnSetTensorNdDescriptor(  cdnnTensorDescriptor_t tensorDesc,
                                                    cdnnDataType_t dataType,
                                                    int nbDims,
                                                    const int dimA[],
                                                    const int strideA[]
                                                 )
{
    tensorDesc->dataType = dataType;
    tensorDesc->nDimension = nbDims;
    for (int i=0; i<nbDims; ++i) {
        tensorDesc->dimA[i] = dimA[i];
        tensorDesc->strideA[i] = strideA[i];   
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * Get a ND tensor descriptor.
 */
cdnnStatus_t CDNNWINAPI cdnnGetTensorNdDescriptor(  const cdnnTensorDescriptor_t tensorDesc,
                                                    int nbDimsRequested,
                                                    cdnnDataType_t *dataType,
                                                    int *nbDims,
                                                    int dimA[],
                                                    int strideA[]
                                                 )
{
    *dataType = tensorDesc->dataType;
    for (int i=0; i<*nbDims; ++i) {
        dimA[i] = tensorDesc->dimA[i];
        strideA[i] = tensorDesc->strideA[i];   
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * Destroy an instance of Tensor4d descriptor 
 */
cdnnStatus_t CDNNWINAPI cdnnDestroyTensorDescriptor( cdnnTensorDescriptor_t tensorDesc )
{
    free(tensorDesc->dimA);
    free(tensorDesc->strideA);
    free(tensorDesc);
    return CDNN_STATUS_SUCCESS;
}

/**
 * Tensor layout conversion helper (dest = alpha * src + beta * dest)
 * Descriptors need to have the same dimemsions but not necessarily the
 * same strides.
 */
cdnnStatus_t CDNNWINAPI cdnnTransformTensor( cdnnHandle_t                    handle,
                                             const void                      *alpha,
                                             const cdnnTensorDescriptor_t    srcDesc,
                                             const void                      *srcData,
                                             const void                      *beta,
                                             const cdnnTensorDescriptor_t    destDesc,
                                             void                            *destData
                                           )
{
    assert( isSameSize(srcDesc, destDesc) );
    int nDim = srcDesc->nDimension;
    long size = srcDesc->strideA[nDim-1] * srcDesc->dimA[nDim-1] ;
    if ( srcDesc->dataType==CDNN_DATA_FLOAT ) {
        float a = *((float*)alpha); 
        float b = *((float*)beta);
        float *dData = (float*)destData;
        float *sData = (float*)srcData;
        for (long i=0; i<size; ++i) {
            dData[i] = a * sData[i] + b * dData[i];
        }
    }
    else {  // CDNN_DATA_DOUBLE
        double a = *((double*)alpha);
        double b = *((double*)beta);
        double *dData = (double*)destData;
        double *sData = (double*)srcData;
        for (long i=0; i<size; ++i) {
            dData[i] = a * sData[i] + b * dData[i];
        }
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * cdnnAddMode_t:
 *   CDNN_ADD_IMAGE   = 0        add one image to every feature maps of each input 
 *   CDNN_ADD_SAME_HW = 0
 *   CDNN_ADD_FEATURE_MAP = 1    add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest
 *   CDNN_ADD_SAME_CHW    = 1
 *   CDNN_ADD_SAME_C      = 2    add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input
 *   CDNN_ADD_FULL_TENSOR = 3    add two tensors with same n,c,h,w 
 */

/**
 * Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc
 * 
 */
cdnnStatus_t CDNNWINAPI cdnnAddTensor(   cdnnHandle_t                    handle,
                                         cdnnAddMode_t                   mode,
                                         const void                      *alpha,
                                         const cdnnTensorDescriptor_t    biasDesc,
                                         const void                      *biasData,
                                         const void                      *beta,
                                         cdnnTensorDescriptor_t          srcDestDesc,
                                         void                            *srcDestData
                                     )
{
    /* Not Implemented */
    switch (mode) {
      case CDNN_ADD_IMAGE:  //      same as CDNN_ADD_SAME_HW: 

      case CDNN_ADD_FEATURE_MAP:  // same as CDNN_ADD_SAME_CHW:

      case CDNN_ADD_SAME_C:

      case CDNN_ADD_FULL_TENSOR:

      default:
         break;
    }
    return CDNN_STATUS_SUCCESS;
}

/* Set all data points of a tensor to a given value : srcDest = value */
cdnnStatus_t CDNNWINAPI cdnnSetTensor(  cdnnHandle_t                   handle,
                                        const cdnnTensorDescriptor_t   srcDestDesc,
                                        void                           *srcDestData,
                                        const void                     *value
                                      )
{
    if ( srcDestDesc->dataType==CDNN_DATA_FLOAT ) {
        float v = *(float*)value;
        float *data = (float*)srcDestData;
        long size = srcDestDesc->size;
        for (long i=0; i<size; ++i) {
            data[i] = v;
        }
    }
    else {
        double v = *(double*)value;
        double *data = (double*)srcDestData;
        long size = srcDestDesc->size;
        for (long i=0; i<size; ++i) {
            data[i] = v;
        }
    }
    return CDNN_STATUS_SUCCESS;
}

/* Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
cdnnStatus_t CDNNWINAPI cdnnScaleTensor(  cdnnHandle_t                    handle,
                                          const cdnnTensorDescriptor_t    srcDestDesc,
                                          void                            *srcDestData,
                                          const void                      *alpha
                                       )
{
    if ( srcDestDesc->dataType==CDNN_DATA_FLOAT ) {
        float a = *(float*)alpha;
        float *data = (float*)srcDestData;
        long size = srcDestDesc->size;
        for (long i=0; i<size; ++i) {
            data[i] = a * data[i];
        }
    }
    else {
        double a = *(double*)alpha;
        double *data = (double*)srcDestData;
        long size = srcDestDesc->size;
        for (long i=0; i<size; ++i) {
            data[i] = a * data[i];
        }
    }
    return CDNN_STATUS_SUCCESS;
}

