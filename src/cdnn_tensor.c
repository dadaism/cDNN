#include <stdio.h>
#include <stdlib.h>

#include "cdnn_core.h"
#include "cdnn.h"

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

    return CDNN_STATUS_SUCCESS;
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

    return CDNN_STATUS_SUCCESS;
}

/**
 * Destroy an instance of Tensor4d descriptor 
 */
cdnnStatus_t CDNNWINAPI cdnnDestroyTensorDescriptor( cdnnTensorDescriptor_t tensorDesc )
{

}


/**
 * Tensor layout conversion helper (dest = alpha * src + beta * dest)
 * Descriptors need to have the same dimemsions but not necessarily the
 * same strides.
 */
cdnnStatus_t CDNNWINAPI cdnnTransformTensor( cdnnHandle_t                         handle,
                                             const void                      *alpha,
                                             const cdnnTensorDescriptor_t    srcDesc,
                                             const void                      *srcData,
                                             const void                      *beta,
                                             const cdnnTensorDescriptor_t    destDesc,
                                             void                            *destData
                                           )
{

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

    return CDNN_STATUS_SUCCESS;
}

/* Set all data points of a tensor to a given value : srcDest = value */
cdnnStatus_t CDNNWINAPI cdnnSetTensor(  cdnnHandle_t                   handle,
                                        const cdnnTensorDescriptor_t   srcDestDesc,
                                        void                           *srcDestData,
                                        const void                     *value
                                      )
{

    return CDNN_STATUS_SUCCESS;
}

/* Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
cdnnStatus_t CDNNWINAPI cdnnScaleTensor(  cdnnHandle_t                    handle,
                                          const cdnnTensorDescriptor_t    srcDestDesc,
                                          void                            *srcDestData,
                                          const void                      *alpha
                                       )
{

    return CDNN_STATUS_SUCCESS;
}

