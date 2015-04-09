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

/* Create an instance of a generic Tensor descriptor */
cdnnStatus_t cdnnCreateTensorDescriptor( cdnnTensorDescriptor_t   *tensorDesc )
{


}

cdnnStatus_t CDNNWINAPI cdnnSetTensor4dDescriptor(   cdnnTensorDescriptor_t   tensorDesc,
                                                       cdnnTensorFormat_t  format,
                                                       cdnnDataType_t dataType, // image data type
                                                       int n,        // number of inputs (batch size)
                                                       int c,        // number of input feature maps
                                                       int h,        // height of input section
                                                       int w         // width of input section
                                                  )
{



}

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



}

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



}

cdnnStatus_t CDNNWINAPI cdnnSetTensorNdDescriptor(  cdnnTensorDescriptor_t tensorDesc,
                                                       cdnnDataType_t dataType,
                                                       int nbDims,
                                                       const int dimA[],
                                                       const int strideA[]
                                                    )
{


}

cdnnStatus_t CDNNWINAPI cdnnGetTensorNdDescriptor(  const cdnnTensorDescriptor_t tensorDesc,
                                                        int nbDimsRequested,
                                                        cdnnDataType_t *dataType,
                                                        int *nbDims,
                                                        int dimA[],
                                                        int strideA[]
                                                    )
{


}

/* Destroy an instance of Tensor4d descriptor */
cdnnStatus_t CDNNWINAPI cdnnDestroyTensorDescriptor( cdnnTensorDescriptor_t tensorDesc );


/* Tensor layout conversion helper (dest = alpha * src + beta * dest) */
cdnnStatus_t CDNNWINAPI cdnnTransformTensor( cdnnHandle_t                         handle,
                                                const void                      *alpha,
                                                const cdnnTensorDescriptor_t    srcDesc,
                                                const void                      *srcData,
                                                const void                      *beta,
                                                const cdnnTensorDescriptor_t    destDesc,
                                                void                            *destData
                                              )
{


}


/* Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
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


}

/* Set all data points of a tensor to a given value : srcDest = value */
cdnnStatus_t CDNNWINAPI cdnnSetTensor(  cdnnHandle_t                   handle,
                                        const cdnnTensorDescriptor_t   srcDestDesc,
                                        void                           *srcDestData,
                                        const void                     *value
                                      )
{


}

/* Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
cdnnStatus_t CDNNWINAPI cdnnScaleTensor(  cdnnHandle_t                    handle,
                                          const cdnnTensorDescriptor_t    srcDestDesc,
                                          void                            *srcDestData,
                                          const void                      *alpha
                                       )
{


}

