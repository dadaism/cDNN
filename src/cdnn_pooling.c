#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

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
 * pooling mode
 * cdnnPoolingMode_t:
 *   CDNN_POOLING_MAX     = 0
 *   CDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1   count for average includes padded values
 *   CDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2   count for average does not include padded values
 */

/**
 * Create an instance of pooling descriptor 
 */
cdnnStatus_t CDNNWINAPI cdnnCreatePoolingDescriptor( cdnnPoolingDescriptor_t *poolingDesc)
{
    *poolingDesc = (cdnnPoolingDescriptor_t)malloc(sizeof(struct cdnnPoolingStruct));
    if (poolingDesc!=NULL) 
        return CDNN_STATUS_SUCCESS;
    else 
        return CDNN_STATUS_ALLOC_FAILED;
}

cdnnStatus_t CDNNWINAPI cdnnSetPooling2dDescriptor(  cdnnPoolingDescriptor_t poolingDesc,
                                                        cdnnPoolingMode_t mode,
                                                        int windowHeight,
                                                        int windowWidth,
                                                        int verticalPadding,
                                                        int horizontalPadding,
                                                        int verticalStride,
                                                        int horizontalStride
                                                   )
{
    poolingDesc->nDimension  = 2;
    poolingDesc->windowDimA  = (int *)malloc(sizeof(int)*2);
    poolingDesc->paddingA    = (int *)malloc(sizeof(int)*2);
    poolingDesc->strideA     = (int *)malloc(sizeof(int)*2);
    if ( poolingDesc->windowDimA==NULL 
         || poolingDesc->paddingA==NULL 
         || poolingDesc->strideA==NULL )
        return CDNN_STATUS_ALLOC_FAILED;
    poolingDesc->windowDimA[1]  = windowHeight;
    poolingDesc->windowDimA[0]  = windowWidth;
    poolingDesc->paddingA[1]    = verticalPadding;
    poolingDesc->paddingA[0]    = horizontalPadding;
    poolingDesc->strideA[1]     = verticalStride;
    poolingDesc->strideA[0]     = horizontalStride;
    return CDNN_STATUS_SUCCESS;
}

cdnnStatus_t CDNNWINAPI cdnnGetPooling2dDescriptor(  const cdnnPoolingDescriptor_t poolingDesc,
                                                     cdnnPoolingMode_t *mode,
                                                     int *windowHeight,
                                                     int *windowWidth,
                                                     int *verticalPadding,
                                                     int *horizontalPadding,
                                                     int *verticalStride,
                                                     int *horizontalStride
                                                   )
{
    *windowHeight      = poolingDesc->windowDimA[1];
    *windowWidth       = poolingDesc->windowDimA[0];
    *verticalPadding   = poolingDesc->paddingA[1];
    *horizontalPadding = poolingDesc->paddingA[0];
    *verticalStride    = poolingDesc->strideA[1];
    *horizontalStride  = poolingDesc->strideA[0];
    return CDNN_STATUS_SUCCESS;
}

/**
 * Initialize a previously created generic pooling descriptor object.
 * @param poolingDesc  pooling descriptor
 * @param mode         enumerant to specify the pooling mode
 * @param nbDims       dimension of the pooling operation
 * @param windowDimA   array of dimension nbDims containing the window size for each dimension
 * @param paddingA     array of dimension nbDims containing the padding size for each dimension
 * @param strideA      array of dimension nbDims containing the striding size for each dimension
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnSetPoolingNdDescriptor(  cdnnPoolingDescriptor_t poolingDesc,
                                                     const cdnnPoolingMode_t mode,
                                                     int nbDims,
                                                     const int windowDimA[],
                                                     const int paddingA[],
                                                     const int strideA[]
                                                   )
{
    poolingDesc->nDimension = nbDims;
    poolingDesc->windowDimA  = (int *)malloc(sizeof(int)*nbDims);
    poolingDesc->paddingA    = (int *)malloc(sizeof(int)*nbDims);
    poolingDesc->strideA     = (int *)malloc(sizeof(int)*nbDims);
    if ( poolingDesc->windowDimA==NULL 
         || poolingDesc->paddingA==NULL 
         || poolingDesc->strideA==NULL )
        return CDNN_STATUS_ALLOC_FAILED;

    for (int i=0; i<nbDims; ++i) {
        poolingDesc->windowDimA[i] = windowDimA[i];
        poolingDesc->paddingA[i] = paddingA[i];
        poolingDesc->strideA[i] = strideA[i];   
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * Query a previously created generic pooling descriptor object.
 */
cdnnStatus_t CDNNWINAPI cdnnGetPoolingNdDescriptor(  const cdnnPoolingDescriptor_t poolingDesc,
                                                     const int nbDimsRequested,
                                                     cdnnPoolingMode_t *mode,
                                                     int *nbDims,
                                                     int windowDimA[],
                                                     int paddingA[],
                                                     int strideA[]
                                                  )
{
    *nbDims = poolingDesc->nDimension;
    for (int i=0; i<*nbDims; ++i) {
        windowDimA[i] = poolingDesc->windowDimA[i];
        paddingA[i]   = poolingDesc->paddingA[i];
        strideA[i]    = poolingDesc->strideA[i]; 
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * Helper function to return the dimensions of the output tensor given a pooling descriptor 
 */
cdnnStatus_t CDNNWINAPI cdnnGetPoolingNdForwardOutputDim( const cdnnPoolingDescriptor_t poolingDesc,
                                                          const cdnnTensorDescriptor_t inputTensorDesc,
                                                          int nbDims,
                                                          int outputTensorDimA[]
                                                        )
{
    for (int i=0; i<nbDims; ++i) {
        outputTensorDimA[i] = (inputTensorDesc->dimA[i]+poolingDesc->paddingA[i]-poolingDesc->windowDimA[i])/poolingDesc->strideA[i] + 1;
    }
    for (int i=nbDims; i<inputTensorDesc->nDimension; ++i) {
        outputTensorDimA[i] = (inputTensorDesc->dimA[i]+poolingDesc->paddingA[i]-poolingDesc->windowDimA[i])/poolingDesc->strideA[i] + 1;
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * Helper function to return the dimensions of the output tensor given a pooling descriptor 
 */
cdnnStatus_t CDNNWINAPI cdnnGetPooling2dForwardOutputDim( const cdnnPoolingDescriptor_t poolingDesc,
                                                          const cdnnTensorDescriptor_t inputTensorDesc,
                                                          int *outN,
                                                          int *outC,
                                                          int *outH,
                                                          int *outW)
{
    *outW = (inputTensorDesc->dimA[0]+poolingDesc->paddingA[0]-poolingDesc->windowDimA[0])/poolingDesc->strideA[0] + 1; // <- do I need a floor function here ?
    *outH = (inputTensorDesc->dimA[1]+poolingDesc->paddingA[1]-poolingDesc->windowDimA[1])/poolingDesc->strideA[1] + 1; 
    *outC = inputTensorDesc->dimA[2];
    *outN = inputTensorDesc->dimA[3];
    return CDNN_STATUS_SUCCESS;
}


/* Destroy an instance of pooling descriptor */
cdnnStatus_t CDNNWINAPI cdnnDestroyPoolingDescriptor( cdnnPoolingDescriptor_t poolingDesc )
{
    free(poolingDesc->windowDimA);
    free(poolingDesc->paddingA);
    free(poolingDesc->strideA);
    free(poolingDesc);
    return CDNN_STATUS_SUCCESS;
}

/* Pooling functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward pooling */
cdnnStatus_t CDNNWINAPI cdnnPoolingForward(  cdnnHandle_t handle,
                                             const cdnnPoolingDescriptor_t   poolingDesc,
                                             const void                      *alpha,
                                             const cdnnTensorDescriptor_t    srcDesc,
                                             const void                      *srcData,
                                             const void                      *beta,
                                             const cdnnTensorDescriptor_t    destDesc,
                                             void                            *destData
                                          )
{
    /* Only support 2D pooling */
    assert(poolingDesc->nDimension==2);



    return CDNN_STATUS_SUCCESS;
}

/**
 * Function to perform backward pooling 
 */
cdnnStatus_t CDNNWINAPI cdnnPoolingBackward( cdnnHandle_t                   handle,
                                             const cdnnPoolingDescriptor_t  poolingDesc,
                                             const void                      *alpha,
                                             const cdnnTensorDescriptor_t   srcDesc,
                                             const void                     *srcData,
                                             const cdnnTensorDescriptor_t   srcDiffDesc,
                                             const void                     *srcDiffData,
                                             const cdnnTensorDescriptor_t   destDesc,
                                             const void                     *destData,
                                             const void                     *beta,
                                             const cdnnTensorDescriptor_t   destDiffDesc,
                                             void                           *destDiffData
                                           )
{

    return CDNN_STATUS_SUCCESS;	
}
