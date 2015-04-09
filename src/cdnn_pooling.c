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

/* Create an instance of pooling descriptor */
cdnnStatus_t CDNNWINAPI cdnnCreatePoolingDescriptor( cdnnPoolingDescriptor_t *poolingDesc)
{


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


}

cdnnStatus_t CDNNWINAPI cdnnSetPoolingNdDescriptor(  cdnnPoolingDescriptor_t poolingDesc,
                                                        const cdnnPoolingMode_t mode,
                                                        int nbDims,
                                                        const int windowDimA[],
                                                        const int paddingA[],
                                                        const int strideA[]
                                                   )
{


}

cdnnStatus_t CDNNWINAPI cdnnGetPoolingNdDescriptor(  const cdnnPoolingDescriptor_t poolingDesc,
                                                        const int nbDimsRequested,
                                                        cdnnPoolingMode_t *mode,
                                                        int *nbDims,
                                                        int windowDimA[],
                                                        int paddingA[],
                                                        int strideA[]
                                                     )
{


}

cdnnStatus_t CDNNWINAPI cdnnGetPoolingNdForwardOutputDim( const cdnnPoolingDescriptor_t poolingDesc,
                                                             const cdnnTensorDescriptor_t inputTensorDesc,
                                                             int nbDims,
                                                             int outputTensorDimA[])
{


}

cdnnStatus_t CDNNWINAPI cdnnGetPooling2dForwardOutputDim( const cdnnPoolingDescriptor_t poolingDesc,
                                                             const cdnnTensorDescriptor_t inputTensorDesc,
                                                             int *outN,
                                                             int *outC,
                                                             int *outH,
                                                             int *outW)
{


}


/* Destroy an instance of pooling descriptor */
cdnnStatus_t CDNNWINAPI cdnnDestroyPoolingDescriptor( cdnnPoolingDescriptor_t poolingDesc )
{


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


}

/* Function to perform backward pooling */
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

	
}
