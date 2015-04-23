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
 * convolution mode
 *   CDNN_CONVOLUTION       = 0
 *   CDNN_CROSS_CORRELATION = 1
 * Different between CONV and CROSS_CORRELATION:
 *   Convolution is correlation with the filter rotated 180 degrees.
 */

/** 
 * Create an instance of FilterStruct 
 * @param *filterDesc filter descriptor
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnCreateFilterDescriptor( cdnnFilterDescriptor_t *filterDesc )
{
    *filterDesc = (cdnnFilterDescriptor_t)malloc(sizeof(struct cdnnFilterStruct));
    if (filterDesc!=NULL)
        return CDNN_STATUS_SUCCESS;
    else
        return CDNN_STATUS_ALLOC_FAILED;
}

/** 
 * Set an instance of 4D Filter
 * @param filterDesc
 * @param dataType 
 * @param k number of output feature maps
 * @param c number of input feature maps
 * @param h height of each input filter
 * @param w width of  each input fitler
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnSetFilter4dDescriptor(  cdnnFilterDescriptor_t filterDesc,
                                                    cdnnDataType_t dataType, // image data type
                                                    int k,        // number of output feature maps
                                                    int c,        // number of input feature maps
                                                    int h,        // height of each input filter
                                                    int w         // width of  each input fitler
                                                  )
{
    filterDesc->dataType = dataType;
    filterDesc->filterDimA = (int *)malloc(sizeof(int)*4);
    if (filterDesc->filterDimA==NULL)
        return CDNN_STATUS_ALLOC_FAILED;
    filterDesc->filterDimA[0] = w;
    filterDesc->filterDimA[1] = h;
    filterDesc->filterDimA[2] = c;
    filterDesc->filterDimA[3] = k;
    return CDNN_STATUS_SUCCESS;
}

/** 
 * Get an instance of 4D Filter
 * @param filterDesc
 * @param *dataType 
 * @param *k number of output feature maps
 * @param *c number of input feature maps
 * @param *h height of each input filter
 * @param *w width of  each input fitler
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnGetFilter4dDescriptor(  const cdnnFilterDescriptor_t filterDesc,
                                                    cdnnDataType_t *dataType, // image data type
                                                    int *k,        // number of output feature maps
                                                    int *c,        // number of input feature maps
                                                    int *h,        // height of each input filter
                                                    int *w         // width of  each input fitler
                                                  )
{

    *dataType = filterDesc->dataType;
    *w = filterDesc->filterDimA[0];
    *h = filterDesc->filterDimA[1];
    *c = filterDesc->filterDimA[2];
    *k = filterDesc->filterDimA[3];
    return CDNN_STATUS_SUCCESS;
}

/** 
 * Set an instance of ND Filter
 * @param filterDesc
 * @param dataType 
 * @param nbDims dimension of the filter
 * @param filterDimA filter dimension array
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnSetFilterNdDescriptor(  cdnnFilterDescriptor_t filterDesc,
                                                    cdnnDataType_t dataType, // image data type
                                                    int nbDims,
                                                    const int filterDimA[]
                                                 )
{
    filterDesc->dataType = dataType;
    filterDesc->filterDimA = (int *)malloc(sizeof(int)*nbDims);
    if (filterDesc->filterDimA==NULL)
        return CDNN_STATUS_ALLOC_FAILED;
    filterDesc->nDimension = nbDims;
    for (int i=0; i<nbDims; ++i) {
        filterDesc->filterDimA[i] = filterDimA[i];
    }
    return CDNN_STATUS_SUCCESS;
}

/** 
 * Get an instance of ND Filter
 * @param filterDesc
 * @param nbDimsRequested Dimension of the expected filter descriptor
 * @param dataType 
 * @param *nbDims dimension of the filter
 * @param filterDimA filter dimension array
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnGetFilterNdDescriptor(  const cdnnFilterDescriptor_t filterDesc,
                                                    int nbDimsRequested,
                                                    cdnnDataType_t *dataType, // image data type
                                                    int *nbDims,
                                                    int filterDimA[]
                                                  )
{
    *dataType = filterDesc->dataType;
    *nbDims = filterDesc->nDimension;
    for (int i=0; i<nbDimsRequested; ++i) {
        filterDimA[i] = filterDesc->filterDimA[i];
    }
    return CDNN_STATUS_SUCCESS;
}

/** 
 * Destroy a filter descriptor
 * @param filterDesc
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnDestroyFilterDescriptor( cdnnFilterDescriptor_t filterDesc )
{
    free(filterDesc->filterDimA);
    free(filterDesc);
    return CDNN_STATUS_SUCCESS;
}

/**
 * Create an instance of convolution descriptor 
 */
cdnnStatus_t CDNNWINAPI cdnnCreateConvolutionDescriptor( cdnnConvolutionDescriptor_t *convDesc )
{
    *convDesc = (cdnnConvolutionDescriptor_t)malloc(sizeof(struct cdnnConvolutionStruct));
    if (convDesc!=NULL)
        return CDNN_STATUS_SUCCESS;
    else
        return CDNN_STATUS_ALLOC_FAILED;
}

/** 
 * Set an instance of 2D convolution descriptor
 * @param convDesc
 * @param pad_h zero-padding height, number of rows of zeros implicitly concatenated
                onto the top and onto the bottom of input images
 * @param pad_w zero-padding width, number of columns of zeros implicitly concatenated
                onto the top and onto the bottom of input images
 * @param u vertical filter stride
 * @param v horizontal filer stride
 * @param upscalex
 * @param upscaley
 * @param mode 
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnSetConvolution2dDescriptor(  cdnnConvolutionDescriptor_t convDesc,
                                                         int pad_h,    // zero-padding height
                                                         int pad_w,    // zero-padding width
                                                         int u,        // vertical filter stride
                                                         int v,        // horizontal filter stride
                                                         int upscalex, // upscale the input in x-direction
                                                         int upscaley, // upscale the input in y-direction
                                                         cdnnConvolutionMode_t mode
                                                      )
{
    convDesc->mode = mode;
    convDesc->arrayLength = 2;
    convDesc->padA = (int *)malloc(sizeof(int)*2);
    convDesc->filterStrideA = (int *)malloc(sizeof(int)*2);
    convDesc->upscaleA = (int *)malloc(sizeof(int)*2);
    if ( convDesc->padA==NULL || convDesc->filterStrideA==NULL || convDesc->upscaleA==NULL)
        return CDNN_STATUS_ALLOC_FAILED;
    convDesc->padA[0] = pad_w;
    convDesc->padA[1] = pad_h; 
    convDesc->filterStrideA[0] = v;
    convDesc->filterStrideA[1] = u;
    convDesc->upscaleA[0] = upscaley;
    convDesc->upscaleA[1] = upscalex;
    return CDNN_STATUS_SUCCESS;
}

/**
 * Get the setting of 2D convolution descriptor
 * @param convDesc
 * @param *pad_h zero-padding height, number of rows of zeros implicitly concatenated
                onto the top and onto the bottom of input images
 * @param *pad_w zero-padding width, number of columns of zeros implicitly concatenated
                onto the top and onto the bottom of input images
 * @param *u vertical filter stride
 * @param *v horizontal filer stride
 * @param *upscalex
 * @param *upscaley
 * @param *mode 
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnGetConvolution2dDescriptor(  const cdnnConvolutionDescriptor_t convDesc,
                                                         int* pad_h,    // zero-padding height
                                                         int* pad_w,    // zero-padding width
                                                         int* u,        // vertical filter stride
                                                         int* v,        // horizontal filter stride
                                                         int* upscalex, // upscale the input in x-direction
                                                         int* upscaley, // upscale the input in y-direction
                                                         cdnnConvolutionMode_t* mode
                                                       )
{
    *mode = convDesc->mode;
    *pad_w = convDesc->padA[0];
    *pad_h = convDesc->padA[1]; 
    *v = convDesc->filterStrideA[0];
    *u = convDesc->filterStrideA[1];
    *upscaley = convDesc->upscaleA[0];
    *upscalex = convDesc->upscaleA[1];

    return CDNN_STATUS_SUCCESS;
}

/** 
 * Helper function to return the dimensions of the output tensor given a convolution descriptor 
 * Equations:
 *
 *
 */
cdnnStatus_t CDNNWINAPI cdnnGetConvolution2dForwardOutputDim( const cdnnConvolutionDescriptor_t convDesc,
                                                              const cdnnTensorDescriptor_t     inputTensorDesc,
                                                              const cdnnFilterDescriptor_t     filterDesc,
                                                              int *n,
                                                              int *c,
                                                              int *h,
                                                              int *w
                                                            )
{
    *n = inputTensorDesc->dimA[3];
    *c = inputTensorDesc->dimA[2];
    /* dim = floor(pad + dataWidth - filterWidth)/stride + 1 */
    *h = (convDesc->padA[1]+inputTensorDesc->dimA[1]-filterDesc->filterDimA[1])/convDesc->filterStrideA[1];
    *w = (convDesc->padA[0]+inputTensorDesc->dimA[0]-filterDesc->filterDimA[0])/convDesc->filterStrideA[0];
    return CDNN_STATUS_SUCCESS;
}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
/**
 * Set an ND convolution descriptor
 * @param convDesc
 * @param arrayLength dimension of the convolution
 * @param padA[] zero-padding size for each dimension. The padding represents the 
                 number of extra zeros implicitly concatenated at the start and
                 at the end of every element of that dimension
                 onto the top and onto the bottom of input images
 * @param filterStrideA
 * @param upscaleA
 * @param *mode 
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnSetConvolutionNdDescriptor( cdnnConvolutionDescriptor_t convDesc,
                                                        int arrayLength,             /* nbDims-2 size */  
                                                        const int padA[],                                          
                                                        const int filterStrideA[],         
                                                        const int upscaleA[],              
                                                        cdnnConvolutionMode_t mode
                                                      )
{
    convDesc->mode = mode;
    convDesc->arrayLength = arrayLength;
    convDesc->padA = (int *)malloc(sizeof(int)*arrayLength);
    convDesc->filterStrideA = (int *)malloc(sizeof(int)*arrayLength);
    convDesc->upscaleA = (int *)malloc(sizeof(int)*arrayLength);
    if ( convDesc->padA==NULL || convDesc->filterStrideA==NULL || convDesc->upscaleA==NULL)
        return CDNN_STATUS_ALLOC_FAILED;
    for (int i=0; i<arrayLength; ++i) {
        convDesc->padA[i]          = padA[i];
        convDesc->filterStrideA[i] = filterStrideA[i];
        convDesc->upscaleA[i]      = upscaleA[i];
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * Get an ND convolution descriptor
 * @param convDesc
 * @param arrayLengthRequested requested dimension
 * @param arrayLength dimension of the convolution
 * @param padA[] zero-padding size for each dimension. The padding represents the 
                 number of extra zeros implicitly concatenated at the start and
                 at the end of every element of that dimension
                 onto the top and onto the bottom of input images
 * @param filterStrideA
 * @param upscaleA
 * @param *mode 
 * @return cdnnStatus_t
 */
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionNdDescriptor( const cdnnConvolutionDescriptor_t convDesc,
                                                        int arrayLengthRequested,
                                                        int *arrayLength,
                                                        int padA[],                                        
                                                        int strideA[],
                                                        int upscaleA[],
                                                        cdnnConvolutionMode_t *mode
                                                      )
{
    *mode = convDesc->mode;
    *arrayLength = convDesc->arrayLength;
    for (int i=0; i<arrayLengthRequested; ++i) {
        padA[i]     = convDesc->padA[i];
        strideA[i]  = convDesc->filterStrideA[i];
        upscaleA[i] = convDesc->upscaleA[i];
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * Helper function to return the dimensions of the output tensor given a convolution descriptor 
 */
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionNdForwardOutputDim( const cdnnConvolutionDescriptor_t convDesc,
                                                              const cdnnTensorDescriptor_t inputTensorDesc,
                                                              const cdnnFilterDescriptor_t filterDesc,
                                                              int nbDims,
                                                              int tensorOutputDimA[]
                                                            )
{
    tensorOutputDimA[nbDims] = inputTensorDesc->dimA[nbDims];
    tensorOutputDimA[nbDims-1] = inputTensorDesc->dimA[nbDims-1];
    /* dim = floor(pad + dataWidth - filterWidth)/stride + 1 */
    for (int i=nbDims-2; i>=0; --i) {
        tensorOutputDimA[i] = (convDesc->padA[i]+inputTensorDesc->dimA[i]-filterDesc->filterDimA[i])/convDesc->filterStrideA[i];
    }
    return CDNN_STATUS_SUCCESS;
}

/**
 * Destroy an instance of convolution descriptor 
 */
cdnnStatus_t CDNNWINAPI cdnnDestroyConvolutionDescriptor( cdnnConvolutionDescriptor_t convDesc )
{
    free(convDesc->padA);
    free(convDesc->filterStrideA);
    free(convDesc->upscaleA);
    return CDNN_STATUS_SUCCESS;
}

/**
 * Select convolution forward algorithm
 * @param preference          enumerant to express the preference criteria in terms of memory requirement and speed
 * @param memoryLiitsInbytes  used when enumerant preference is set to CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMTI
                              to specify the maximum amount of GPU memory the user is willing to use as a workspace
 * @param algo                enumerant that specifies which convolution algorithm should be used to compute 
 */
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionForwardAlgorithm( cdnnHandle_t                       handle,
                                                            const cdnnTensorDescriptor_t      srcDesc,
                                                            const cdnnFilterDescriptor_t      filterDesc,
                                                            const cdnnConvolutionDescriptor_t convDesc, 
                                                            const cdnnTensorDescriptor_t      destDesc,
                                                            cdnnConvolutionFwdPreference_t    preference, 
                                                            size_t                            memoryLimitInbytes,
                                                            cdnnConvolutionFwdAlgo_t          *algo                                                  
                                                          )
{

    return CDNN_STATUS_SUCCESS;
}      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
/**
 * convolution algorithm (which requires potentially some workspace)
 * cdnnConvolutionFwdAlgo_t
 *   CDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0
 *   CDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
 *   CDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2
 *   CDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3
 */

/**
 * Helper function to return the minimum size of the workspace to be passed to the convolution given an algo
 */ 
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionForwardWorkspaceSize( cdnnHandle_t                      handle, 
                                                                const cdnnTensorDescriptor_t      srcDesc,
                                                                const cdnnFilterDescriptor_t      filterDesc,
                                                                const cdnnConvolutionDescriptor_t convDesc,  
                                                                const cdnnTensorDescriptor_t      destDesc,
                                                                cdnnConvolutionFwdAlgo_t          algo,
                                                                size_t                            *sizeInBytes
                                                              )
{

    return CDNN_STATUS_SUCCESS;
}        

/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/**
 * Function to perform the forward multiconvolution. Similar to conv('valid')
 */
cdnnStatus_t CDNNWINAPI cdnnConvolutionForward( cdnnHandle_t                 handle,
                                                const void                         *alpha,
                                                const cdnnTensorDescriptor_t       srcDesc,
                                                const void                         *srcData,
                                                const cdnnFilterDescriptor_t       filterDesc,
                                                const void                         *filterData,
                                                const cdnnConvolutionDescriptor_t  convDesc,
                                                cdnnConvolutionFwdAlgo_t           algo,
                                                void                               *workSpace,
                                                size_t                              workSpaceSizeInBytes,            
                                                const void                         *beta,
                                                const cdnnTensorDescriptor_t       destDesc,
                                                void                               *destData
                                              )
{

    return CDNN_STATUS_SUCCESS;
}

/** 
 * Functions to compute the convolution gradient with respect to the bias, which is the sum of every
 * element belonging to the same feature map across all of the images in this mini-batch (n). The number
 * of produced elements is equal to the number of feature maps of the input tensor (c) 
 * @param *srcData [n c h w] tensor
 * @param *destData [1 c 1 1] tensor
 */
cdnnStatus_t CDNNWINAPI cdnnConvolutionBackwardBias(   cdnnHandle_t                   handle,
                                                       const void                     *alpha,
                                                       const cdnnTensorDescriptor_t   srcDesc,
                                                       const void                     *srcData,
                                                       const void                     *beta,
                                                       const cdnnTensorDescriptor_t   destDesc,
                                                       void                           *destData
                                                    )
{

    return CDNN_STATUS_SUCCESS;
}

/** 
 * Functions to compute the convolution gradient with respect to the filter coefficients. The gradData
 * is the output and has the same dimension as filter tensor.
 * @param 
 * @param 
 */
cdnnStatus_t CDNNWINAPI cdnnConvolutionBackwardFilter( cdnnHandle_t                       handle,
                                                       const void                         *alpha,
                                                       const cdnnTensorDescriptor_t       srcDesc,
                                                       const void                         *srcData,
                                                       const cdnnTensorDescriptor_t       diffDesc,
                                                       const void                         *diffData,
                                                       const cdnnConvolutionDescriptor_t  convDesc,
                                                       const void                         *beta,
                                                       cdnnFilterDescriptor_t       gradDesc,
                                                       void                               *gradData
                                                     )
{

    return CDNN_STATUS_SUCCESS;
}

/** 
 * Functions to compute the convolution gradient with respect to the output tensor. It is passing diffData
 * which is after convolution with small dimension to pre covnolution with large dimension. Similar to
 * conv('full')
 * @param 
 * @param 
 */
cdnnStatus_t CDNNWINAPI cdnnConvolutionBackwardData(  cdnnHandle_t                       handle,
                                                      const void                         *alpha,
                                                      const cdnnFilterDescriptor_t       filterDesc,
                                                      const void                         *filterData,
                                                      const cdnnTensorDescriptor_t       diffDesc,
                                                      const void                         *diffData,
                                                      const cdnnConvolutionDescriptor_t  convDesc,
                                                      const void                         *beta,
                                                      const cdnnTensorDescriptor_t       gradDesc,
                                                      void                               *gradData
                                                   )
{

    return CDNN_STATUS_SUCCESS;
}

/**
 * Rearrange image blocks into columns
 */                                                       
cdnnStatus_t CDNNWINAPI cdnnIm2Col(  cdnnHandle_t                       handle,
                                     const void                         *alpha,
                                     const cdnnTensorDescriptor_t       srcDesc,
                                     const void                         *srcData,
                                     const cdnnFilterDescriptor_t       filterDesc,                                        
                                     const cdnnConvolutionDescriptor_t  convDesc,
                                     void                               *colBuffer
                                  )
{

    return CDNN_STATUS_SUCCESS;
}
