/* 
 * Copyright (c) 2015 Da Li

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 * cdnn: Neural Networks Library on CPU
 *
 */

#ifndef __CDNN_H__
#define __CDNN_H__

#ifndef CDNNWINAPI
#ifdef _WIN32
#define CDNNWINAPI __stdcall
#else
#define CDNNWINAPI
#endif
#endif

#if defined (__cplusplus)
extern "C" {
#endif

struct cdnnContext;
typedef struct cdnnContext *cdnnHandle_t;

/*
 * CDNN return codes
 */
typedef enum
{
    CDNN_STATUS_SUCCESS          = 0,
    CDNN_STATUS_NOT_INITIALIZED  = 1,
    CDNN_STATUS_ALLOC_FAILED     = 2,
    CDNN_STATUS_BAD_PARAM        = 3,
    CDNN_STATUS_INTERNAL_ERROR   = 4,
    CDNN_STATUS_INVALID_VALUE    = 5,
    CDNN_STATUS_ARCH_MISMATCH    = 6,
    CDNN_STATUS_MAPPING_ERROR    = 7,
    CDNN_STATUS_EXECUTION_FAILED = 8,
    CDNN_STATUS_NOT_SUPPORTED    = 9,
    CDNN_STATUS_LICENSE_ERROR    = 10
} cdnnStatus_t;

// human-readable error messages
const char * CDNNWINAPI cdnnGetErrorString(cdnnStatus_t status);

cdnnStatus_t CDNNWINAPI cdnnCreate(cdnnHandle_t *handle);
cdnnStatus_t CDNNWINAPI cdnnDestroy(cdnnHandle_t handle);


/* Data structures to represent Image/Filter and the Neural Network Layer */
typedef struct cdnnTensorStruct*        cdnnTensorDescriptor_t;
typedef struct cdnnConvolutionStruct*   cdnnConvolutionDescriptor_t;
typedef struct cdnnPoolingStruct*       cdnnPoolingDescriptor_t;
typedef struct cdnnFilterStruct*        cdnnFilterDescriptor_t;

/*
 * CDNN data type
 */
typedef enum
{
    CDNN_DATA_FLOAT  = 0,
    CDNN_DATA_DOUBLE = 1
} cdnnDataType_t;

/* Create an instance of a generic Tensor descriptor */
cdnnStatus_t cdnnCreateTensorDescriptor( cdnnTensorDescriptor_t   *tensorDesc );

typedef enum
{
    CDNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
    CDNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
} cdnnTensorFormat_t;

cdnnStatus_t CDNNWINAPI cdnnSetTensor4dDescriptor(   cdnnTensorDescriptor_t   tensorDesc,
                                                       cdnnTensorFormat_t  format,
                                                       cdnnDataType_t dataType, // image data type
                                                       int n,        // number of inputs (batch size)
                                                       int c,        // number of input feature maps
                                                       int h,        // height of input section
                                                       int w         // width of input section
                                                    );

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
                                                      );

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
                                                    );

cdnnStatus_t CDNNWINAPI cdnnSetTensorNdDescriptor(  cdnnTensorDescriptor_t tensorDesc,
                                                       cdnnDataType_t dataType,
                                                       int nbDims,
                                                       const int dimA[],
                                                       const int strideA[]
                                                    );
//
cdnnStatus_t CDNNWINAPI cdnnGetTensorNdDescriptor(  const cdnnTensorDescriptor_t tensorDesc,
                                                        int nbDimsRequested,
                                                        cdnnDataType_t *dataType,
                                                        int *nbDims,
                                                        int dimA[],
                                                        int strideA[]
                                                    );

/* PixelOffset( n, c, h, w ) = n *input_stride + c * feature_stride + h * h_stride + w * w_stride

   1)Example of all images in row major order one batch of features after the other (with an optional padding on row)
   input_stride :  c x h x h_stride
   feature_stride : h x h_stride
   h_stride  :  >= w  ( h_stride = w if no padding)
   w_stride  : 1

   2)Example of all images in row major with features maps interleaved
   input_stride :  c x h x h_stride
   feature_stride : 1
   h_stride  :  w x c
   w_stride  : c

   3)Example of all images in column major order one batch of features after the other (with optional padding on column)
   input_stride :  c x w x w_stride
   feature_stride : w x w_stride
   h_stride  :  1
   w_stride  :  >= h

*/


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
                                              );

typedef enum
{
    CDNN_ADD_IMAGE   = 0,       /* add one image to every feature maps of each input */
    CDNN_ADD_SAME_HW = 0,

    CDNN_ADD_FEATURE_MAP = 1,   /* add a set of feature maps to a batch of inputs : tensorBias has n=1 , same nb feature than Src/dest */
    CDNN_ADD_SAME_CHW    = 1,

    CDNN_ADD_SAME_C      = 2,   /* add a tensor of size 1,c,1,1 to every corresponding point of n,c,h,w input */

    CDNN_ADD_FULL_TENSOR = 3    /* add 2 tensors with same n,c,h,w */
} cdnnAddMode_t;

/* Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc  */
cdnnStatus_t CDNNWINAPI cdnnAddTensor(  cdnnHandle_t                    handle,
                                        cdnnAddMode_t                   mode,
                                        const void                      *alpha,
                                        const cdnnTensorDescriptor_t    biasDesc,
                                        const void                      *biasData,
                                        const void                      *beta,
                                        cdnnTensorDescriptor_t          srcDestDesc,
                                        void                            *srcDestData
                                     );

/* Set all data points of a tensor to a given value : srcDest = value */
cdnnStatus_t CDNNWINAPI cdnnSetTensor(  cdnnHandle_t                   handle,
                                        const cdnnTensorDescriptor_t   srcDestDesc,
                                        void                           *srcDestData,
                                        const void                     *value
                                     ) ;

/* Set all data points of a tensor to a given value : srcDest = alpha * srcDest */
cdnnStatus_t CDNNWINAPI cdnnScaleTensor(  cdnnHandle_t                    handle,
                                          const cdnnTensorDescriptor_t    srcDestDesc,
                                          void                            *srcDestData,
                                          const void                      *alpha
                                       ) ;

/*
*  convolution mode
*/
typedef enum
{
    CDNN_CONVOLUTION       = 0,
    CDNN_CROSS_CORRELATION = 1
} cdnnConvolutionMode_t;


/* Create an instance of FilterStruct */
cdnnStatus_t CDNNWINAPI cdnnCreateFilterDescriptor( cdnnFilterDescriptor_t *filterDesc );

cdnnStatus_t CDNNWINAPI cdnnSetFilter4dDescriptor(  cdnnFilterDescriptor_t filterDesc,
                                                      cdnnDataType_t dataType, // image data type
                                                      int k,        // number of output feature maps
                                                      int c,        // number of input feature maps
                                                      int h,        // height of each input filter
                                                      int w         // width of  each input fitler
                                                    );

cdnnStatus_t CDNNWINAPI cdnnGetFilter4dDescriptor(  const cdnnFilterDescriptor_t filterDesc,
                                                      cdnnDataType_t *dataType, // image data type
                                                      int *k,        // number of output feature maps
                                                      int *c,        // number of input feature maps
                                                      int *h,        // height of each input filter
                                                      int *w         // width of  each input fitler
                                                    );

cdnnStatus_t CDNNWINAPI cdnnSetFilterNdDescriptor(  cdnnFilterDescriptor_t filterDesc,
                                                    cdnnDataType_t dataType, // image data type
                                                    int nbDims,
                                                    const int filterDimA[]
                                                 );

cdnnStatus_t CDNNWINAPI cdnnGetFilterNdDescriptor(  const cdnnFilterDescriptor_t filterDesc,
                                                    int nbDimsRequested,
                                                    cdnnDataType_t *dataType, // image data type
                                                    int *nbDims,
                                                    int filterDimA[]
                                                );

cdnnStatus_t CDNNWINAPI cdnnDestroyFilterDescriptor( cdnnFilterDescriptor_t filterDesc );

/* Create an instance of convolution descriptor */
cdnnStatus_t CDNNWINAPI cdnnCreateConvolutionDescriptor( cdnnConvolutionDescriptor_t *convDesc );


cdnnStatus_t CDNNWINAPI cdnnSetConvolution2dDescriptor(  cdnnConvolutionDescriptor_t convDesc,
                                                         int pad_h,    // zero-padding height
                                                         int pad_w,    // zero-padding width
                                                         int u,        // vertical filter stride
                                                         int v,        // horizontal filter stride
                                                         int upscalex, // upscale the input in x-direction
                                                         int upscaley, // upscale the input in y-direction
                                                         cdnnConvolutionMode_t mode
                                                      );


cdnnStatus_t CDNNWINAPI cdnnGetConvolution2dDescriptor(   const cdnnConvolutionDescriptor_t convDesc,
                                                          int* pad_h,    // zero-padding height
                                                          int* pad_w,    // zero-padding width
                                                          int* u,        // vertical filter stride
                                                          int* v,        // horizontal filter stride
                                                          int* upscalex, // upscale the input in x-direction
                                                          int* upscaley, // upscale the input in y-direction
                                                          cdnnConvolutionMode_t* mode
                                                      );

/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cdnnStatus_t CDNNWINAPI cdnnGetConvolution2dForwardOutputDim( const cdnnConvolutionDescriptor_t convDesc,
                                                              const cdnnTensorDescriptor_t     inputTensorDesc,
                                                                const cdnnFilterDescriptor_t     filterDesc,
                                                                int *n,
                                                                int *c,
                                                                int *h,
                                                                int *w
                                                              );
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                

cdnnStatus_t CDNNWINAPI cdnnSetConvolutionNdDescriptor( cdnnConvolutionDescriptor_t convDesc,
                                                        int arrayLength,             /* nbDims-2 size */  
                                                        const int padA[],                                          
                                                        const int filterStrideA[],         
                                                        const int upscaleA[],              
                                                        cdnnConvolutionMode_t mode
                                                      );

cdnnStatus_t CDNNWINAPI cdnnGetConvolutionNdDescriptor( const cdnnConvolutionDescriptor_t convDesc,
                                                        int arrayLengthRequested,
                                                        int *arrayLength,
                                                        int padA[],                                        
                                                        int strideA[],
                                                        int upscaleA[],
                                                        cdnnConvolutionMode_t *mode
                                                      );


/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionNdForwardOutputDim( const cdnnConvolutionDescriptor_t convDesc,
                                                              const cdnnTensorDescriptor_t inputTensorDesc,
                                                              const cdnnFilterDescriptor_t filterDesc,
                                                              int nbDims,
                                                              int tensorOuputDimA[]
                                                            );

/* Destroy an instance of convolution descriptor */
cdnnStatus_t CDNNWINAPI cdnnDestroyConvolutionDescriptor( cdnnConvolutionDescriptor_t convDesc );


/* helper function to provide the convolution algo that fit best the requirement */
typedef enum
{
    CDNN_CONVOLUTION_FWD_NO_WORKSPACE        = 0,
    CDNN_CONVOLUTION_FWD_PREFER_FASTEST      = 1,
    CDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cdnnConvolutionFwdPreference_t;  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
typedef enum
{
    CDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3    
} cdnnConvolutionFwdAlgo_t;

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionForwardAlgorithm( cdnnHandle_t                      handle,
                                                            const cdnnTensorDescriptor_t      srcDesc,
                                                            const cdnnFilterDescriptor_t      filterDesc,
                                                            const cdnnConvolutionDescriptor_t convDesc, 
                                                            const cdnnTensorDescriptor_t      destDesc,
                                                            cdnnConvolutionFwdPreference_t    preference, 
                                                            size_t                             memoryLimitInbytes,
                                                            cdnnConvolutionFwdAlgo_t         *algo                                                  
                                                          );        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
/*
*  convolution algorithm (which requires potentially some workspace)
*/

/* Helper function to return the minimum size of the workspace to be passed to the convolution given an algo*/ 
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionForwardWorkspaceSize( cdnnHandle_t                      handle, 
                                                                const cdnnTensorDescriptor_t      srcDesc,
                                                                const cdnnFilterDescriptor_t      filterDesc,
                                                                const cdnnConvolutionDescriptor_t convDesc,  
                                                                const cdnnTensorDescriptor_t      destDesc,
                                                                cdnnConvolutionFwdAlgo_t          algo,
                                                                size_t                            *sizeInBytes
                                                              );        


/* Convolution functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform the forward multiconvolution */
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
                                              );

/* Functions to perform the backward multiconvolution */
cdnnStatus_t CDNNWINAPI cdnnConvolutionBackwardBias( cdnnHandle_t                   handle,
                                                     const void                     *alpha,
                                                     const cdnnTensorDescriptor_t   srcDesc,
                                                     const void                      *srcData,
                                                     const void                      *beta,
                                                     const cdnnTensorDescriptor_t   destDesc,
                                                     void                           *destData
                                                   );

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
                                                      );

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
                                                    );
                                                       
cdnnStatus_t CDNNWINAPI cdnnIm2Col(  cdnnHandle_t                       handle,
                                     const void                         *alpha,
                                     const cdnnTensorDescriptor_t       srcDesc,
                                     const void                         *srcData,
                                     const cdnnFilterDescriptor_t       filterDesc,                                        
                                     const cdnnConvolutionDescriptor_t  convDesc,
                                     void                               *colBuffer
                                  );


/*
 *  softmax algorithm
 */
typedef enum
{
    CDNN_SOFTMAX_FAST     = 0,        /* straightforward implementation */
    CDNN_SOFTMAX_ACCURATE = 1         /* subtract max from every point to avoid overflow */
} cdnnSoftmaxAlgorithm_t;

typedef enum
{
    CDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    CDNN_SOFTMAX_MODE_CHANNEL = 1     /* compute the softmax over all C for each H, W, N */
} cdnnSoftmaxMode_t;

/* Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward softmax */
cdnnStatus_t CDNNWINAPI cdnnSoftmaxForward(  cdnnHandle_t                    handle,
                                             cdnnSoftmaxAlgorithm_t          algorithm,
                                             cdnnSoftmaxMode_t               mode,
                                             const void                      *alpha,
                                             const cdnnTensorDescriptor_t    srcDesc,
                                             const void                      *srcData,
                                             const void                      *beta,
                                             const cdnnTensorDescriptor_t    destDesc,
                                             void                            *destData
                                          );

/* Function to perform backward softmax */
cdnnStatus_t CDNNWINAPI cdnnSoftmaxBackward( cdnnHandle_t                    handle,
                                             cdnnSoftmaxAlgorithm_t          algorithm,
                                             cdnnSoftmaxMode_t               mode,
                                             const void                      *alpha,
                                             const cdnnTensorDescriptor_t    srcDesc,
                                             const void                      *srcData,
                                             const cdnnTensorDescriptor_t    srcDiffDesc,
                                             const void                      *srcDiffData,
                                             const void                      *beta,
                                             const cdnnTensorDescriptor_t    destDiffDesc,
                                             void                            *destDiffData
                                           );

/*
 *  pooling mode
 */
typedef enum
{
    CDNN_POOLING_MAX     = 0,
    CDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, // count for average includes padded values
    CDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2 // count for average does not include padded values
} cdnnPoolingMode_t;

/* Create an instance of pooling descriptor */
cdnnStatus_t CDNNWINAPI cdnnCreatePoolingDescriptor( cdnnPoolingDescriptor_t *poolingDesc);

cdnnStatus_t CDNNWINAPI cdnnSetPooling2dDescriptor(  cdnnPoolingDescriptor_t poolingDesc,
                                                     cdnnPoolingMode_t mode,
                                                     int windowHeight,
                                                     int windowWidth,
                                                     int verticalPadding,
                                                     int horizontalPadding,
                                                     int verticalStride,
                                                     int horizontalStride
                                                   );

cdnnStatus_t CDNNWINAPI cdnnGetPooling2dDescriptor(  const cdnnPoolingDescriptor_t poolingDesc,
                                                     cdnnPoolingMode_t *mode,
                                                     int *windowHeight,
                                                     int *windowWidth,
                                                     int *verticalPadding,
                                                     int *horizontalPadding,
                                                     int *verticalStride,
                                                     int *horizontalStride
                                                   );

cdnnStatus_t CDNNWINAPI cdnnSetPoolingNdDescriptor(  cdnnPoolingDescriptor_t poolingDesc,
                                                     const cdnnPoolingMode_t mode,
                                                     int nbDims,
                                                     const int windowDimA[],
                                                     const int paddingA[],
                                                     const int strideA[]
                                                   );

cdnnStatus_t CDNNWINAPI cdnnGetPoolingNdDescriptor(  const cdnnPoolingDescriptor_t poolingDesc,
                                                     const int nbDimsRequested,
                                                     cdnnPoolingMode_t *mode,
                                                     int *nbDims,
                                                     int windowDimA[],
                                                     int paddingA[],
                                                     int strideA[]
                                                  );

cdnnStatus_t CDNNWINAPI cdnnGetPoolingNdForwardOutputDim( const cdnnPoolingDescriptor_t poolingDesc,
                                                          const cdnnTensorDescriptor_t inputTensorDesc,
                                                          int nbDims,
                                                          int outputTensorDimA[]);

cdnnStatus_t CDNNWINAPI cdnnGetPooling2dForwardOutputDim( const cdnnPoolingDescriptor_t poolingDesc,
                                                          const cdnnTensorDescriptor_t inputTensorDesc,
                                                          int *outN,
                                                          int *outC,
                                                          int *outH,
                                                          int *outW);


/* Destroy an instance of pooling descriptor */
cdnnStatus_t CDNNWINAPI cdnnDestroyPoolingDescriptor( cdnnPoolingDescriptor_t poolingDesc );

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
                                          );

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
                                           );

/*
 * activation mode
 */
typedef enum
{
    CDNN_ACTIVATION_SIGMOID = 0,
    CDNN_ACTIVATION_RELU    = 1,
    CDNN_ACTIVATION_TANH    = 2
} cdnnActivationMode_t;

/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/* Function to perform forward activation  */
cdnnStatus_t CDNNWINAPI cdnnActivationForward( cdnnHandle_t                    handle,
                                               cdnnActivationMode_t            mode,
                                               const void                      *alpha,
                                               const cdnnTensorDescriptor_t    srcDesc,
                                               const void                      *srcData,
                                               const void                      *beta,
                                               const cdnnTensorDescriptor_t    destDesc,
                                               void                            *destData
                                             );

/* Function to perform backward activation  */
cdnnStatus_t CDNNWINAPI cdnnActivationBackward( cdnnHandle_t                    handle,
                                                cdnnActivationMode_t            mode,
                                                const void                      *alpha,
                                                const cdnnTensorDescriptor_t    srcDesc,
                                                const void                      *srcData,
                                                const cdnnTensorDescriptor_t    srcDiffDesc,
                                                const void                      *srcDiffData,
                                                const cdnnTensorDescriptor_t    destDesc,
                                                const void                      *destData,
                                                const void                      *beta,
                                                const cdnnTensorDescriptor_t    destDiffDesc,
                                                void                            *destDiffData
                                              );
#if defined (__cplusplus)
}
#endif

#endif
