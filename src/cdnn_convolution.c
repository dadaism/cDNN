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


/* Create an instance of FilterStruct */
cdnnStatus_t CDNNWINAPI cdnnCreateFilterDescriptor( cdnnFilterDescriptor_t *filterDesc )
{


}

cdnnStatus_t CDNNWINAPI cdnnSetFilter4dDescriptor(  cdnnFilterDescriptor_t filterDesc,
                                                    cdnnDataType_t dataType, // image data type
                                                    int k,        // number of output feature maps
                                                    int c,        // number of input feature maps
                                                    int h,        // height of each input filter
                                                    int w         // width of  each input fitler
                                                  )
{


}

cdnnStatus_t CDNNWINAPI cdnnGetFilter4dDescriptor(  const cdnnFilterDescriptor_t filterDesc,
                                                    cdnnDataType_t *dataType, // image data type
                                                    int *k,        // number of output feature maps
                                                    int *c,        // number of input feature maps
                                                    int *h,        // height of each input filter
                                                    int *w         // width of  each input fitler
                                                  )
{



}

cdnnStatus_t CDNNWINAPI cdnnSetFilterNdDescriptor(  cdnnFilterDescriptor_t filterDesc,
                                                      cdnnDataType_t dataType, // image data type
                                                       int nbDims,
                                                     const int filterDimA[]
)
{


}

cdnnStatus_t CDNNWINAPI cdnnGetFilterNdDescriptor(  const cdnnFilterDescriptor_t filterDesc,
                                                     int nbDimsRequested,
                                                     cdnnDataType_t *dataType, // image data type
                                                     int *nbDims,
                                                     int filterDimA[]
                                                  )
{

}

cdnnStatus_t CDNNWINAPI cdnnDestroyFilterDescriptor( cdnnFilterDescriptor_t filterDesc )
{

}

/* Create an instance of convolution descriptor */
 cdnnStatus_t CDNNWINAPI cdnnCreateConvolutionDescriptor( cdnnConvolutionDescriptor_t *convDesc )
 {


 }

cdnnStatus_t CDNNWINAPI cdnnSetConvolution2dDescriptor(  cdnnConvolutionDescriptor_t convDesc,
                                                         int pad_h,    // zero-padding height
                                                         int pad_w,    // zero-padding width
                                                         int u,        // vertical filter stride
                                                         int v,        // horizontal filter stride
                                                         int upscalex, // upscale the input in x-direction
                                                         int upscaley, // upscale the input in y-direction
                                                         cdnnConvolutionMode_t mode
                                                      );


cdnnStatus_t CDNNWINAPI cdnnGetConvolution2dDescriptor(  const cdnnConvolutionDescriptor_t convDesc,
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
                                    )
{


}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

cdnnStatus_t CDNNWINAPI cdnnSetConvolutionNdDescriptor( cdnnConvolutionDescriptor_t convDesc,
                                                        int arrayLength,             /* nbDims-2 size */  
                                                        const int padA[],                                          
                                                        const int filterStrideA[],         
                                                        const int upscaleA[],              
                                                        cdnnConvolutionMode_t mode
                                                      )
{


}

cdnnStatus_t CDNNWINAPI cdnnGetConvolutionNdDescriptor( const cdnnConvolutionDescriptor_t convDesc,
                                                        int arrayLengthRequested,
                                                        int *arrayLength,
                                                        int padA[],                                        
                                                        int strideA[],
                                                        int upscaleA[],
                                                        cdnnConvolutionMode_t *mode
                                                      )
{


}


/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionNdForwardOutputDim( const cdnnConvolutionDescriptor_t convDesc,
                                                              const cdnnTensorDescriptor_t inputTensorDesc,
                                                              const cdnnFilterDescriptor_t filterDesc,
                                                              int nbDims,
                                                              int tensorOuputDimA[]
                                                            )
{

}

/* Destroy an instance of convolution descriptor */
cdnnStatus_t CDNNWINAPI cdnnDestroyConvolutionDescriptor( cdnnConvolutionDescriptor_t convDesc )
{


}

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
cdnnStatus_t CDNNWINAPI cdnnGetConvolutionForwardAlgorithm( cdnnHandle_t                      handle,
                                                            const cdnnTensorDescriptor_t      srcDesc,
                                                            const cdnnFilterDescriptor_t      filterDesc,
                                                            const cdnnConvolutionDescriptor_t convDesc, 
                                                            const cdnnTensorDescriptor_t      destDesc,
                                                            cdnnConvolutionFwdPreference_t    preference, 
                                                            size_t                             memoryLimitInbytes,
                                                            cdnnConvolutionFwdAlgo_t         *algo                                                  
                                                          )
{


}      
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
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
                                                               )
{


}        

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
                                              )
{


}

/* Functions to perform the backward multiconvolution */
cdnnStatus_t CDNNWINAPI cdnnConvolutionBackwardBias(   cdnnHandle_t                   handle,
                                                       const void                     *alpha,
                                                       const cdnnTensorDescriptor_t   srcDesc,
                                                       const void                      *srcData,
                                                       const void                      *beta,
                                                       const cdnnTensorDescriptor_t   destDesc,
                                                       void                           *destData
                                                    )
{


}

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


}

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


}
                                                       
cdnnStatus_t CDNNWINAPI cdnnIm2Col(  cdnnHandle_t                       handle,
                                     const void                         *alpha,
                                     const cdnnTensorDescriptor_t       srcDesc,
                                     const void                         *srcData,
                                     const cdnnFilterDescriptor_t       filterDesc,                                        
                                     const cdnnConvolutionDescriptor_t  convDesc,
                                     void                               *colBuffer
                                  )
{


}
