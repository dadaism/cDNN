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
 * Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" 
 * cdnnSoftmaxAlgorithm_t: 
 *     CDNN_SOFTMAX_MODE_INSTANCE = 0    compute the softmax over all C, H, W for each N 
 *     CDNN_SOFTMAX_MODE_CHANNEL = 1     compute the softmax over all C for each H, W, N 
 *
 *
 *
 */

/** 
 * Function to perform forward softmax. output = alpha * Softmax(inputs) + beta * output
 * @param handle 
 * @param algorithm
 * @param mode
 * @param *alpha
 * @param srcDesc
 * @param *srcData 
 * @param *beta
 * @param destDesc
 * @param *destData 
 * @see cdnnSoftmaxBackward()
 * @return cdnnStatus_t
 */ 
cdnnStatus_t CDNNWINAPI cdnnSoftmaxForward(  cdnnHandle_t                    handle,
                                             cdnnSoftmaxAlgorithm_t          algorithm,
                                             cdnnSoftmaxMode_t               mode,
                                             const void                      *alpha,
                                             const cdnnTensorDescriptor_t    srcDesc,
                                             const void                      *srcData,
                                             const void                      *beta,
                                             const cdnnTensorDescriptor_t    destDesc,
                                             void                            *destData
                                          )
{



}

/** 
 * Function to perform backward softmax.
 * @param handle 
 * @param algorithm
 * @param mode
 * @param *alpha
 * @param srcDesc
 * @param *srcData 
 * @param srcDiffDesc
 * @param *srcDiffData 
 * @param *beta
 * @param destDiffDesc
 * @param *destDiffData 
 * @return cdnnStatus_t
 */
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
                                              )
{


}



/* Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" */

/** 
 * Function to perform forward activation.
 * @param handle 
 * @param algorithm
 * @param mode
 * @param *alpha
 * @param srcDesc
 * @param *srcData 
 * @param *beta
 * @param destDesc
 * @param *destData 
 * @return cdnnStatus_t
 */ 
cdnnStatus_t CDNNWINAPI cdnnActivationForward( cdnnHandle_t                    handle,
                                                  cdnnActivationMode_t            mode,
                                                  const void                      *alpha,
                                                  const cdnnTensorDescriptor_t    srcDesc,
                                                  const void                      *srcData,
                                                  const void                      *beta,
                                                  const cdnnTensorDescriptor_t    destDesc,
                                                  void                            *destData
                                                )
{


}


/** 
 * Function to perform backward activation.
 * @param handle 
 * @param algorithm
 * @param mode
 * @param *alpha
 * @param srcDesc
 * @param *srcData 
 * @param *beta
 * @param destDesc
 * @param *destData 
 * @return cdnnStatus_t
 */ 
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
                                              )
{

	
}
