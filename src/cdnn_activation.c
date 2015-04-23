#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cdnn_core.h"
#include "cdnn.h"

#ifndef CDNNWINAPI
#ifdef _WIN32
#define CDNNWINAPI __stdcall
#else
#define CDNNWINAPI
#endif
#endif

// Note: tanh is available in math.h

inline float sigmoidFast(float x)
{
    return x/(1+abs(x));
}


inline float sigmoid(float x)
{
    return 1/(1+exp(-x));
}


inline float ReLU(float x)
{
    return x>0? x : 0;
}

/**
 * Softmax functions: All of the form "output = alpha * Op(inputs) + beta * output" 
 * cdnnSoftmaxAlgorithm_t: 
 *     CDNN_SOFTMAX_MODE_INSTANCE = 0    compute the softmax over all C, H, W for each N 
 *     CDNN_SOFTMAX_MODE_CHANNEL = 1     compute the softmax over all C for each H, W, N 
 *
 * f(x) = exp(x) / sum( exp(Xi) ) 
 * f'(x) = (1-f(x))*f(x)
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


    return CDNN_STATUS_SUCCESS;
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
cdnnStatus_t CDNNWINAPI cdnnSoftmaxBackward(  cdnnHandle_t                    handle,
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

    return CDNN_STATUS_SUCCESS;
}



/**
 * Activation functions: All of the form "output = alpha * Op(inputs) + beta * output" 
 * cdnnActivationMode_t:
 *    CUDNN_ACTIVATION_SIGMOID = 0
 *    CUDNN_ACTIVATION_RELU    = 1
 *    CUDNN_ACTIVATION_TANH    = 2
 *
 * Sigmoid:
 *    f(x) = 1/(1+exp(-x))
 *    f'(x) = (1-f(x))*f(x)
 *
 * ReLU:
 *    f(x) =
 *    f'(x) = 
 *
 * Tanh: 
 *    f(x) =
 *    f'(x) = 
 */


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
    /* Function Pointer */


    return CDNN_STATUS_SUCCESS;
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
    return CDNN_STATUS_SUCCESS;
}
