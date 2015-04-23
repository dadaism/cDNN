#include <stdio.h>
#include <math.h>
#include "cdnn.h"
#include "cdnn_util.h"

typedef float FLOAT_T;

cdnnHandle_t handle;
cdnnTensorDescriptor_t srcDesc;
cdnnTensorDescriptor_t destDesc;
cdnnTensorFormat_t  format = CDNN_TENSOR_NCHW;
cdnnDataType_t dataType = CDNN_DATA_FLOAT;
cdnnStatus_t status;

int main()
{
    bool flag = true;
    int n = 1, c = 1, h = 8, w = 8;
    int nStride, cStride, hStride, wStride;
    float value, alpha, beta;

    status = cdnnCreateTensorDescriptor(&srcDesc);
    if (status==CDNN_STATUS_SUCCESS)
        fprintf(stderr, "cdnnCreateTensorDescriptor PASS.\n");
    else
        fprintf(stderr, "cdnnCreateTensorDescriptor FAIL.\n");

    cdnnCreateTensorDescriptor(&destDesc);
    
    status = cdnnSetTensor4dDescriptor(srcDesc, format, dataType, n, c, h, w );
    if (status==CDNN_STATUS_SUCCESS)
        fprintf(stderr, "cdnnSetTensor4dDescriptor PASS.\n");
    else
        fprintf(stderr, "cdnnSetTensor4dDescriptor FAIL.\n");

    cdnnSetTensor4dDescriptor(destDesc, format, dataType, n, c, h, w );

//    printTensorDesc(srcDesc);
//    printTensorDesc(destDesc);

    FLOAT_T *srcData = (FLOAT_T*)allocTensor(srcDesc);
    FLOAT_T *destData = (FLOAT_T*)allocTensor(destDesc);

    /* Test cdnnSetTensor */
    flag = true; value = 2.5;
    cdnnSetTensor(handle, srcDesc, srcData, &value);
    status = cdnnGetTensor4dDescriptor(srcDesc, &dataType, 
                                       &n, &c, &h, &w, 
                                       &nStride, &cStride, 
                                       &hStride, &wStride);
    if (status==CDNN_STATUS_SUCCESS)
        fprintf(stderr, "cdnnGetTensor4dDescriptor PASS.\n");
    else
        fprintf(stderr, "cdnnGetTensor4dDescriptor FAIL.\n");

    for (int i=0; i<n*c*h*w; ++i) {
        if ( fabs(srcData[i] - value) > 0.00001 )  flag = false;
    }
    if (flag) fprintf(stderr, "cdnnSetTensor PASS.\n");
    else fprintf(stderr, "cdnnSetTensor FAIL.\n");

    /* Test cdnnTransformTensor */
    flag = true; value = 2.5;
    cdnnSetTensor(handle, srcDesc, srcData, &value);
    value = 3.14;
    cdnnSetTensor(handle, destDesc, destData, &value);
    alpha = 2.7; beta = 32.1;
    cdnnTransformTensor( handle, &alpha, srcDesc, srcData,
                         &beta, destDesc, destData );
    for (int i=0; i<n*c*h*w; ++i) {
        if ( fabs(destData[i] - (2.5*alpha+value*beta) ) > 0.0001 )  flag = false;
    }
    if (flag) fprintf(stderr, "cdnnTransformTensor PASS.\n");
    else fprintf(stderr, "cdnnTransformTensor FAIL.\n");

    /* Test cdnnScaleTensor */
    // Not Implemented yet



    /* Test cdnnScaleTensor */
    flag = true; value = 2.5; alpha = 1.95;
    
    cdnnSetTensor(handle, srcDesc, srcData, &value);
    cdnnScaleTensor(handle, srcDesc, srcData, &alpha);

    for (int i=0; i<n*c*h*w; ++i) {
        if ( fabs(srcData[i] - value*alpha) > 0.00001 )  flag = false;
    }
    if (flag) fprintf(stderr, "cdnnScaleTensor PASS.\n");
    else fprintf(stderr, "cdnnScaleTensor FAIL.\n");
//    printTensor(srcDesc, srcData);

    status = cdnnDestroyTensorDescriptor(srcDesc);
    if (status==CDNN_STATUS_SUCCESS)
        fprintf(stderr, "cdnnDestroyTensorDescriptor PASS.\n");
    else
        fprintf(stderr, "cdnnDestroyTensorDescriptor FAIL.\n");

    cdnnDestroyTensorDescriptor(destDesc);
}
