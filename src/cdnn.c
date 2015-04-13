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
 * Return error string from error code. 
 */
const char * CDNNWINAPI cdnnGetErrorString(cdnnStatus_t status)
{

    return "Yep";
}

/**
 * Create an instance of cdnnHandle_t. 
 */
cdnnStatus_t CDNNWINAPI cdnnCreate(cdnnHandle_t *handle)
{

     return CDNN_STATUS_SUCCESS;
}

/**
 * Destroy an instance of cdnnHandle_t. 
 */
cdnnStatus_t CDNNWINAPI cdnnDestroy(cdnnHandle_t handle)
{

     return CDNN_STATUS_SUCCESS;
}
