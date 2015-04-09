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

const char * CDNNWINAPI cdnnGetErrorString(cdnnStatus_t status)
{

 
}

cdnnStatus_t CDNNWINAPI cdnnCreate(cdnnHandle_t *handle)
{

 
}

cdnnStatus_t CDNNWINAPI cdnnDestroy(cdnnHandle_t handle)
{

 
}