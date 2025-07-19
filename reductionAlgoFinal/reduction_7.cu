#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

template<unsigned int blockSize>
__device__ int* reduction_7(int *g_idata, int *g_odata)
{
    static __shared__ int sdata[THREADS];
    // each thread loads one element from global to shared mem
    //unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[threadIdx.x] = 0;
    while (i < CUDASIZE)
    {
        sdata[threadIdx.x] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (threadIdx.x < 256)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (threadIdx.x < 64)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 64];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32)
        warpReduce<blockSize>(sdata, threadIdx.x);
    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    //implement second reduction for the summed array
    __syncthreads();
    i = blockIdx.x*(blockSize*2) + threadIdx.x;
    gridSize = blockSize*2*gridDim.x;
    sdata[threadIdx.x] = 0;
    while (i < CUDASIZE)
    {
        sdata[threadIdx.x] += g_odata[i] + g_odata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    if (blockSize >= 512)
    {
        if (threadIdx.x < 256)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (threadIdx.x < 64)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 64];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32) warpReduce<blockSize>(sdata, threadIdx.x);
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    return g_odata;
}
template<unsigned int blockSize>
__device__ int* reduction_71(int *g_idata, int *g_odata)
{
    static __shared__ int sdata[THREADS];
    // each thread loads one element from global to shared mem
    //unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;
    sdata[threadIdx.x] = 0;
    while (i < CUDASIZE)
    {
        sdata[threadIdx.x] += g_idata[i] + g_idata[i+blockDim.x];
        i += gridSize;
    }
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 1024)
    {
        if (threadIdx.x < 512)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512)
    {
        if (threadIdx.x < 256)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (threadIdx.x < 64)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 64];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32)
        warpReduce<blockSize>(sdata, threadIdx.x);
    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    return g_odata;
}

template<unsigned int blockSize>
__device__ int* reduction_72(int *g_idata, int *g_odata)
{
    static __shared__ int sdata[BLOCKS/4];
    // each thread loads one element from global to shared mem
    //unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;
    sdata[threadIdx.x] = 0;
    while (i < BLOCKS/4)
    {
        sdata[threadIdx.x] += g_idata[i] + g_idata[i+blockSize];
        i += gridSize;
    }
    __syncthreads();
    // do reduction in shared mem
    if (blockSize >= 512)
    {
        if (threadIdx.x < 256)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (threadIdx.x < 128)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (threadIdx.x < 64)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + 64];
        }
        __syncthreads();
    }
    if (threadIdx.x < 32)
        warpReduce<blockSize>(sdata, threadIdx.x);
    // write result for this block to global mem
    if (threadIdx.x == 0)
    {
        g_odata[blockIdx.x] = sdata[0];
    }
    return g_odata;
}
