#include "reduction_header.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include "reduction_warp.cu"
#include "reduction_7.cu"
#include "reduction_1.cu"
#include "mul.cu"

// ************************************************************
// Answer or Service task 2: The maximum size allowed is 1024**
// ************************************************************
__global__ void cuda_global(int *dev_a, int *dev_b)
{

    //dev_b = reduction_7<THREADS>(dev_a, dev_b);
    // if(blockDim.x == THREADS)
    //   dev_b = reduction_71<THREADS>(dev_a, dev_b);
    // if(blockDim.x == BLOCKS/4)
    //   dev_b = reduction_72<BLOCKS/4>(dev_a, dev_b);  

    dev_b = reduction_10(dev_a, dev_b);
}

int* initArray()
{
  static int array[CUDASIZE];
  for(int i = 0; i < CUDASIZE; i++)
  {
    array[i] = 1;
  }
  return array;
}

int checkResults(int *a)
{
  int sum = 0;
  for(int i = 0; i < CUDASIZE; i++)
  {

    sum = sum + a[i];
  }
  return sum;
}

void print_array(int *array, int length)
{
    printf("{");
    for (int i = 0; i < length; i++) { 
      printf(" %d", array[i]);
      }
    printf("...........}\n\n");
}

void wrapper()
{

  int *a, *b, *c_cpu, *c_gpu;
  size_t size = (BLOCKS* BLOCKS) * sizeof(int);
  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c_cpu, size);
  cudaMallocManaged(&c_gpu, size);

  // Initialize memory for arrays
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = -9 + rand()%19;  
      b[row*N + col] = -9 + rand()%19; 
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  printf("MATRIX A = ");
  print_array(a, 10);

  printf("MATRIX B = ");
  print_array(b, 10);

  dim3 threads_per_block (16, 16, 1);
  dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

  matrixMul <<< number_of_blocks, threads_per_block >>> (a, b, c_gpu);

  cudaDeviceSynchronize(); 

  verifyOnCPU( a, b, c_cpu );
  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("ALL test cases for muliplication are passed!\n\n");

  printf("MATRIX C = ");
  print_array(c_cpu, 10);
  // REDUCTION PART
  int *dev_b;
  int *dev_a;


  cudaDeviceProp device;
  cudaGetDeviceProperties(&device, 0);
  printf("  --- General information for device START ---\n");
  printf("Name: %s;\n", device.name);
  printf("Max threads Per Block: %d\n", device.maxThreadsPerBlock);
  printf("Max thread dimensions: (%d, %d, %d)\n", device.maxThreadsDim[0], device.maxThreadsDim[1], device.maxThreadsDim[2]);
  printf("Max grid dimensions: (%d, %d, %d)\n", device.maxGridSize[0], device.maxGridSize[1], device.maxGridSize[2]);
  printf("  --- General information for device END ---\n\n");

  cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

  cudaMalloc((void**)&dev_a, CUDASIZE*sizeof(int));
  cudaMemcpy(dev_a, c_cpu, CUDASIZE*sizeof(int), cudaMemcpyHostToDevice);

  // VARIANT 7
  // cudaMalloc((void**)&dev_b, (BLOCKS/2)*sizeof(int));
  // cuda_global<<<BLOCKS/2, THREADS>>>(dev_a, dev_b);
  // cuda_global<<<1, BLOCKS/4>>>(dev_a, dev_b);

  // VARIANT 1
  cudaMalloc((void**)&dev_b, BLOCKS*sizeof(int));
  cuda_global<<<BLOCKS, THREADS>>>(dev_a, dev_b);
  cuda_global<<<1, BLOCKS>>>(dev_b, dev_b);

  cudaMemcpy(b, dev_b, sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
  printf(" --- Elapsed time  ---\n %.5f ms; \n\n", elapsedTime);
  printf("  ---  TEST CASE FOR REDUCTION --- \n");
  
  printf("GPU RESULTS: sum = %d \n", b[0]);
  int sum = checkResults(c_cpu);
  printf("CPU RESULTS: sum = %d\n\n", sum);

  printf("Answer or Service task 2: The maximum size allowed is 1024 by 1024 (1048576)\n");


}
