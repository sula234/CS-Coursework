#include <stdio.h>
#include <stdlib.h> 
#define N  1024

__global__ void matrixMul(int* a, int* b, int* c )
{
  int value = 0;
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < N && col < N)
  {
    for ( int k = 0; k < N; ++k )
      value += a[row * N + k] * b[k * N + col];
    c[row * N + col] = value;
  }
}

void verifyOnCPU( int * a, int * b, int * c )
{
  int val = 0;
  int counter = 0;
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      val = 0;
      for ( int k = 0; k < N; ++k )
        val += a[row * N + k] * b[k * N + col];
      c[row * N + col] = val;
      counter++;
    }


}

// int main()
// {
//   int *a, *b, *c_cpu, *c_gpu;
//   size_t size = (N * N) * sizeof(int);
//   cudaMallocManaged(&a, size);
//   cudaMallocManaged(&b, size);
//   cudaMallocManaged(&c_cpu, size);
//   cudaMallocManaged(&c_gpu, size);

//   // Initialize memory
//   for( int row = 0; row < N; ++row )
//     for( int col = 0; col < N; ++col )
//     {
//       a[row*N + col] = rand() % 100 + 1;  
//       b[row*N + col] = rand() % 100 + 1; 
//       c_cpu[row*N + col] = 0;
//       c_gpu[row*N + col] = 0;
//     }

//   dim3 threads_per_block (16, 16, 1);
//   dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

//   matrixMul <<< number_of_blocks, threads_per_block >>> (a, b, c_gpu);

//   cudaDeviceSynchronize(); 

//   verifyOnCPU( a, b, c_cpu );
//   // Compare the two answers to make sure they are equal
//   bool error = false;
//   for( int row = 0; row < N && !error; ++row )
//     for( int col = 0; col < N && !error; ++col )
//       if (c_cpu[row * N + col] != c_gpu[row * N + col])
//       {
//         printf("ERROR at c[%d][%d]\n", row, col);
//         error = true;
//         break;
//       }
//   if (!error)
//     printf("ALL test cases are passed!\n");

//   // Free MEM
//   cudaFree(a); cudaFree(b);
//   cudaFree( c_cpu ); cudaFree( c_gpu );
// }
