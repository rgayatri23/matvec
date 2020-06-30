#include "matvec.h"

__global__ void
d_matvec(ARRAY3DGPU m, ARRAY2DGPU x, ARRAY2DGPU y)
{
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    for (int j = threadIdx.y; j < N; j += blockDim.y) {
      __shared__ DataType result;
      extern __shared__ DataType tmp[];
      result = 0;
      __syncthreads();
      for (int k = threadIdx.x; k < N; k += blockDim.x)
      {
        atomicAdd(&result, m(i, j, k) * x(i, k));
//        result += m(i, j, k) * x(i, k);
      }
      y(i, j) += result;
    }
  }
}

void
d_batched_matrix_vector(ARRAY3DGPU m, ARRAY2DGPU x, ARRAY2DGPU y)
{
  size_t Ns = 32*sizeof(DataType);
  dim3 numThreads(32,32,1);
  d_matvec<<<N, 1, Ns>>>(m, x, y);
  cudaDeviceSynchronize();
}
