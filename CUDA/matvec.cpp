#include "matvec.h"

__device__ void
dot(int i, int j, DataType result[], ARRAY3DGPU& m, ARRAY2DGPU& x)
{
  int yId = threadIdx.y * blockDim.x;
  for (int k = threadIdx.x; k < blockDim.x; k += blockDim.x)
    result[yId + k] = 0;

  for (int k = threadIdx.x; k < N; k += blockDim.x)
    result[yId + (k % blockDim.x)] += m(i, j, k) * x(i, k);

  for (int k = threadIdx.x + 1; k < blockDim.x; k += blockDim.x)
    atomicAdd(&(result[yId]), result[yId + k]);
}

__global__ void
matvec(ARRAY3DGPU m, ARRAY2DGPU x, ARRAY2DGPU y)
{
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    extern __shared__ DataType result[];
    int yId = threadIdx.y * blockDim.x;

    for (int j = threadIdx.y; j < N; j += blockDim.y) {
      dot(i, j, result, m, x);

      y(i, j) += result[yId];
    }
  }
}

void
batched_matrix_vector(ARRAY3DGPU m, ARRAY2DGPU x, ARRAY2DGPU y)
{
  const int Nx = 32, Ny = 32;
  size_t Ns = Nx * Ny * sizeof(DataType);
  dim3 numThreads(Nx, Ny, 1);
  matvec<<<N, numThreads, Ns>>>(m, x, y);
  cudaDeviceSynchronize();
}

int
main(int argc, char** argv)
{
  std::cout << "Running the CUDA version." << std::endl;

  // Using time point and system_clock
  time_point<system_clock> start, end, k_start, k_end;
  start = system_clock::now();

  // Use default_random_engine object to introduce randomness.
  std::default_random_engine generator;
  // Initialize uniform_int_distribution class.
  std::uniform_int_distribution<DataType> distribution(0, N);

  // Host Arrays
  ARRAY2D y(N, N);
  ARRAY2D x(N, N);
  ARRAY3D m(N, N, N);

  // Device Arrays
  ARRAY2DGPU d_x(N, N);
  ARRAY2DGPU d_y(N, N);
  ARRAY3DGPU d_m(N, N, N);

  std::cout << "Memory foot-print = "
            << (y.size + x.size + m.size) * (sizeof(DataType)) /
                 (1024 * 1024 * 1024)
            << "GBs" << std::endl;

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k)
        m(i, j, k) = ((i + 1) * (j + 1) * (k + 1)) % INT_MAX;

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      x(i, j) = ((i + 1) * (j + 1) * distribution(generator)) % INT_MAX;
      y(i, j) = 0;
    }

  k_start = system_clock::now();
  // copy host data to device data
  checkCudaErrors(cudaMemcpy(
    d_x.dptr, x.dptr, x.size * sizeof(DataType), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(
    d_y.dptr, y.dptr, y.size * sizeof(DataType), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(
    d_m.dptr, m.dptr, m.size * sizeof(DataType), cudaMemcpyHostToDevice));

  // Start GPU wrok from here....
  for (int i = 0; i < repeat; ++i)
    batched_matrix_vector(d_m, d_x, d_y);

  checkCudaErrors(cudaMemcpy(
    y.dptr, d_y.dptr, y.size * sizeof(DataType), cudaMemcpyDeviceToHost));

  end = system_clock::now();

  duration<double> k_elapsed = end - k_start;
  duration<double> elapsed = end - start;

  std::cout << "Kernel time taken including data movement = "
            << k_elapsed.count() << " seconds" << std::endl;
  std::cout << "All done and time taken = " << elapsed.count() << " seconds"
            << std::endl;

#if PRINT
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j)
      std::cout << "y(" << i << ", " << j << ") = " << y(i, j) << "\t";
    std::cout << "\n";
  }
#endif

  return 0;
}
