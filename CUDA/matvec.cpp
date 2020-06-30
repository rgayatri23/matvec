#include "matvec.h"

int
dot(DataType* m, DataType* x)
{
  int result = 0;
  for (int k = 0; k < N; ++k)
    result += m[k] * x[k];

  return result;
}

void
matvec(int i, ARRAY3D& m, ARRAY2D& x, DataType* y)
{
  for (int j = 0; j < N; ++j)
    y[j] += dot(m.subArray(i, j), x.subArray(i));
}

void
batched_matrix_vector(ARRAY3D& m, ARRAY2D& x, ARRAY2D& y)
{
  for (int i = 0; i < N; ++i)
    matvec(i, m, x, y.subArray(i));
}

int
main(int argc, char** argv)
{
  std::cout << "Running the basic sequential version." << std::endl;

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
    d_batched_matrix_vector(d_m, d_x, d_y);

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
