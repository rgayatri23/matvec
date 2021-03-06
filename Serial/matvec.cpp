#include "../arrayMD/arrayMD.h"
#include <bits/stdc++.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>

using namespace std::chrono;
using DataType = int;
#define ARRAY2D ArrayMD<DataType, 2, Device::cpu>
#define ARRAY3D ArrayMD<DataType, 3, Device::cpu>

const int N = 1000;
const int repeat = 100;
#define PRINT 1

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
    y[j] += dot(m.subArray(i,j), x.subArray(i));
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

  ARRAY2D y(N, N);
  ARRAY2D x(N, N);
  ARRAY3D m(N, N, N);

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

  for (int i = 0; i < repeat; ++i)
    batched_matrix_vector(m, x, y);

  end = system_clock::now();

  duration<double> k_elapsed = end - k_start;
  duration<double> elapsed = end - start;

  std::cout << "Kernel time taken = " << k_elapsed.count() << " seconds"
            << std::endl;
  std::cout << "All done and time taken = " << elapsed.count() << " seconds"
            << std::endl;

  // Print out the output. Comment out unless needed.
#if PRINT
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j)
      std::cout << "y(" << i << ", " << j << ") = " << y(i, j) << "\t";
    std::cout << "\n";
  }
#endif

  return 0;
}
