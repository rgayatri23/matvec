#include "matvec.h"

const int N = 1000;
const int repeat = 100;
#define PRINT 1

int
dot(int i, int j, DataType* m, DataType* x)
{
  int result = 0;
  //#if defined(OPENMP_TARGET)
  //#pragma omp simd
  //#endif
  for (int k = 0; k < N; ++k)
    result += m[k] * x[k];

  return result;
}

void
matvec(int i, ARRAY3D& m, ARRAY2D& x, DataType* y)
{
#if defined(OPENMP_TARGET)
#pragma omp for
#endif
  for (int j = 0; j < N; ++j) {
    y[j] += dot(i, j, m.subArray(i, j), x.subArray(i));
  }
}

void
batched_matrix_vector(ARRAY3D& m, ARRAY2D& x, ARRAY2D& y)
{
#if defined(OPENMP_TARGET)
  // Map m,x,y onto the device
#pragma omp target enter data map(to : m, x, y)
#pragma omp target enter data map(                                             \
  to                                                                           \
  : m.dptr [0:m.size], x.dptr [0:x.size], y.dptr [0:y.size])
#pragma omp target teams distribute
#elif defined(_OPENMP)
#pragma omp parallel for default(none) shared(m, x, y, N)
#endif
  for (int i = 0; i < N; ++i) {
    // Launch parallel threads
#if defined(OPENMP_TARGET)
#pragma omp parallel
#endif
    {
      matvec(i, m, x, y.subArray(i));
    }
  }

#if defined(OPENMP_TARGET)
  // Write back results from device to y
#pragma omp target exit data map(from : y.dptr [0:y.size])
#endif
}

int
main(int argc, char** argv)
{
#if defined(_OPENMP)
  int tid = 0, numThreads = 0, numTeams = 0;

#if defined(OPENMP_TARGET)
  std::cout << "Running the OpenMP TARGET version." << std::endl;
#pragma omp target map(tofrom : numTeams, numThreads)
#pragma omp teams shared(numTeams) private(tid)
  {
    tid = omp_get_team_num();
    if (tid == 0) {
      numTeams = omp_get_num_teams();
#pragma omp parallel
      {
        int ttid = omp_get_thread_num();
        if (ttid == 0)
          numThreads = omp_get_num_threads();
      }
    }
  }
  std::cout << "Number of OpenMP Teams = " << numTeams << std::endl;
  std::cout << "Number of OpenMP DEVICE Threads = " << numThreads << std::endl;
#else

  // Print number of threads used by OpenMP
  std::cout << "Running the OpenMP~3.0 version." << std::endl;
#pragma omp parallel shared(numThreads) private(tid)
  {
    tid = omp_get_thread_num();
    if (tid == 0)
      numThreads = omp_get_num_threads();
  }
  printf("Number of OpenMP Threads = %d\n", numThreads);
#endif

#endif

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
