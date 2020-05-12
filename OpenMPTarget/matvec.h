#include <bits/stdc++.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>

#if _OPENMP
#include <omp.h>
#endif

using DataType = int;

// Use RightLayout for GPUs and LeftLayout for CPUs in ArrayMD class
#if defined(OPENMP_TARGET)
#define ARRAY2D Array2D<DataType>
#define ARRAY3D Array3D<DataType>
#else
#define ARRAY2D Array2D<DataType>
#define ARRAY3D Array3D<DataType>
#endif

using namespace std::chrono;

#pragma omp declare target
template<typename T>
class Array2D
{
private:
  unsigned b1;

public:
  unsigned n1, n2;
  unsigned size;
  T* dptr;

  inline T& operator()(unsigned i1, unsigned i2)
  {
    return dptr[i2 + (n2 * i1)];
  }

  Array2D()
  {
    n1 = n2 = 0;
    size = 0;
    dptr = NULL;
  }

  Array2D(const Array2D& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    size = 0;
    dptr = p.dptr;
  }

  Array2D(int in1, int in2)
  {
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = new T[size];
  }

  Array2D(int in1, int in2, T* ptr)
  {
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = ptr;
  }

  ~Array2D()
  {
    if (size && dptr)
      delete dptr;
  }

  T* subArray(unsigned in1) { return (dptr + in1 * n2); }
  inline T& operator()(unsigned i2) { return dptr[b1 + i2]; }
};

template<typename T>
class Array3D
{
private:
  unsigned f2, f1;
  unsigned b1, b2;

public:
  unsigned n1, n2, n3;
  unsigned size;
  T* dptr;
  inline T& operator()(unsigned i1, unsigned i2, unsigned i3)
  {
    return dptr[i3 + i2 * f2 + i1 * f1];
  }

  Array3D()
  {
    n1 = n2 = n3 = 0;
    size = 0;
    b1 = 0;
    b2 = 0;
    dptr = NULL;
  }

  Array3D(const Array3D& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    n3 = p.n3;
    size = 0;
    dptr = p.dptr;
    f2 = n3;
    f1 = f2 * n2;
  }

  Array3D(unsigned in1, unsigned in2, unsigned in3)
  {
    n1 = in1;
    n2 = in2;
    n3 = in3;
    size = n1 * n2 * n3;
    f2 = n3;
    f1 = f2 * n2;
    b1 = 0;
    b2 = 0;
    dptr = new T[size];
  }

  T* subArray(int in1, int in2) { return (dptr + (in1 * n2 * n3 + in2 * n2)); }

  ~Array3D()
  {
    if (size && dptr)
      delete dptr;
    ;
  }
};

// Function Definitions.
void
batched_matrix_vector(ARRAY3D& m, ARRAY2D& x, ARRAY2D& y);
void
matvec(int i, ARRAY2D& m, ARRAY2D& x, DataType* y);

int
dot(int i, int j, DataType* m, DataType* x);
#pragma omp end declare target
