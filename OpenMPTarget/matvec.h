#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <bits/stdc++.h>

#if _OPENMP
#include <omp.h>
#endif

using DataType = int;

//Use RightLayout for GPUs and LeftLayout for CPUs in ArrayMD class
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
  ~Array2D()
  {
    if (size && dptr)
      delete dptr;
  }
};

template<typename T>
class Array3D
{
private:
  unsigned f2, f1;

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
    dptr = new T[size];
  }
  ~Array3D()
  {
    if (size && dptr)
      delete dptr;
    ;
  }
};

//RightLayout for GPUs
template<typename T>
class Array2Dgpu
{
public:
  unsigned n1, n2;
  unsigned size;
  T* dptr;

  inline T& operator()(unsigned i1, unsigned i2)
  {
    return dptr[i1 + (n1 * i2)];
  }

  Array2Dgpu()
  {
    n1 = n2 = 0;
    size = 0;
    dptr = NULL;
  }
  Array2Dgpu(const Array2Dgpu& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    size = 0;
    dptr = p.dptr;
  }
  Array2Dgpu(int in1, int in2)
  {
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = new T[size];
  }
  ~Array2Dgpu()
  {
    if (size && dptr)
      delete dptr;
  }
};

template<typename T>
class Array3Dgpu
{
private:
  unsigned f2, f3;

public:
  unsigned n1, n2, n3;
  unsigned size;
  T* dptr;
  inline T& operator()(unsigned i1, unsigned i2, unsigned i3)
  {
    return dptr[i1 + i2 * n1 + i3 * n1 * n2];
  }

  Array3Dgpu()
  {
    n1 = n2 = n3 = 0;
    size = 0;
    dptr = NULL;
  }
  Array3Dgpu(const Array3Dgpu& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    n3 = p.n3;
    size = 0;
    dptr = p.dptr;
    f2 = n1;
    f3 = f2 * n2;
  }
  Array3Dgpu(unsigned in1, unsigned in2, unsigned in3)
  {
    n1 = in1;
    n2 = in2;
    n3 = in3;
    size = n1 * n2 * n3;
    f2 = n1;
    f2 = f2 * n2;
    dptr = new T[size];
  }
  ~Array3Dgpu()
  {
    if (size && dptr)
      delete dptr;
    ;
  }
};

//Function Definitions.
void batched_matrix_vector(ARRAY3D& m,
                      ARRAY2D& x,
                      ARRAY2D& y);
void matvec(int i, ARRAY3D& m, ARRAY2D& x, ARRAY2D& y);

int dot(int i, int j, ARRAY3D& m, ARRAY2D& x);
#pragma omp end declare target
