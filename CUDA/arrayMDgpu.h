#ifndef LMP_ARRAYMD_H
#define LMP_ARRAYMD_H

/* ----------------------------------------------------------------
   Class library implementing simple array structures similar to Fortran,
   except since this is c++ the fast running index are the rightmost
   ( For this "CPU" version. )

   These are the necessary set for the snap demo code, and may not be
   completely defined for all combinations of sub-dimensional references
   and copy constructors.

   2D through 6D arrays are supported,
       ArrahxD<type> newarray;
   The default constructor initializes bounds, but not memory
   For a 3D example, to allocate memory use newarray.resize( n1,n2,n3)

   The '()' operator is overlaaded here to access array elements.

   Copy constructors are provided for openmp private/firstprivate.
   In this case, the data pointer must be allocated with resize or
   assignment as above.

   When using the GPU, the data pointers must be set up another way,
   presumably with "#pragma enter data map(alloc..." and a deep copy
   of the pointer. The "rebound" routine is a copy of "resize" without
   allocation.

   Sarah Anderson, Cray Inc.
   ----------------------------------------------------------------*/

// version that prints
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define CUDA_HOSTDEV __host__ __device__
template<typename T>
struct Array2D
{
  unsigned n1, n2;
  unsigned size;
  T* dptr;

  CUDA_HOSTDEV
  inline T& operator()(unsigned i1, unsigned i2)
  {
    return dptr[i2 + (n2 * i1)];
  }

  Array2D() = default;

  CUDA_HOSTDEV
  Array2D(const Array2D& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    size = 0;
    dptr = p.dptr;
  }

  CUDA_HOSTDEV
  Array2D(int in1, int in2)
  {
    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    checkCudaErrors(cudaMalloc(&dptr, size * sizeof(T)));
  }

  ~Array2D()
  {
    if (size && dptr)
      checkCudaErrors(cudaFree(dptr));
  }
};

template<typename T>
struct Array3D
{
private:
  unsigned f2, f3, b1, b2;

public:
  unsigned n1, n2, n3;
  unsigned size;
  T* dptr;

  CUDA_HOSTDEV
  inline T& operator()(unsigned i1, unsigned i2, unsigned i3)
  {
//    return dptr[i1 + i2 * f2 + i3 * f3];
    return dptr[i3 + i2*n3 + i1*n2*n3];
  }

  Array3D()
  {
    n1 = n2 = n3 = 0;
    size = 0;
    dptr = NULL;
  }

  CUDA_HOSTDEV
  Array3D(const Array3D& p)
  {
    n1 = p.n1;
    n2 = p.n2;
    n3 = p.n3;
    size = 0;
    dptr = p.dptr;
    f2 = n1;
    f3 = f2 * n2;
  }

  CUDA_HOSTDEV
  Array3D(unsigned in1, unsigned in2, unsigned in3)
  {
    n1 = in1;
    n2 = in2;
    n3 = in3;
    size = n1 * n2 * n3;
    f2 = n1;
    f3 = f2 * n2;
    checkCudaErrors(cudaMalloc(&dptr, size * sizeof(T)));
  }

  ~Array3D()
  {
    if (size && dptr)
      checkCudaErrors(cudaFree(dptr));
  }
};

template<typename T>
struct DeviceArray2D
{
  Array2D<T> m_array;

  DeviceArray2D() = default;

  ~DeviceArray2D() = default;

  DeviceArray2D(const DeviceArray2D&p) = default;

  CUDA_HOSTDEV
  DeviceArray2D(const Array2D<T>& p)
    :m_array(p)
  {}

  CUDA_HOSTDEV
  inline T& operator()(unsigned i1, unsigned i2)
  {
    return m_array(i1,i2);
  }

};

#endif
