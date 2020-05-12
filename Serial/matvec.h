#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <bits/stdc++.h>

using namespace std::chrono;
using DataType = int;

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
