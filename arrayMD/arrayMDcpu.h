#ifndef _ARRAYMD_H
#define _ARRAYMD_H

template<typename T>
struct Array2D
{
  unsigned n1, n2, b1;
  unsigned size;
  T* dptr;

  inline T& operator()(unsigned i1, unsigned i2)
  {
    return dptr[i2 + (n2 * i1)];
  }

  Array2D()
  {
    n1 = n2 = 0, b1 = 0;
    size = 0;
    dptr = nullptr;
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
      delete[] dptr;
  }

  void resize(unsigned in1, unsigned in2)
  {
    if (size && dptr)
      delete[] dptr;

    n1 = in1;
    n2 = in2;
    size = n1 * n2;
    dptr = new T[size];
  }

  T* subArray(int in1)
  {
    return dptr+(in1*n2);
  }
};

template<typename T>
struct Array3D
{
  unsigned n1, n2, n3;
  unsigned size;
  T* dptr;

  inline T& operator()(unsigned i1, unsigned i2, unsigned i3)
  {
    return dptr[i3 + i2 * f2 + i1 * f1];
  }

  Array3D() = default;

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
      delete[] dptr;
  }

  void resize(unsigned in1, unsigned in2, unsigned in3)
  {
    if (size && dptr)
      delete[] dptr;

    n1 = in1;
    n2 = in2;
    n3 = in3;
    size = n1 * n2 * n3;
    f2 = n3;
    f1 = f2 * n2;
    dptr = new T[size];
  }

  T* subArray(int in1, int in2)
  {
    return dptr+(in1*n2 + in2*n2*n3);
  }

private:
  unsigned f2, f1, b1, b2;
};

#endif

