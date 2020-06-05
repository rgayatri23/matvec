/*
 * ArrayMD class for multidimensional arrays
 *
 * Rahulkumar Gayatri
 * */
#include <array>
#include <cassert>
#include <iostream>

template<size_t dim, typename T>
void
compute_offsets(std::array<size_t, dim>& offsets, size_t& size, T arg)
{
  offsets[dim - 1] = arg;
  size *= arg;
}

template<size_t dim, typename T, typename... Idx>
void
compute_offsets(std::array<size_t, dim>& offsets,
                size_t& size,
                T arg,
                Idx... args)
{
  offsets[dim - (sizeof...(args)) - 1] = arg;
  size *= arg;
  compute_offsets(offsets, size, args...);
}

template<size_t dim, typename T>
void
compute_args(std::array<size_t, dim>& offsets, size_t& index, T arg)
{
  index += arg;
}

template<size_t dim, typename T, typename... Idx>
void
compute_args(std::array<size_t, dim>& offsets,
             size_t& index,
             T arg,
             Idx... args)
{
  //  TODO - The assert in this case would not work if the build type is set to
  //  "Release" instead of "Debug" Need to make our own version of always_assert
  //  that stays on always on. assert(arg > offsets[dim-sizeof...(args)-1] &&
  //  "Whoops the index is higher than the dimension ");
  if (arg > offsets[dim - sizeof...(args) - 1]) {
    printf("Whoops the index is higher than the dimension \n");
    abort();
  }

  auto index_eval = arg;
  for (int i = dim - sizeof...(args); i < dim; ++i)
    index_eval *= offsets[i];
  index += index_eval;

  compute_args(offsets, index, args...);

  return;
}

template<typename T, size_t dim>
struct ArrayMD
{
  std::array<size_t, dim> m_offsets;
  size_t size;
  T* dptr;

  ArrayMD() = default;

  template<typename... Idx>
  ArrayMD(Idx... args)
  {
    const auto N = sizeof...(args);
    size = 1;
    static_assert(N == dim,
                  "Dimensionality passed does not match the argument list");

    compute_offsets(m_offsets, size, args...);

    // Allocate memory for size number of T-elements
    dptr = new T[size];
  }

  template<typename... Idx>
  inline T& operator()(Idx... args)
  {
    size_t index = 0;
    compute_args(m_offsets, index, args...);
    return dptr[index];
  }

  ArrayMD(const ArrayMD& p)
  {
    m_offsets = p.m_offsets;
    size = 0;
    dptr = p.dptr;
  }

  void operator=(const ArrayMD& p)
  {
    m_offsets = p.m_offsets;
    size = 0;
    dptr = p.dptr;
  }

  // Destructor
  ~ArrayMD()
  {
    if (size && dptr)
      delete[] dptr;
  }
};
