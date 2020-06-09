/*
 * ArrayMD class for multidimensional arrays
 *
 * Rahulkumar Gayatri
 * */
#ifndef _ARRAYMD_H
#define _ARRAYMD_H

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
compute_subarray_offset(std::array<size_t, dim>& offsets,
                        size_t& subarray_offset,
                        T arg)
{
  subarray_offset += arg * offsets[dim - 1];
}

template<size_t dim, typename T, typename... Idx>
void
compute_subarray_offset(std::array<size_t, dim>& offsets,
                        size_t& subarray_offset,
                        T arg,
                        Idx... args)
{
  size_t tmp = 1;
  for (int i = 0; i < dim - 1; ++i)
    tmp *= offsets[dim - (sizeof...(args)) - i];
  subarray_offset += arg * tmp;

  compute_subarray_offset(offsets, subarray_offset, args...);
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
  std::array<size_t, dim> m_subarray_offsets;
  size_t size;
  size_t subarray_index;
  T* dptr;

  ArrayMD() = default;

  template<typename... Idx>
  ArrayMD(Idx... args)
  {
    const auto N = sizeof...(args);
    static_assert(N == dim,
                  "Dimensionality passed does not match the argument list");
    size = 1;
    subarray_index = 0;
    m_offsets.fill(0);
    m_subarray_offsets.fill(0);

    compute_offsets(m_offsets, size, args...);

    // Allocate memory for size number of T-elements
    dptr = new T[size];
  }

  template<typename... Idx>
  inline T& operator()(Idx... args)
  {
    const auto N = sizeof...(args);
    static_assert(N == dim, "parameters passed exceed the dimensionality");
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

  template<typename... Idx>
  T* subArray(Idx... args)
  {
    size_t subarray_offset = 0;
    const auto N = sizeof...(args);
    static_assert(N == dim - 1,
                  "parameters passed to subArray should be 1 less than actual "
                  "dimension of the arrayMD");

    compute_subarray_offset(m_offsets, subarray_offset, args...);

    return (dptr + subarray_offset);
  }

  // Destructor
  ~ArrayMD()
  {
    if (size && dptr)
      delete[] dptr;
  }
};

#endif
