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

template<bool B, typename T>
using enable_if_t = typename std::enable_if<B, T>::type;

template<size_t dim, typename T>
inline void
compute_offsets(std::array<size_t, dim>& offsets, size_t& size, T arg)
{
  offsets[dim - 1] = arg;
  size *= arg;
}

template<size_t dim, typename T, typename... Idx>
inline void
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
inline void
compute_args(std::array<size_t, dim>& offsets, size_t& index, T arg)
{
  index += arg;
}

template<size_t dim, typename T, typename... Idx>
inline void
compute_args(std::array<size_t, dim>& offsets,
             size_t& index,
             T arg,
             Idx... args)
{

  auto index_eval = arg;
  for (int i = dim - sizeof...(args); i < dim; ++i)
    index_eval *= offsets[i];
  index += index_eval;

  compute_args(offsets, index, args...);

  return;
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

template<typename T, size_t dim>
struct ArrayMD
{
  std::array<size_t, dim> m_offsets;
  size_t size;
  size_t subarray_offset;
  T* dptr;

  ArrayMD() = default;

  template<typename... Idx>
  ArrayMD(Idx... args)
  {
    const auto N = sizeof...(args);
    static_assert(N == dim,
                  "Dimensionality passed does not match the argument list");
    size = 1;
    m_offsets.fill(0);

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

  ArrayMD& operator=(ArrayMD&&) = default;

  ArrayMD& operator=(const ArrayMD& p)
  {
    m_offsets = p.m_offsets;
    size = 0;
    dptr = p.dptr;

    return *this;
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

  template<typename... Idx>
  void resize(Idx... args)
  {
    const auto N = sizeof...(args);
    static_assert(N == dim,
                  "Dimensionality passed does not match the argument list");
    size = 1;
    m_offsets.fill(0);

    compute_offsets(m_offsets, size, args...);

    // Allocate memory for size number of T-elements
    dptr = new T[size];
  }

  T* data() { return dptr; }

  // Destructor
  ~ArrayMD()
  {
    if (size && dptr)
      delete[] dptr;
  }
};
#endif
