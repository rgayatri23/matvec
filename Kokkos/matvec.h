#include <Kokkos_Core.hpp>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>

// using namsespace std::chrono;
using DataType = int;

using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using Layout = ExecSpace::array_layout;
using MemSpace = ExecSpace::memory_space;

using Kokkos::View;
using team_policy = Kokkos::TeamPolicy<ExecSpace>;
using member_type = team_policy::member_type;
using Kokkos::ALL;
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::subview;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;

// 1D,2D,3D views for int's
// using vector_t = View<DataType**, Layout>;
// using matrix_t = View<DataType***, Layout>;
using array_t = View<DataType*, Kokkos::LayoutRight>;
using vector_t = View<DataType**, Kokkos::LayoutRight>;
using matrix_t = View<DataType***, Kokkos::LayoutRight>;
