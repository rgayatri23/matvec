#include "matvec.h"

const int N = 1000;
const int repeat = 100;
#define PRINT 1

KOKKOS_INLINE_FUNCTION
int
dot(const member_type& team, array_t m, array_t x)
{
  int result = 0;
  parallel_reduce(
    ThreadVectorRange(team, N),
    [&](const int k, int& update) { update += m(k) * x(k); },
    result);

  return result;
}

KOKKOS_INLINE_FUNCTION void
matvec(const member_type& team, vector_t m, array_t x, array_t y)
{

  parallel_for(
    TeamThreadRange(team, 0, N),
    KOKKOS_LAMBDA(const int j) { y(j) += dot(team, subview(m, j, ALL), x); });
}

void
batched_matrix_vector(matrix_t m, vector_t x, vector_t y)
{
  int vector_size = 32;
  team_policy policy(m.extent(0), Kokkos::AUTO, vector_size);

  parallel_for(
    policy, KOKKOS_LAMBDA(const member_type& team) {
      int i = team.league_rank();
      matvec(
        team, subview(m, i, ALL, ALL), subview(x, i, ALL), subview(y, i, ALL));
    });
}

int
main(int argc, char** argv)
{
  std::cout << "Running the kokkos version." << std::endl;
  Kokkos::initialize(argc, argv);
  {
    // Using time point and system_clock
    std::chrono::time_point<std::chrono::system_clock> start, end, k_start;
    start = std::chrono::system_clock::now();

    // Use default_random_engine object to introduce randomness.
    std::default_random_engine generator;
    // Initialize uniform_int_distribution class.
    std::uniform_int_distribution<DataType> distribution(0, N);

    vector_t y("array-y", N, N);
    vector_t x("array-x", N, N);
    matrix_t m("matrix-m", N, N, N);

    std::cout << "Memory foot-print = "
              << (y.span() + x.span() + m.span()) * (sizeof(DataType)) /
                   (1024 * 1024 * 1024)
              << "GBs" << std::endl;

    auto h_m = create_mirror_view(Kokkos::HostSpace(), m);
    auto h_x = create_mirror_view(Kokkos::HostSpace(), x);
    auto h_y = create_mirror_view(Kokkos::HostSpace(), y);

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k)
          h_m(i, j, k) = ((i + 1) * (j + 1) * (k + 1)) % INT_MAX;

    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        h_x(i, j) = ((i + 1) * (j + 1) * distribution(generator)) % INT_MAX;
        h_y(i, j) = 0;
      }

    k_start = std::chrono::system_clock::now();
    Kokkos::deep_copy(m, h_m);
    Kokkos::deep_copy(x, h_x);
    Kokkos::deep_copy(y, h_y);

    for (int r = 0; r < repeat; ++r)
      batched_matrix_vector(m, x, y);

    h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);

    end = std::chrono::system_clock::now();
    std::chrono::duration<double> k_elapsed = end - k_start;
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Kernel time taken = " << k_elapsed.count() << " seconds"
              << std::endl;
    std::cout << "All done and time taken = " << elapsed.count() << " seconds"
              << std::endl;

#if PRINT
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j)
        std::cout << "y(" << i << ", " << j << ") = " << h_y(i, j) << "\t";
      std::cout << "\n";
    }
#endif
  }
  Kokkos::finalize();

  return 0;
}
