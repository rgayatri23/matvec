#ifndef _MATVEC_H
#define _MATVEC_H

#include "arrayMD.h"
#include "arrayMDgpu.h"
#include <bits/stdc++.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <random>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>

using namespace std::chrono;
using DataType = int;
#define ARRAY2D ArrayMD<DataType, 2>
#define ARRAY3D ArrayMD<DataType, 3>

#define ARRAY2DGPU Array2D<DataType>
#define ARRAY3DGPU Array3D<DataType>

// Function definitions for GPU functions
__global__
void
d_matvec(ARRAY3DGPU m, ARRAY2DGPU x, ARRAY2DGPU y);

void
d_batched_matrix_vector(ARRAY3DGPU m, ARRAY2DGPU x, ARRAY2DGPU y);

const int N = 1000;
const int repeat = 100;
#define PRINT 1

#endif
