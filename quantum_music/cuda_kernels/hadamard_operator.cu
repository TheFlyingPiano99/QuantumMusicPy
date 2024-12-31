#include <cupy/complex.cuh>
#include "cuda_kernels/common.cu"


/*
Calculates the complex valued evolution operator
*/
extern "C" __global__
void hadamard_operator(
    complex<double>* __restrict__ operator_matrix,
    int n
)
{
    uint2 pos = get_matrix_coords_2d();
    unsigned int idx = get_array_index_2d();
    operator_matrix[idx] = 1.0 / (double)pow(2.0, (double)n / 2.0) * (double)pow(-1.0, (double)(pos.x * pos.y));
}

