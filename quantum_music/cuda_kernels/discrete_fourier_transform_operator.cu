#include <cupy/complex.cuh>
#include "cuda_kernels/common.cu"


/*
Calculates the complex valued evolution operator
*/
extern "C" __global__
void discrete_fourier_transform_operator(
    complex<double>* __restrict__ operator_matrix
)
{
    double N = (double)(gridDim.x * blockDim.x);
    uint2 pos = get_matrix_coords_2d();
    double angle = -2.0 * M_PI / N * (double)(pos.x * pos.y);
    unsigned int idx = get_array_index_2d();
    operator_matrix[idx] = exp_i(angle) / sqrt(N);
}

