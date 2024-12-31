#ifndef COMMON_CU
#define COMMON_CU

#include <cupy/complex.cuh>

constexpr double M_PI = 3.14159265358979323846264338327950288419716939937510;


__device__ uint2 get_matrix_coords_2d()
{
    return {
        blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y
    };
}


__device__ unsigned int get_array_index_2d()
{
    uint2 pixel = get_matrix_coords_2d();
    return pixel.x * gridDim.y * blockDim.y
            + pixel.y;
}

__device__ unsigned int get_array_index(const uint2& pos, const uint2& N)
{
    return pos.x * N.y + pos.y;
}

__device__ complex<double> exp_i(double angle)
{
    return complex<double>(cos(angle), sin(angle));
}


#endif // COMMON_CU