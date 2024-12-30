#ifndef COMMON_CU
#define COMMON_CU

#include <cupy/complex.cuh>



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

#endif // COMMON_CU