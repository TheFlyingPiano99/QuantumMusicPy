#include <cupy/complex.cuh>
#include "cuda_kernels/common.cu"


/*
Calculates the complex valued evolution operator
*/
extern "C" __global__
void bidirectional_chromatic_operator(
    complex<double>* __restrict__ operator_matrix,
    double phase,
    unsigned int used_pitch_count
)
{
    const complex<double> val = {cos(phase) / sqrt(2.0), sin(phase) / sqrt(2.0)};
    unsigned int idx = get_array_index_2d();
    uint2 pos = get_matrix_coords_2d();
    operator_matrix[idx] = {0.0, 0.0};
    if ((pos.y) % (used_pitch_count + 1) < 11) {    // Transition to the next note
        if (pos.x == pos.y + 1) {
            operator_matrix[idx] = val;
        }
    }
    else if ((pos.y + 2) % (used_pitch_count + 1) == 0) {   // Loop-back from B to C
        if (pos.x == pos.y - (used_pitch_count - 1)) {
            operator_matrix[idx] = val;
        }
    }

    if (pos.y % (used_pitch_count + 1) < 12 && pos.y % (used_pitch_count + 1) > 0) {    // Transition to the previous note
        if (pos.x == pos.y - 1) {
            operator_matrix[idx] = val;
        }
    }
    else if (pos.y % (used_pitch_count + 1) == 0) {   // Loop from C to B
        if (pos.x == pos.y + (used_pitch_count - 1)) {
            operator_matrix[idx] = val;
        }
    }
    else if ((pos.y + 1) % (used_pitch_count + 1) == 0) {    // Stay on rest
        if (pos.x == pos.y) {
            operator_matrix[idx] = val * sqrt(2.0);
        }
    }
}

