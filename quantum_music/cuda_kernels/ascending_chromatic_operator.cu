#include <cupy/complex.cuh>
#include "cuda_kernels/common.cu"


/*
Calculates the complex valued evolution operator
*/
extern "C" __global__
void ascending_chromatic_operator(
    complex<double>* __restrict__ operator_matrix,
    double phase,
    unsigned int used_pitch_count
)
{
/*
        Original Python code:
        val = math.cos(phase) + 1j * math.sin(phase)
        for c in range(N):
            r = c + 1  # Transition to the next chromatic note
            if c > 0 and (c + 1) % (used_pitch_count + 1) == 0:  # Stay in rest
                r = c
            elif c > 0 and (c + 2) % (used_pitch_count + 1) == 0:  # Transition from B to C of the same length
                r -= used_pitch_count
            self.__evolution_operator[r][c] = val
*/
    const complex<double> val = {cos(phase), sin(phase)};
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
    else if ((pos.y + 1) % (used_pitch_count + 1) == 0) {    // Stay on rest
        if (pos.x == pos.y) {
            operator_matrix[idx] = val;
        }
    }
}

