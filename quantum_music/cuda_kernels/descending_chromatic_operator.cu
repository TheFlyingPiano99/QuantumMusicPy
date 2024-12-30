#include <cupy/complex.cuh>
#include "cuda_kernels/common.cu"


/*
Calculates the complex valued evolution operator
*/
extern "C" __global__
void descending_chromatic_operator(
    complex<double>* __restrict__ operator_matrix,
    double phase,
    unsigned int used_pitch_count,
    unsigned int used_length_count,
    unsigned int look_back_steps
)
{
    const complex<double> val = {cos(phase), sin(phase)};
    unsigned int N = gridDim.x * blockDim.x;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int current_note_pitch = c % (used_pitch_count + 1);
    int current_note_length = (c / (used_pitch_count + 1)) % used_length_count;

    if (current_note_pitch == used_pitch_count) {   // It's a rest
        int r = c;  // Identity transform
        unsigned int idx = get_array_index(uint2{(unsigned int)r, (unsigned int)c}, uint2{N, N});
        operator_matrix[idx] = val;
        return;
    }

    int offset = 0;
    for (int i = 0; i < look_back_steps; i++) {
        int divider = ((int)pow((double)(used_pitch_count + 1), (int)look_back_steps - i) * used_length_count);
        int prev_note_pitch = (c / divider) % (int)(used_pitch_count + 1);
        if (prev_note_pitch == used_pitch_count) {  // It's a rest
            int r = c;  // Identity transform
            unsigned int idx = get_array_index(uint2{(unsigned int)r, (unsigned int)c}, uint2{N, N});
            operator_matrix[idx] = val;
            return;
        }
        if (current_note_pitch + look_back_steps - i < used_pitch_count
            && prev_note_pitch != current_note_pitch + look_back_steps - i) {
            int r = c;  // Identity transform
            unsigned int idx = get_array_index(uint2{(unsigned int)r, (unsigned int)c}, uint2{N, N});
            operator_matrix[idx] = val;
            return;
        }
        else if (current_note_pitch + look_back_steps - i >= used_pitch_count
            && prev_note_pitch != (current_note_pitch + look_back_steps - i) % used_pitch_count) {
            int r = c;  // Identity transform
            unsigned int idx = get_array_index(uint2{(unsigned int)r, (unsigned int)c}, uint2{N, N});
            operator_matrix[idx] = val;
            return;
        }

        if (prev_note_pitch > 0) {
            offset += (prev_note_pitch - 1) * divider;
        }
        else {
            offset += (used_pitch_count - 1) * divider; // Loop-back
    }
    }

    if (current_note_pitch > 0) {
        offset += current_note_pitch - 1 + current_note_length * (used_pitch_count + 1);
    }
    else {
        offset += used_pitch_count - 1 + current_note_length * (used_pitch_count + 1);  // Loop-back
    }

    int idx = get_array_index(uint2{(unsigned int)offset, (unsigned int)c}, uint2{N, N});
    operator_matrix[idx] = val;
}

