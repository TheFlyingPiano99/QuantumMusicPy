import math
import pathlib
from math import prod

import cupy as cp
import numpy as np
import quantum_music.math_utils as math_utils
import quantum_music.music_layer as music
from tqdm import tqdm
from pathlib import Path
import quantum_music.cuda_utils as cuda_utils
import os

from quantum_music import music_layer

note2idx_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}  # Maps notes to indexes
length2idx_dict = {0.5: 0, 1: 1, 1.5: 2, 2: 3, 3: 4, 4: 5}  # Maps note lengths to indexes
idx2note_dict = {v: k for k, v in note2idx_dict.items()}  # Maps indexes to notes
idx2length_dict = {v: k for k, v in length2idx_dict.items()}  # Maps indexes to note lengths
used_pitch_count = len(note2idx_dict)
used_length_count = len(length2idx_dict)


class QuantumModel:
    __evolution_operator: cp.ndarray
    __state: cp.ndarray
    __measurement_base: list[cp.ndarray]
    __random_gen: np.random.Generator
    __look_back_steps: int  # The current state is product state of the current and previous notes
    __look_back_note_length: bool
    __harmony_probability_threshold: float
    # Optional cached kernels:
    __ascending_chromatic_operator_kernel: cuda_utils.KernelWithSize
    __descending_chromatic_operator_kernel: cuda_utils.KernelWithSize
    __bidirectional_chromatic_operator_kernel: cuda_utils.KernelWithSize
    __hadamard_operator_kernel: cuda_utils.KernelWithSize
    __fdt_operator_kernel: cuda_utils.KernelWithSize

    def __init__(self, look_back_steps: int = 0, look_back_note_length: bool = False):
        self.__random_gen = np.random.default_rng()
        self.__look_back_steps = look_back_steps
        self.__look_back_note_length = look_back_note_length
        self.__harmony_probability_threshold = 0.001

    def state_dimensionality(self):
        return (
            int(math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps + 1))
            if self.__look_back_note_length
            else int(math.pow((used_pitch_count + 1), self.__look_back_steps + 1) * used_length_count)
        )

    def notes2idx(self, notes: list[music.Note]) -> int:
        idx = 0
        if self.__look_back_note_length:
            for step, note in enumerate(notes):
                idx += ((
                                (used_pitch_count if note.is_rest else note2idx_dict[note.note % 12])
                                + (used_pitch_count + 1) * length2idx_dict[note.length_beats]  # Length for each note
                        ) * int(math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps - step)))
        else:  # Don't look back at note lengths
            for step, note in enumerate(notes):
                idx += (
                        (used_pitch_count if note.is_rest else note2idx_dict[note.note % 12])
                        * int(math.pow(used_pitch_count + 1, self.__look_back_steps - step)
                              * (used_length_count if step < self.__look_back_steps else 1))
                )
            # Only account for the length of the latest note:
            idx += (used_pitch_count + 1) * length2idx_dict[notes[-1].length_beats]
            if idx >= self.state_dimensionality():
                raise RuntimeError("Indexing error")
        return idx

    def idx2notes(self, idx: int) -> list[music.Note]:
        notes = []
        for step in range(
                self.__look_back_steps + 1):  # start from the oldest look-back note, which is the outer grouping
            if self.__look_back_note_length:
                denominator = int(math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps - step))
            else:
                denominator = int(math.pow(used_pitch_count + 1, self.__look_back_steps - step) * (used_length_count if step < self.__look_back_steps else 1))
            current_note_idx = idx // denominator
            length_idx = 0
            if self.__look_back_note_length:
                length_idx = current_note_idx // (used_pitch_count + 1)
            elif step == self.__look_back_steps:
                length_idx = current_note_idx // (used_pitch_count + 1)
            pitch_idx = current_note_idx % (used_pitch_count + 1)
            idx %= denominator  # Peel off the outer grouping to access the inner groupings
            if used_pitch_count == pitch_idx:  # The note is a rest
                notes.append(music.Note(
                    note=0,
                    length_beats=idx2length_dict[length_idx],
                    is_rest=True
                ))
            else:
                notes.append(music.Note(idx2note_dict[pitch_idx], idx2length_dict[length_idx]))
        return notes

    def __recursive_gather_look_back_note_indices_for_note(self, step: int, offset: int) -> list[int]:
        indices = []
        if step == self.__look_back_steps:  # Halting condition (Arrived to the innermost grouping)
            return [offset]
        for p in range(used_pitch_count + 1):
            if self.__look_back_note_length:
                for l in range(used_length_count):
                    local_offset = (p + (used_pitch_count + 1) * l) * int(
                        math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps - step)
                    )
                    indices += self.__recursive_gather_look_back_note_indices_for_note(
                        step + 1, offset + local_offset)
            else:  # No note-length look-back
                local_offset = p * int(
                    math.pow((used_pitch_count + 1), self.__look_back_steps - step) * used_length_count)
                indices += self.__recursive_gather_look_back_note_indices_for_note(
                    step + 1, offset + local_offset)
        return indices

    def gather_indices_for_current_note(self, note: music.Note) -> list[int]:
        indices = []
        if note.is_rest:
            current_pitch_offset = used_pitch_count
        else:
            current_pitch_offset = note2idx_dict[note.note % 12]
        current_length_idx = length2idx_dict[note.length_beats]
        length_offset = current_length_idx * (used_pitch_count + 1)
        offset = current_pitch_offset + length_offset
        indices += self.__recursive_gather_look_back_note_indices_for_note(0, offset)
        return indices

    def build_operator_from_notes(self, notes: list[music.Note]):
        N = self.state_dimensionality()
        transition_weight = 100.0
        longest_length = 0.0
        for i in range(used_length_count):
            if longest_length < idx2length_dict[i]:
                longest_length = idx2length_dict[i]
        for note in notes:
            if note.length_beats > longest_length:
                note.length_beats = longest_length
        print(f'Building {N}x{N} dimensional evolution operator.')
        self.__evolution_operator = cp.identity(N, dtype=cp.complex128)
        index_of_starting_state: int = 0
        for i in tqdm(range(len(notes) - 1)):  # Stop before the last note because the last is not transitioning
            from_notes = []
            for j in range(i - self.__look_back_steps, i + 1):  # Sliding windows of last notes
                if j < 0:
                    from_notes.append(self.placeholder_rest())  # Rest at the start
                else:
                    from_notes.append(notes[j])
            if 0 == i:
                index_of_starting_state = self.notes2idx(from_notes)
            to_notes = []
            for j in range(i - self.__look_back_steps + 1,
                           i + 2):  # Sliding windows of last notes including the next note
                if j < 0:
                    to_notes.append(self.placeholder_rest())  # Rest at the start
                else:
                    to_notes.append(notes[j])
            column_idx = self.notes2idx(from_notes)
            row_idx = self.notes2idx(to_notes)
            self.__evolution_operator[row_idx][column_idx] += transition_weight
        for i in range(N):  # Make connection from each note to the start of the song
            self.__evolution_operator[index_of_starting_state][i] += transition_weight
        print(f'Determinant before orthonormalization: {cp.linalg.det(self.__evolution_operator)}')
        self.__evolution_operator = math_utils.gs_orthonormalization(self.__evolution_operator).astype(cp.complex128)
        print(f'Determinant: {cp.linalg.det(self.__evolution_operator)}')

    def build_ascending_chromatic_scale_operator(self, phase: float = 0.0):
        N = self.state_dimensionality()
        self.__evolution_operator = cp.zeros(shape=[N, N], dtype=cp.complex128)

        if not hasattr(self, '__ascending_chromatic_operator_kernel'):  # Init CUDA kernel
            kernel_source = Path("quantum_music/cuda_kernels/ascending_chromatic_operator.cu").read_text()
            func_name = 'ascending_chromatic_operator'
            self.__ascending_chromatic_operator_kernel = cuda_utils.KernelWithSize()
            self.__ascending_chromatic_operator_kernel.kernel = cp.RawModule(
                code=kernel_source,
                name_expressions=[func_name],
                options=("-std=c++20", f"-I{os.path.abspath('quantum_music')}")
            ).get_function(func_name)
            (self.__ascending_chromatic_operator_kernel.grid_size,
             self.__ascending_chromatic_operator_kernel.block_size) = cuda_utils.get_grid_size_block_size(
                shape=[self.state_dimensionality(), 1],
                reduced_thread_count=False
            )

        self.__ascending_chromatic_operator_kernel.kernel(
            self.__ascending_chromatic_operator_kernel.grid_size,
            self.__ascending_chromatic_operator_kernel.block_size,
            (
                self.__evolution_operator,
                cp.float64(phase),
                cp.uint32(used_pitch_count),
                cp.uint32(used_length_count),
                cp.uint32(self.__look_back_steps),
            )
        )

    def build_descending_chromatic_scale_operator(self, phase: float = 0.0):
        N = self.state_dimensionality()
        self.__evolution_operator = cp.zeros(shape=[N, N], dtype=cp.complex128)

        if not hasattr(self, '__descending_chromatic_operator_kernel'):  # Init CUDA kernel
            kernel_source = Path("quantum_music/cuda_kernels/descending_chromatic_operator.cu").read_text()
            func_name = 'descending_chromatic_operator'
            self.__descending_chromatic_operator_kernel = cuda_utils.KernelWithSize()
            self.__descending_chromatic_operator_kernel.kernel = cp.RawModule(
                code=kernel_source,
                name_expressions=[func_name],
                options=("-std=c++20", f"-I{os.path.abspath('quantum_music')}")
            ).get_function(func_name)
            (self.__descending_chromatic_operator_kernel.grid_size,
             self.__descending_chromatic_operator_kernel.block_size) = cuda_utils.get_grid_size_block_size(
                shape=[N, 1],
                reduced_thread_count=False
            )

        self.__descending_chromatic_operator_kernel.kernel(
            self.__descending_chromatic_operator_kernel.grid_size,
            self.__descending_chromatic_operator_kernel.block_size,
            (
                self.__evolution_operator,
                cp.float64(phase),
                cp.uint32(used_pitch_count),
                cp.uint32(used_length_count),
                cp.uint32(self.__look_back_steps),
            )
        )

    def build_bidirectional_chromatic_scale_operator(self, phase: float = 0.0):
        self.build_ascending_chromatic_scale_operator(phase)
        asc_op = self.__evolution_operator.copy()
        self.build_descending_chromatic_scale_operator(phase)
        desc_op = self.__evolution_operator
        self.__evolution_operator = cp.matmul(asc_op, desc_op)

    def build_ascending_major_scale_operator(self, root_note: int = 0, phase: float = 0.0):
        root_note %= 12  # Fix out-of-range values
        degrees = [0, 2, 4, 5, 7, 9, 11]  # scale degrees and rest
        for i in range(7):
            degrees[i] = (degrees[i] + root_note) % 12
        N = self.state_dimensionality()
        markov_mtx = cp.zeros(shape=[N, N], dtype=cp.complex128)
        val = math.cos(phase) + 1j * math.sin(phase)
        for c in range(N):
            note = c % (used_pitch_count + 1)
            next_note = note
            for i in range(len(degrees)):
                if note == degrees[i]:
                    next_note = degrees[0] if i == len(degrees) - 1 else degrees[i + 1]
                    break
            r = (c // (used_pitch_count + 1)) * (used_pitch_count + 1) + next_note
            markov_mtx[r][c] = val
        print(f'Determinant of the operator: {cp.linalg.det(markov_mtx)}')
        self.__evolution_operator = markov_mtx

    def build_descending_major_scale_operator(self, root_note: int = 0, phase: float = 0.0):
        root_note %= 12  # Fix out-of-range values
        degrees = [0, 2, 4, 5, 7, 9, 11]  # scale degrees and rest
        for i in range(7):
            degrees[i] = (degrees[i] + root_note) % 12
        N = self.state_dimensionality()
        markov_mtx = cp.zeros(shape=[N, N], dtype=cp.complex128)
        val = math.cos(phase) + 1j * math.sin(phase)
        for c in range(N):
            note = c % (used_pitch_count + 1)
            next_note = note
            for i in range(len(degrees)):
                if note == degrees[i]:
                    next_note = degrees[len(degrees) - 1] if i == 0 else degrees[i - 1]
                    break
            r = (c // (used_pitch_count + 1)) * (used_pitch_count + 1) + next_note
            markov_mtx[r][c] = val
        print(f'Determinant of the operator: {cp.linalg.det(markov_mtx)}')
        self.__evolution_operator = markov_mtx

    def build_bidirectional_major_scale_operator(self, root_note: int = 0, phase: float = 0.0):
        print('Building bidirectionala major scale operator.')
        root_note %= 12  # Fix out-of-range values
        degrees = [0, 2, 4, 5, 7, 9, 11]  # scale degrees and rest
        for i in range(7):
            degrees[i] = (degrees[i] + root_note) % 12
        N = self.state_dimensionality()
        markov_mtx = cp.zeros(shape=[N, N], dtype=cp.complex128)
        val_divided = (math.cos(phase) + 1j * math.sin(phase)) * math_utils.one_per_sqrt_2
        val_unit = math.cos(phase) + 1j * math.sin(phase)
        for c in range(N):
            note = c % (used_pitch_count + 1)
            next_note0 = note
            next_note1 = note
            for i in range(len(degrees)):
                if note == degrees[i]:
                    next_note0 = degrees[len(degrees) - 1] if i == 0 else degrees[i - 1]
                    next_note1 = degrees[0] if i == len(degrees) - 1 else degrees[i + 1]
                    break
            r0 = (c // (used_pitch_count + 1)) * (used_pitch_count + 1) + next_note0
            r1 = (c // (used_pitch_count + 1)) * (used_pitch_count + 1) + next_note1
            if r0 == r1:
                markov_mtx[r0][c] = val_unit
            else:
                markov_mtx[r0][c] = val_divided
                markov_mtx[r1][c] = val_divided
        markov_mtx = math_utils.gs_orthonormalization(markov_mtx)
        print(f'Determinant of the operator: {cp.linalg.det(markov_mtx)}')
        self.__evolution_operator = markov_mtx

    def build_hadamard_operator(self):
        N = self.state_dimensionality()
        if math.log2(N) % 1 > 1e-10:
            raise RuntimeError('For the Hadamard gate, the dimensionality of the state vector must be a power of 2.')
        self.__evolution_operator = cp.zeros(shape=[N, N], dtype=cp.complex128)

        if not hasattr(self, '__hadamard_operator_kernel'):  # Init CUDA kernel
            kernel_source = Path("quantum_music/cuda_kernels/hadamard_operator.cu").read_text()
            func_name = 'hadamard_operator'
            self.__hadamard_operator_kernel = cuda_utils.KernelWithSize()
            self.__hadamard_operator_kernel.kernel = cp.RawModule(
                code=kernel_source,
                name_expressions=[func_name],
                options=("-std=c++20", f"-I{os.path.abspath('quantum_music')}")
            ).get_function(func_name)
            (self.__hadamard_operator_kernel.grid_size,
             self.__hadamard_operator_kernel.block_size) = cuda_utils.get_grid_size_block_size(
                shape=[N, N],
                reduced_thread_count=False
            )
        n = math.log2(N)    # Supposed number of qubits
        self.__hadamard_operator_kernel.kernel(
            self.__hadamard_operator_kernel.grid_size,
            self.__hadamard_operator_kernel.block_size,
            (
                self.__evolution_operator,
                cp.int32(n)
            )
        )

    def build_discrete_fourier_transform_operator(self):
        N = self.state_dimensionality()
        self.__evolution_operator = cp.zeros(shape=[N, N], dtype=cp.complex128)

        if not hasattr(self, '__fdt_operator_kernel'):  # Init CUDA kernel
            kernel_source = Path("quantum_music/cuda_kernels/discrete_fourier_transform_operator.cu").read_text()
            func_name = 'discrete_fourier_transform_operator'
            self.__fdt_operator_kernel = cuda_utils.KernelWithSize()
            self.__fdt_operator_kernel.kernel = cp.RawModule(
                code=kernel_source,
                name_expressions=[func_name],
                options=("-std=c++20", f"-I{os.path.abspath('quantum_music')}")
            ).get_function(func_name)
            (self.__fdt_operator_kernel.grid_size,
             self.__fdt_operator_kernel.block_size) = cuda_utils.get_grid_size_block_size(
                shape=[N, N],
                reduced_thread_count=False
            )
        self.__fdt_operator_kernel.kernel(
            self.__fdt_operator_kernel.grid_size,
            self.__fdt_operator_kernel.block_size,
            (
                self.__evolution_operator,
            )
        )

    def calculate_eigenstate(self):
        eigen_values, eigen_vectors = cp.linalg.eigh(self.__evolution_operator)
        print('Eigenvalues:')
        print(eigen_values)
        print('Eigenstates:')
        print(eigen_vectors)
        return eigen_vectors

    def placeholder_rest(self):
        return music.Note(0, 1, is_rest=True)

    def init_classical_state(self, notes: list[music.Note], phase: float = 0.0):
        if len(notes) > 1 + self.__look_back_steps:
            raise RuntimeError('The provided number of notes is greater than the look-back step count + 1')
        for i in range(len(notes) - self.__look_back_steps - 1):
            notes.insert(0, self.placeholder_rest())
        N = self.state_dimensionality()
        self.__state = cp.zeros(shape=[1, N], dtype=cp.complex128)
        self.__state[0][self.notes2idx(notes)] = math.cos(phase) + 1j * math.sin(phase)

    def init_superposition_state(self, superposed_notes: list[list[music.Note]], coefficients: list[complex]):
        prob_sum = 0.0
        for a in coefficients:
            prob_sum += a.conjugate() * a
        if abs(prob_sum - 1.0) > 1e-10:
            raise RuntimeError(f'Probability sum P = {prob_sum} is not 1')
        if len(superposed_notes) != len(coefficients):
            raise RuntimeError(f'Number of superposed_notes and coefficients is not equal')
        N = self.state_dimensionality()
        self.__state = np.zeros(shape=[1, N], dtype=cp.complex128)
        for i, superposed in enumerate(superposed_notes):
            if len(superposed) > 1 + self.__look_back_steps:
                raise RuntimeError('The provided number of notes is greater than the look-back step count + 1')
            for j in range(len(superposed) - self.__look_back_steps - 1):   # Add rests
                superposed.insert(0, self.placeholder_rest())
            self.__state[0][self.notes2idx(superposed)] += coefficients[i]
        self.__state /= math.sqrt(len(superposed_notes))
        self.__state = cp.asarray(self.__state)

    def init_eigen_state(self, state_index):
        if state_index < 0 or state_index > self.state_dimensionality():
            raise ValueError('state_index must be between 0 and state_dimensionality')
        eig_states = self.calculate_eigenstate()
        self.__state = cp.reshape(eig_states[state_index], [1, self.state_dimensionality()]).astype(cp.complex128)

    def init_state_as_base_state(self, base_vec_index: int = 0):
        self.__state = self.__measurement_base[base_vec_index]

    def init_state_as_superposition_of_base_states(self, base_vec_indices: list[int] = [0], phase: float = 0.0):
        multiplier = (math.cos(phase) + 1j * math.sin(phase)) / math.sqrt(len(base_vec_indices))
        self.__state = cp.zeros(shape=[1, self.state_dimensionality()], dtype=cp.complex128)
        for i in base_vec_indices:
            self.__state += multiplier * self.__measurement_base[i]

    def init_measurement_base(self, phase: float = 0.0):
        self.__measurement_base = []
        print('Initializing projective measurement base.')
        c_value = math.cos(phase) + 1j * math.sin(phase)
        base_dim = (used_pitch_count + 1) * used_length_count
        for idx in range(base_dim):
            base_vec = np.zeros(shape=[1, base_dim], dtype=cp.complex128)
            base_vec[0, idx] = c_value
            self.__measurement_base.append(base_vec)

    def evolve_state(self, iteration_count: int = 1):
        for i in range(iteration_count):
            #np_state0 = cp.asnumpy(self.__state)
            self.__state = cp.matmul(self.__evolution_operator, self.__state.T).T
            self.__state /= cp.linalg.norm(self.__state)  # Extra normalization to prevent numeric error build-up

    def _idx_of_max_probability(self, probs: list[float]) -> int:
        current_max = 0.0
        max_idx = 0
        for i in range(len(probs)):
            if probs[i] > current_max:
                current_max = probs[i]
                max_idx = i
        return max_idx

    def note2idx_in_integrated(self, note: music.Note) -> int:
        if note.is_rest:
            pitch_idx = used_pitch_count
        else:
            pitch_idx = note2idx_dict[note.note % 12]
        return pitch_idx + (used_pitch_count + 1) * length2idx_dict[note.length_beats]

    def idx2note_in_integrated(self, idx: int) -> music.Note:
        length_idx = idx // (used_pitch_count + 1)
        pitch_idx = idx % (used_pitch_count + 1)
        if pitch_idx < used_pitch_count:
            note = music.Note(note=idx2note_dict[pitch_idx % 12], length_beats=idx2length_dict[length_idx],
                              is_rest=False)
        else:
            note = music.Note(note=0, length_beats=idx2length_dict[length_idx], is_rest=True)
        return note

    def integrated_probabilities_for_single_note(self, probs: list[float]) -> list[float]:
        integrated_probs = [0.0] * (used_pitch_count + 1) * used_length_count
        for p in range(used_pitch_count + 1):
            for l in range(used_length_count):
                current_note = music.Note(note=idx2note_dict[p] if p < used_pitch_count else 0,
                                          length_beats=idx2length_dict[l],
                                          is_rest=(p == used_pitch_count))
                merged_idx = self.note2idx_in_integrated(current_note)
                indices = self.gather_indices_for_current_note(current_note)
                for original_idx in indices:
                    integrated_probs[merged_idx] += probs[original_idx]
        return integrated_probs

    def entangled_state_from_current_note(self, current_note: music_layer.Note) -> cp.ndarray:
        N = self.state_dimensionality()
        state = cp.zeros(shape=[1, N], dtype=cp.complex128)
        indices = self.gather_indices_for_current_note(current_note)
        val = 1.0 / math.sqrt(len(indices))
        for idx in indices:
            state[0, idx] = val
        return state

    def measure_state(self, max_velocity: int = 64, superposition_voices: int = 1, collapse_state: bool = True,
                      fuzzy_measurement: bool = True) -> list[music.Note]:
        if (superposition_voices < 1):
            raise ValueError('superposition_voices can not be lower than 1')
        measurement_probs = math_utils.mixed_state_measurement_probabilities(
            density_matrix=self.density_matrix_for_current_note(),
            proj_measurement_base=self.__measurement_base
        )
        if collapse_state:
            if fuzzy_measurement:  # Pseudo-random generated result with the correct distribution
                result_state_idx = self.__random_gen.choice(
                    np.arange(0, len(measurement_probs)),
                    p=measurement_probs)
            else:  # Measurement result is the state with the highest probability
                result_state_idx = self._idx_of_max_probability(measurement_probs)
            current_note = self.idx2note_in_integrated(result_state_idx)
            self.__state = self.entangled_state_from_current_note(current_note)

        print(measurement_probs)
        superposition_harmony = []
        selected_probs = []
        selected_prob_sum = 0.0
        selected_pitches = set()
        for i in range(superposition_voices):
            idx = self._idx_of_max_probability(measurement_probs)
            p = measurement_probs[idx]
            if self.__harmony_probability_threshold > p:  # Don't add notes with small probability
                continue
            note = self.idx2note_in_integrated(idx)
            measurement_probs[idx] = 0.0  # Avoid multiple selection of the same note
            if note.note in selected_pitches:  # Skip note if note with the same pitch (different length) is already selected
                continue
            selected_prob_sum += p
            superposition_harmony.append(note)
            selected_probs.append(p)
            selected_pitches.add(note.note)
        for p in selected_probs:
            p /= selected_prob_sum
        max_prob = 0.0
        for p in selected_probs:
            if p > max_prob:
                max_prob = p
        for i, p in enumerate(selected_probs):
            superposition_harmony[i].velocity = int(p / max_prob * max_velocity)
        return superposition_harmony

    def test_indexing(self):
        print('Testing indexing')
        for i in range(self.state_dimensionality()):
            notes = self.idx2notes(i)
            after_transform = self.notes2idx(notes)
            print(f'{i} -> {after_transform}:')
            for note in notes:
                print(note)
            assert i == after_transform

        print('\nTest index gathering')
        for i in range((used_pitch_count + 1) * used_length_count):
            current_note = self.idx2note_in_integrated(i)
            print(f'Current note: {str(current_note)}')
            indices = self.gather_indices_for_current_note(current_note)
            for i in indices:
                notes = self.idx2notes(i)
                for note in notes:
                    print(note)
                print('')

    def test_measurement_base(self):
        print('Testing measurement base')
        N = self.state_dimensionality()
        print(f'Quantum state dimensionality: {N}, number of base states: {len(self.__measurement_base)}')
        sum_mtx = cp.zeros(shape=[len(self.__measurement_base), len(self.__measurement_base)], dtype=cp.complex128)
        error_margin = 0.0001
        for i, vec in enumerate(self.__measurement_base):
            print(f'Base vector #{i} has dimensions {vec.shape}')
            magnitude = math_utils.abs_squared(vec)
            print(f'Absolute value squared = {magnitude}')
            assert abs(magnitude - 1.0) < error_margin
            print('')
            sum_mtx += math_utils.outer(vec, vec)
        print('Sum of all measurement operator matrices:')
        print(sum_mtx.shape)
        print(sum_mtx)
        dots = cp.zeros(shape=[len(self.__measurement_base), len(self.__measurement_base)], dtype=cp.complex128)
        for i in range(len(self.__measurement_base)):
            for j in range(len(self.__measurement_base)):
                dots[i][j] = cp.dot(self.__measurement_base[i], math_utils.adjoint(self.__measurement_base[j]))[0]
        np_dots = cp.asnumpy(dots)
        print(np_dots)

    def __recursive_sum_density_matrix(self, step: int, offset: int) -> np.ndarray:
        if step == self.__look_back_steps:    # Halting condition
            sub_state = self.__state[:, offset: offset + (used_pitch_count + 1) * used_length_count]
            return math_utils.outer(sub_state, sub_state)   # Nx1 * 1xN
        sum = cp.zeros(
            shape=[(used_pitch_count + 1) * used_length_count, (used_pitch_count + 1) * used_length_count],
            dtype=cp.complex128
        )
        for p in range(used_pitch_count + 1):
            if self.__look_back_note_length:
                for l in range(used_length_count):
                    local_offset = offset + ((l * (used_pitch_count + 1) + p)
                               * int(math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps - step)))
                    sum += self.__recursive_sum_density_matrix(step + 1, local_offset)
            else:
                local_offset = offset + p * int(math.pow((used_pitch_count + 1), self.__look_back_steps - step)) * used_length_count
                sum += self.__recursive_sum_density_matrix(step + 1, local_offset)
        return sum

    def density_matrix_for_current_note(self) -> np.ndarray:
        return self.__recursive_sum_density_matrix(0, 0)

    def invert_evolution_opearotor(self):
        self.__evolution_operator = math_utils.adjoint(self.__evolution_operator)

    def test_density_matrix(self):
        print('Testing density matrix for current note:')
        density_matrix = self.density_matrix_for_current_note()
        print(density_matrix)
        trace = density_matrix.trace()
        np_dm = cp.asnumpy(density_matrix)
        assert abs(trace - 1.0) < 1e-5
        print(f'Trace of the density matrix: {trace}')
        print(f'Trace of the density matrix^2: {cp.matmul(density_matrix, density_matrix).trace()}')

    def transfer_measurement_base_to_gpu(self):
        for i in range(len(self.__measurement_base)):
            self.__measurement_base[i] = cp.asarray(self.__measurement_base[i])
        cp.cuda.runtime.deviceSynchronize()

    def serialise_evolution_operator(self, file_path: pathlib.Path):
        np_op = cp.asnumpy(self.__evolution_operator)
        np.save(file_path.absolute(), np_op)
        print(f'Serialised evolution operator to file: {file_path.absolute()}')

    def load_evolution_operator(self, file_path: pathlib.Path):
        np_op = np.load(file_path.absolute())
        self.__evolution_operator = cp.asarray(np_op)
        print(f'Loaded evolution operator from file: {file_path.absolute()}')
        print(f'Shape of the operator is: {self.__evolution_operator.shape}')
