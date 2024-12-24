import math
from math import prod

import numpy as np
import quantum_music.math_utils as math_utils
import quantum_music.music_layer as music
from tqdm import tqdm


note2idx_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}  # Maps notes to indexes
length2idx_dict = {1: 0, 2: 1, 4: 2}  # Maps note lengths to indexes
idx2note_dict = {v: k for k, v in note2idx_dict.items()}  # Maps indexes to notes
idx2length_dict = {v: k for k, v in length2idx_dict.items()}  # Maps indexes to note lengths
used_pitch_count = len(note2idx_dict)
used_length_count = len(length2idx_dict)


class QuantumModel:
    __evolution_operator: np.ndarray
    __state: np.ndarray
    __measurement_base: list[np.ndarray]
    __random_gen: np.random.Generator
    __look_back_steps: int      # The current state is product state of the current and previous notes
    __look_back_note_length: bool

    def __init__(self, look_back_steps: int = 0, look_back_note_length: bool = False):
        self.__random_gen = np.random.default_rng()
        self.__look_back_steps = look_back_steps
        self.__look_back_note_length = look_back_note_length

    def state_dimensionality(self):
        return (
            int(math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps + 1))
            if self.__look_back_note_length
            else int(math.pow((used_pitch_count + 1), self.__look_back_steps + 1) * used_length_count)
        )


    def notes2idx(self, notes: list[music.Note]) -> int:
        idx = 0
        notes.reverse() # The current note is the innermost grouping in the matrices (Current-Note-Pitch-Major Order)
        if self.__look_back_note_length:
            for step, note in enumerate(notes):
                idx += ((
                            (used_pitch_count if note.is_rest else note2idx_dict[note.note % 12])
                            + (used_pitch_count + 1) * length2idx_dict[note.length_beats]   # Length for each note
                        ) * int(math.pow((used_pitch_count + 1) * used_length_count, step)))
        else:   # Don't look back at note lengths
            for step, note in enumerate(notes):
                idx += (
                            (used_pitch_count if note.is_rest else note2idx_dict[note.note % 12])
                            * int(math.pow(used_pitch_count + 1, step))
                )
            # Only account for the length of the latest note:
            # The current-note-length is the outermost grouping (The values for the same note-length are stored continuously.)
            idx += int(math.pow(used_pitch_count + 1, self.__look_back_steps + 1)) * length2idx_dict[notes[-1].length_beats]
            if idx >= self.state_dimensionality():
                raise RuntimeError("Indexing error")
        return idx

    def idx2notes(self, idx: int) -> list[music.Note]:
        notes = []
        if not self.__look_back_note_length:
            # The current-note-length is the outermost grouping (The values for the same note-length are stored continuously.)
            length_denominator = int(math.pow(used_pitch_count + 1, self.__look_back_steps + 1))
            length_idx = idx // length_denominator
            idx %= length_denominator   # Peel off the outer grouping to access the inner groupings
        for step in range(self.__look_back_steps):  # start from the oldest look-back note, which is the outer grouping
            if self.__look_back_note_length:
                denominator = int(math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps - step))
            else:
                denominator = int(math.pow(used_pitch_count + 1, self.__look_back_steps - step))
            note_idx = idx // denominator
            if self.__look_back_note_length:
                length_idx = note_idx // (used_pitch_count + 1)
                note_idx %= (used_pitch_count + 1)
            else:
                length_idx = 0  # Will be discarded
            idx %= denominator  # Peel off the outer grouping to access the inner groupings
            if used_pitch_count == note_idx:   # The note is a rest
                notes.append(music.Note(
                    note=0,
                    length_beats=idx2length_dict[length_idx],
                    is_rest=True
                ))
            else:
                notes.append(music.Note(idx2note_dict[note_idx], idx2length_dict[length_idx]))
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
            else:   # No note-length look-back
                local_offset = p * int(
                    math.pow((used_pitch_count + 1), self.__look_back_steps - step)
                )
                indices += self.__recursive_gather_look_back_note_indices_for_note(
                    step + 1, offset + local_offset)
        return indices

    def gather_indices_for_current_note(self, note: music.Note) -> list[int]:
        indices = []
        if note.note < used_pitch_count:
            current_pitch_offset = note2idx_dict[note.note % 12]
        else:   # The note is a rest
            current_pitch_offset = used_pitch_count
        current_length_idx = length2idx_dict[note.length_beats]
        if not self.__look_back_note_length:
            length_offset = current_length_idx * int(math.pow((used_pitch_count + 1), self.__look_back_steps + 1))
        else:
            length_offset = (used_pitch_count + 1)
        offset = current_pitch_offset + length_offset
        indices += self.__recursive_gather_look_back_note_indices_for_note(0, offset)
        return indices

    def build_operator_from_notes(self, notes: list[music.Note]):
        N = self.state_dimensionality()
        transition_weight = 10000.0
        print(f'Building {N}x{N} dimensional evolution operator.')
        markov_mtx = np.identity(N, dtype=np.float64)
        for i in range(len(notes) - 1): # Stop before the last note because the last is not transitioning
            from_notes = []
            for j in range(i - self.__look_back_steps, i + 1): # Sliding windows of last notes
                if j < 0:
                    from_notes.append(self.placeholder_rest())   # Rest at the start
                else:
                    from_notes.append(notes[j])
            to_notes = []
            for j in range(i - self.__look_back_steps + 1, i + 2): # Sliding windows of last notes including the next note
                if j < 0:
                    to_notes.append(self.placeholder_rest())   # Rest at the start
                else:
                    to_notes.append(notes[j])
            column_idx = self.notes2idx(from_notes)
            row_idx = self.notes2idx(to_notes)
            markov_mtx[row_idx][column_idx] += transition_weight
        self.__evolution_operator = math_utils.gs_orthonormalization(markov_mtx).astype(np.complex128)

    def build_chromatic_scale_operator(self):
        N = self.state_dimensionality()
        markov_mtx = np.zeros(shape=[N, N], dtype=np.float64)
        for r in range(N):
            c = (r + 1) % N
            markov_mtx[r][c] = 10000.0
        self.__evolution_operator = math_utils.gs_orthonormalization(markov_mtx).astype(np.complex128)


    def placeholder_rest(self):
        return music.Note(0, 1, is_rest=True)

    def init_classical_state(self, notes: list[music.Note], phase: float = 0.0):
        if len(notes) != 1 + self.__look_back_steps:
            raise RuntimeError('The provided number of notes does not correspond to the look-back step count')
        N = self.state_dimensionality()
        self.__state = np.zeros(shape=[1, N], dtype=np.complex128)
        self.__state[0][self.notes2idx(notes)] = math.cos(phase) + 1j * math.sin(phase)

    def init_state_as_base_state(self, base_vec_index: int = 0):
        self.__state = self.__measurement_base[base_vec_index]

    def init_projective_measurement_base(self, phase: float = 0.0):
        self.__measurement_base = []
        N = self.state_dimensionality()
        print('Initializing projective measurement base.')
        test_sum = np.zeros(shape=[N, N], dtype=np.complex128)
        c_value = math.cos(phase) + 1j * math.sin(phase)
        if self.__look_back_note_length:
            c_value /= math.sqrt(math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps))
        else:
            c_value /= math.sqrt(math.pow(used_pitch_count + 1, self.__look_back_steps))

        for p in range(used_pitch_count + 1):
            for l in range(used_length_count):
                base_vec = np.zeros(shape=[1, N], dtype=np.complex128)
                is_rest = (p == used_pitch_count)
                if is_rest:
                    current_note = music.Note(note=0,
                                              length_beats=idx2length_dict[l],
                                              is_rest=True)
                else:
                    current_note = music.Note(note=idx2note_dict[p],
                                              length_beats=idx2length_dict[l],
                                              is_rest=False)
                indices = self.gather_indices_for_current_note(current_note)
                for idx in indices:
                    base_vec[0][idx] = c_value
                test_sum += np.outer(base_vec, math_utils.adjoint(base_vec))
                self.__measurement_base.append(base_vec)
        print(test_sum)

    def evolve_state(self, iteration_count: int = 1):
        for i in range(iteration_count):
            self.__state = np.matvec(self.__evolution_operator, self.__state)
            self.__state /= np.linalg.norm(self.__state)    # Extra normalization to prevent numeric error build-up

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
            note_idx = used_pitch_count
        else:
            note_idx = note2idx_dict[note.note % 12]
        return note_idx + (used_pitch_count + 1) * length2idx_dict[note.length_beats]

    def idx2current_note_in_integrated(self, idx: int) -> music.Note:
        length_idx = idx // (used_pitch_count + 1)
        note_idx = idx % (used_pitch_count + 1)
        if note_idx < used_pitch_count:
            note = music.Note(note=idx2note_dict[note_idx % 12], length_beats=idx2length_dict[length_idx], is_rest=False)
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

    def measure_state(self, max_velocity: int = 64, superposition_voices: int = 1, collapse_state: bool = True, fuzzy_measurement: bool = True) -> list[music.Note]:
        if (superposition_voices < 1):
            raise ValueError('superposition_voices can not be lower than 1')
        measurement_probs = math_utils.proj_measurement_probabilities(state=self.__state,
                                                          proj_measurement_base=self.__measurement_base)
        if collapse_state:
            if fuzzy_measurement:   # Pseudo-random generated result with the correct distribution
                result_state_idx = self.__random_gen.choice(np.arange(0, len(measurement_probs)), p=measurement_probs)
            else: # Measurement result is the state with the highest probability
                result_state_idx = self._idx_of_max_probability(measurement_probs)
            self.__state = math_utils.collapse_state_using_projector(self.__state, self.__measurement_base[result_state_idx])

        print(measurement_probs)
        superposition_harmony = []
        selected_probs = []
        selected_prob_sum = 0.0
        for i in range(superposition_voices):
            idx = self._idx_of_max_probability(measurement_probs)
            p = measurement_probs[idx]
            selected_prob_sum += p
            measurement_probs[idx] = 0.0    # Avoid multiple selection of the same note
            note = self.idx2current_note_in_integrated(idx)
            superposition_harmony.append(note)
            selected_probs.append(p)
        for i in range(superposition_voices):
            selected_probs[i] /= selected_prob_sum
        max_prob = 0.0
        for prob in selected_probs:
            if prob > max_prob:
                max_prob = prob
        for i in range(superposition_voices):
            superposition_harmony[i].velocity = int(selected_probs[i] / max_prob * max_velocity)
        return superposition_harmony


