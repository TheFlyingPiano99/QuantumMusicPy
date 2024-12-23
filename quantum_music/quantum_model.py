import math
from math import prod

import numpy as np
import quantum_music.math_utils as math_utils
import quantum_music.music_layer as music
from tqdm import tqdm


note2idx_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11}  # Maps notes to indexes
length2idx_dict = {0.25: 0, 0.5: 1, 1: 2, 2: 3, 4: 4}  # Maps note lengths to indexes
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
        return int(math.pow((used_pitch_count + 1) * used_length_count, self.__look_back_steps + 1))



    def notes2idx(self, notes: list[music.Note]) -> int:
        idx = 0
        if self.__look_back_note_length:
            for step, note in enumerate(notes):
                idx += ((
                            (12 if note.is_rest else note2idx_dict[note.note % 12])
                            + (used_pitch_count + 1) * length2idx_dict[note.length_beats]   # Length for each note
                        ) * int(math.pow((used_pitch_count + 1) * used_length_count, step)))
        else:   # Don't look back at note lengths
            for step, note in enumerate(notes):
                idx += (
                            (12 if note.is_rest else note2idx_dict[note.note % 12])
                            * int(math.pow(used_pitch_count + 1, step))
                )
            # Only account for the length of the latest note:
            idx += int(math.pow(used_pitch_count + 1, len(notes)) * length2idx_dict[notes[-1].length_beats])
        return idx

    def idx2notes(self, idx: int) -> list[music.Note]:
        notes = []
        for step in range(self.__look_back_steps, -1, -1):
            denominator = int(math.pow((used_pitch_count + 1) * used_length_count, step))
            note_idx = idx // denominator
            idx = idx % denominator
            pitch_idx = note_idx % (used_pitch_count + 1)
            length_idx = note_idx // (used_pitch_count + 1)
            if 12 == pitch_idx:
                notes.append(music.Note(
                    note=0,
                    length_beats=idx2length_dict[length_idx],
                    is_rest=True
                ))
            else:
                notes.append(music.Note(idx2note_dict[pitch_idx], idx2length_dict[length_idx]))
        notes.reverse()
        return notes

    def build_operator_from_notes(self, notes: list[music.Note]):
        N = self.state_dimensionality()
        transition_weight = 10000.0
        print(f'Building {N}x{N} dimensional evolution operator')
        markov_mtx = np.identity(N, dtype=np.float64)
        for i in tqdm(range(len(notes) - 1)): # Stop before the last note because the last is not transitioning
            from_notes = []
            for j in range(i - self.__look_back_steps, i + 1): # Sliding windows of last notes
                if j < 0:
                    from_notes.append(music.Note(0, 4, is_rest=True))   # Rest at the start
                else:
                    from_notes.append(notes[j])
            to_notes = []
            for j in range(i - self.__look_back_steps + 1, i + 2): # Sliding windows of last notes including the next note
                if j < 0:
                    to_notes.append(music.Note(0, 4, is_rest=True))   # Rest at the start
                else:
                    to_notes.append(notes[j])
            row_idx = self.notes2idx(from_notes)
            column_idx = self.notes2idx(to_notes)
            markov_mtx[row_idx][column_idx] += transition_weight
        self.__evolution_operator = math_utils.gs_orthonormalization(markov_mtx).astype(np.complex128)

    def init_classical_state(self, note: music.Note, phase: float = 0.0):
        N = self.state_dimensionality()
        self.__state = np.zeros(shape=[1, N], dtype=np.complex128)
        self.__state[0][self.notes2idx([note])] = math.cos(phase) + 1j * math.sin(phase)

    def init_projective_measurement_operators(self, phase: float = 0.0):
        self.__measurement_base = []
        N = self.state_dimensionality()
        for i in tqdm(range(N)):
            base_vec = np.zeros(shape=[1, N], dtype=np.complex128)
            base_vec[0][i] = math.cos(phase) + 1j * math.sin(phase)
            self.__measurement_base.append(base_vec)


    def evolve_state(self, iteration_count: int = 1):
        for i in range(iteration_count):
            self.__state = np.matvec(self.__evolution_operator, self.__state)
            self.__state /= np.linalg.norm(self.__state)    # Extra normalization to prevent of numeric error build-up

    def _idx_of_max_probability(self):
        probs = math_utils.proj_measurement_probabilities(state=self.__state,
                                                          proj_measurement_ops=self.__measurement_base)
        current_max = 0.0
        max_idx = 0
        for i in range(self.state_dimensionality()):
            if probs[i] > current_max:
                current_max = probs[i]
                max_idx = i
        return max_idx

    def measure_state(self, collapse_state: bool = True, probabilistic: bool = True) -> list[music.Note]:
        if collapse_state:
            if probabilistic:
                probs = math_utils.proj_measurement_probabilities(state=self.__state,
                                                                  proj_measurement_ops=self.__measurement_base)
                result_state_idx = self.__random_gen.choice(np.arange(0, len(probs)), p=probs)
                self.__state = math_utils.collapse_state(self.__state, self.__measurement_base[result_state_idx])
            else:
                result_state_idx = self._idx_of_max_probability()
                self.__state = math_utils.collapse_state(self.__state, self.__measurement_base[result_state_idx])
        else:   # No collapse
            if probabilistic:
                probs = math_utils.proj_measurement_probabilities(state=self.__state,
                                                                  proj_measurement_base=self.__measurement_base)
                result_state_idx = self.__random_gen.choice(np.arange(0, len(probs)), p=probs)
            else:
                result_state_idx = self._idx_of_max_probability()
        return self.idx2notes(result_state_idx)


