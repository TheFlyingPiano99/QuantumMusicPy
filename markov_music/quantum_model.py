import math
from math import prod

import numpy as np
import markov_music.math_utils as math_utils

class QuantumModel:
    __matrix: np.ndarray

    def __init__(self):
        self.__matrix = np.zeros(shape=[720, 720], dtype=np.complex128)

        testMat: np.ndarray = np.matrix([
                [1/math.sqrt(2), 1/math.sqrt(2), 0.0],
                [0.0, 1/math.sqrt(2), 1/math.sqrt(2)],
                [1/math.sqrt(2), 0.0, 1/math.sqrt(2)]])
        testMat = math_utils.gs_orthonormalization(testMat)
        testMatTransp = np.transpose(np.conjugate(testMat))
        print("After norm and self inversion:")
        print(math_utils.gs_orthonormalization(np.matmul(testMatTransp, testMat)))

        hadamard = np.matrix(
            [[np.complex128(1 / math.sqrt(2), 0), np.complex64(1 / math.sqrt(2), 0)],
             [np.complex128(1 / math.sqrt(2), 0), np.complex64(-1 / math.sqrt(2), 0)]])
        hadamard = math_utils.gs_orthonormalization(hadamard)
        invH = np.transpose(np.conjugate(hadamard))
        print(np.matmul(invH, hadamard))


        I = np.identity(10)
        I[5][5] = 0.1
        I[5][4] = 1
        math_utils.gs_orthonormalization(I)

        ket_zero = np.zeros(shape=[1, 2], dtype=np.complex128)
        ket_zero[0][0] = 1
        ket_one = np.zeros(shape=[1, 2], dtype=np.complex128)
        ket_one[0][1] = 1
        print(np.kron(ket_zero, ket_one))   # Kronecker product of two quantum states

        print(np.outer(ket_zero, ket_one))

        # Measurement:
        proj0 = np.outer(ket_zero, ket_zero)
        proj1 = np.outer(ket_one, ket_one)

        ops = [proj0, proj1]
        probs = math_utils.measurement_probabilities(ket_zero, ops)
        print(probs)

        ket_plus = math_utils.ket_plus()
        ket_min = math_utils.ket_minus()
        proj0 = np.outer(ket_plus, ket_plus)
        proj1 = np.outer(ket_min, ket_min)

        ops = [proj0, proj1]
        probs = math_utils.measurement_probabilities(ket_zero, ops)
        print(probs)

        identity = np.identity(2)
        psi = math_utils.ket([0.5, 0.5 + 1j])
        psi /= np.linalg.norm(psi)
        print(f"Abs squared of psi: {math_utils.abs_squared(psi)}")

        probs = math_utils.measurement_probabilities(ket_zero, ops)
        print(f"Probabilities: {probs}")
        print(f"Collapsed state: {math_utils.collapse_state(ket_zero, ops[0])}")

        # Create base:
        base_states = []
        N = 720
        for i in range(N):
            psi = np.zeros(shape=[1, N], dtype=np.complex128)
            psi[0][i] = 1j
            base_states.append(psi)
        ops = math_utils.create_projectors(base_states)
        print(f"Measurement system health is {'good' if math_utils.check_measurement_operators(ops) else 'bad'}.")

        probs = math_utils.proj_measurement_probabilities(base_states[0], ops)
        print(probs)

