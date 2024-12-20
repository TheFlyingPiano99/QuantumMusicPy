import numpy as np
import math


one_per_sqrt_2 = 1 / math.sqrt(2)

# Gram-Schmidt orthonormalization
def gs_orthonormalization(matrix :np.ndarray):
    for i in range(0, matrix.shape[0]):
        sum = np.zeros(shape=[1, matrix.shape[1]], dtype=matrix.dtype)
        for j in range(0, i):   # Subtracting projected vectors
            sum += np.linalg.vecdot(matrix[j], matrix[i]) * matrix[j] / np.linalg.norm(matrix[j])
        matrix[i] = matrix[i] - sum
    for i in range(matrix.shape[0]):    # Normalisation
        matrix[i] = matrix[i] / np.linalg.norm(matrix[i])
    return matrix


def adjoint(m: np.ndarray):
    return np.transpose(np.conjugate(m))

def abs_squared(psi: np.ndarray):
    return float(np.vecdot(psi, psi)[:, 0].real)


def normalize_state(psi: np.ndarray):
    return psi / np.linalg.norm(psi)


def ket(elements, dtype=np.complex128):
    return np.matrix(data=[elements], dtype=dtype)


def bra(elements, dtype=np.complex128):
    return adjoint(ket(elements, dtype=dtype))


def ket_zero(dtype=np.complex128):
    return np.matrix(data=[[1, 0]], dtype=dtype)


def ket_one(dtype=np.complex128):
    return np.matrix(data=[[0, 1]], dtype=dtype)


def ket_plus(dtype=np.complex128):
    return np.matrix(data=[[one_per_sqrt_2, one_per_sqrt_2]], dtype=dtype)


def ket_minus(dtype=np.complex128):
    return np.matrix(data=[[one_per_sqrt_2, -one_per_sqrt_2]], dtype=dtype)


def hadamard(dtype=np.complex128):
    return np.matrix(
        [[one_per_sqrt_2, one_per_sqrt_2],
         [one_per_sqrt_2, -one_per_sqrt_2]], dtype=dtype)


def measurement_probabilities(state: np.ndarray, measurement_ops):
    probs = []
    for op in measurement_ops:
        M_adj_M = np.matmul(adjoint(op), op)
        temp = np.vecmat(state, M_adj_M)
        p = float(np.matvec(temp, state)[:, 0].real)
        probs.append(p)
    return probs


def proj_measurement_probabilities(state: np.ndarray, proj_measurement_ops):
    probs = []
    for op in proj_measurement_ops:
        temp = np.vecmat(state, op)
        p = float(np.matvec(temp, state)[:, 0].real)
        probs.append(p)
    return probs


def collapse_state(state: np.ndarray, measurement_op: np.ndarray):
    M_adj_M = np.matmul(adjoint(measurement_op), measurement_op)
    temp = np.vecmat(state, M_adj_M)
    p = float(np.matvec(temp, state)[:, 0].real)
    if p == 0:
        raise "Trying to collapse state to zero-probability outcome!"
    return np.matvec(measurement_op, state) / math.sqrt(p)
