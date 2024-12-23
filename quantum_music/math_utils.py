import numpy as np
import math
from tqdm import tqdm

one_per_sqrt_2 = 1 / math.sqrt(2)

# Gram-Schmidt orthonormalization
def gs_orthonormalization(matrix :np.ndarray):
    Q, R = np.linalg.qr(matrix)
    return Q


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

def check_measurement_operators(ops) -> bool:
    sum = np.zeros(shape=ops[0].shape, dtype=ops[0].dtype)
    for op in ops:
        sum += op
    for i in range(sum.shape[0]):
        for j in range(sum.shape[1]):
            if i == j:
                if abs(sum[i][j]) < 0.99 or abs(sum[i][j]) > 1.01:
                    return False
            else:
                if abs(sum[i][j]) > 0.01:
                    return False
    return True


def normalize_probabilities(probs: list[float]) -> list[float]:
    sum = 0.0
    for i in range(len(probs)):
        sum += probs[i]
    for i in range(len(probs)):
        probs[i] /= sum
    return probs


def measurement_probabilities(state: np.ndarray, measurement_ops):
    probs = []
    for op in measurement_ops:
        M_adj_M = np.matmul(adjoint(op), op)
        temp = np.vecmat(state, M_adj_M)
        p = float(np.matvec(temp, state)[:, 0].real)
        probs.append(p)
    return probs


def proj_measurement_probabilities(state: np.ndarray, proj_measurement_base):
    probs = []
    for base_vec in tqdm(proj_measurement_base):
        op = np.outer(base_vec, adjoint(base_vec))
        temp = np.vecmat(state, op)
        p = float(np.matvec(temp, state)[:, 0].real)
        probs.append(p)
    return probs


def collapse_state(state: np.ndarray, measurement_op: np.ndarray):
    M_adj_M = np.matmul(adjoint(measurement_op), measurement_op)
    p = float(np.matvec(np.vecmat(state, M_adj_M), state)[:, 0].real)
    if p == 0:
        raise "Trying to collapse state to zero-probability outcome!"
    return np.matvec(measurement_op, state) / math.sqrt(p)


def collapse_state_using_projector(state: np.ndarray, projector: np.ndarray):
    p = float(np.matvec(np.vecmat(state, projector), state)[:, 0].real)
    if p == 0:
        raise "Trying to collapse state to zero-probability outcome!"
    return np.matvec(projector, state) / math.sqrt(p)

def evolve_state(state: np.ndarray, unitary_op: np.ndarray):
    return np.matvec(unitary_op, state)


