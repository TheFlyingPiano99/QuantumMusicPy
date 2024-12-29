import cupy as cp
import math
from tqdm import tqdm

one_per_sqrt_2 = 1 / math.sqrt(2)

# Gram-Schmidt orthonormalization
def gs_orthonormalization(matrix :cp.ndarray):
    Q, R = cp.linalg.qr(matrix)
    return Q


def adjoint(m: cp.ndarray):
    return cp.transpose(cp.conjugate(m))

def abs_squared(psi: cp.ndarray):
    return float(cp.dot(psi, adjoint(psi))[0, 0].real)


def normalize_state(psi: cp.ndarray):
    return psi / cp.linalg.norm(psi)


def ket(elements, dtype=cp.complex128):
    return cp.matrix(data=[elements], dtype=dtype)


def bra(elements, dtype=cp.complex128):
    return adjoint(ket(elements, dtype=dtype))


def ket_zero(dtype=cp.complex128):
    return cp.matrix(data=[[1, 0]], dtype=dtype)


def ket_one(dtype=cp.complex128):
    return cp.matrix(data=[[0, 1]], dtype=dtype)


def ket_plus(dtype=cp.complex128):
    return cp.matrix(data=[[one_per_sqrt_2, one_per_sqrt_2]], dtype=dtype)


def ket_minus(dtype=cp.complex128):
    return cp.matrix(data=[[one_per_sqrt_2, -one_per_sqrt_2]], dtype=dtype)


def hadamard(dtype=cp.complex128):
    return cp.matrix(
        [[one_per_sqrt_2, one_per_sqrt_2],
         [one_per_sqrt_2, -one_per_sqrt_2]], dtype=dtype)

def check_measurement_operators(ops) -> bool:
    sum = cp.zeros(shape=ops[0].shape, dtype=ops[0].dtype)
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


def measurement_probabilities(state: cp.ndarray, measurement_ops):
    probs = []
    for op in measurement_ops:
        M_adj_M = cp.matmul(adjoint(op), op)
        temp = cp.matmul(adjoint(state).T, M_adj_M)
        p = float(cp.matmul(temp, state.T).T[:, 0].real)
        probs.append(p)
    return probs


def proj_measurement_probabilities(state: cp.ndarray, proj_measurement_base: list[cp.ndarray]):
    probs = list[float]()
    sum = 0.0
    for base_vec in proj_measurement_base:
        op = cp.outer(base_vec, adjoint(base_vec))
        temp = cp.matmul(adjoint(state).T, op)
        p = float(cp.matmul(temp, state.T).T[:, 0].real)
        sum += p
        probs.append(p)
    for i in range(len(probs)):
        probs[i] /= sum
    return probs

def probability_distribution(state: cp.ndarray):
    return cp.real(cp.multiply(adjoint(state), state))

def collapse_state(state: cp.ndarray, measurement_op: cp.ndarray):
    M_adj_M = cp.matmul(adjoint(measurement_op), measurement_op)
    p = float(cp.matvec(cp.matmul(adjoint(state).T, M_adj_M), state.T).T[:, 0].real)
    if p == 0:
        raise "Trying to collapse state to zero-probability outcome!"
    return cp.matmul(measurement_op, state.T).T / math.sqrt(p)


def collapse_state_using_projector(state: cp.ndarray, base_vec: cp.ndarray):
    projector_op = cp.outer(base_vec, adjoint(base_vec))
    p = float(cp.matmul(cp.matmul(adjoint(state).T, projector_op), state.T).T[:, 0].real)
    if p == 0:
        raise "Trying to collapse state to zero-probability outcome!"
    return cp.matmul(projector_op, state.T).T / math.sqrt(p)

def evolve_state(state: cp.ndarray, unitary_op: cp.ndarray):
    return cp.matmul(unitary_op, state.T).T


