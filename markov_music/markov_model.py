import numpy as np
import math

class MarkovModel:
    __matrix: np.ndarray

    # Gram-Schmidt orthonormalization
    def gs_orthonormalization(self, matrix:np.ndarray):
        for i in range(0, matrix.shape[0]):
            sum = np.zeros(shape=[1, matrix.shape[1]], dtype=matrix.dtype)
            for j in range(0, i):
                proj = np.linalg.vecdot(matrix[j], matrix[i]) * matrix[j] / np.linalg.norm(matrix[j])
                sum += proj
            matrix[i] = matrix[i] - sum
        for i in range(matrix.shape[0]):
            matrix[i] = matrix[i] / np.linalg.norm(matrix[i])
        return matrix

    def __init__(self):
        self.__matrix = np.zeros(shape=[720, 720], dtype=np.float64)

        testMat: np.ndarray = np.matrix([
                [1/math.sqrt(2), 1/math.sqrt(2), 0.0],
                [0.0, 1/math.sqrt(2), 1/math.sqrt(2)],
                [1/math.sqrt(2), 0.0, 1/math.sqrt(2)]])
        testMat = self.gs_orthonormalization(testMat)
        testMatTransp = np.transpose(np.conjugate(testMat))
        print("After norm and self inversion:")
        print(self.gs_orthonormalization(np.matmul(testMatTransp, testMat)))

        hadamard = np.matrix(
            [[np.complex64(1 / math.sqrt(2), 0), np.complex64(1 / math.sqrt(2), 0)],
             [np.complex64(1 / math.sqrt(2), 0), np.complex64(-1 / math.sqrt(2), 0)]])
        hadamard = self.gs_orthonormalization(hadamard)
        invH = np.transpose(np.conjugate(hadamard))
        print(np.matmul(invH, hadamard))


        I = np.identity(10)
        I[5][5] = 0.1
        I[5][4] = 1
        self.gs_orthonormalization(I)

