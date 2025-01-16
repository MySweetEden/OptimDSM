import numpy as np
import scip

def create_matrix(size=10):
    matrix = np.random.randint(0, 2, size=(size, size))
    return matrix

def sequence(matrix):
    return scip.sequence(matrix)

def optimize(matrix, algorithm=None):
    if algorithm == None:
        return matrix
    if algorithm == "sequencing":
        return sequence(matrix)


matrix = create_matrix()
print(matrix)
print()

optimized_matrix = optimize(matrix, algorithm="sequencing")
print(optimized_matrix)