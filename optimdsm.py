import numpy as np

import sa
import scip

np.random.seed(0)


def create_matrix(size=4):
    matrix = np.random.randint(0, 2, size=(size, size))
    return matrix


def scip_sequence(matrix):
    """
    Compute the optimal sequence of tasks using the exact sequencing technique provided by scip.

    Args:
        matrix (nd.array): n x n matrix representing the cost of performing task i after task j.

    Returns:
        nd.array: n x n matrix representing the optimal sequence of tasks.
    """

    return scip.sequence(matrix)


def scip_cluster(matrix):
    """
    Compute the optimal clustering of tasks using the exact clustering technique provided by scip.

    Args:
        matrix (nd.array): n x n matrix representing the cost of clustering task i with task j.

    Returns:
        nd.array: n x n matrix representing the optimal clustering of tasks.
    """

    return matrix


def sa_sequence(matrix):
    """
    Compute the optimal sequence of tasks using a simulated annealing technique.

    Args:
        matrix (nd.array): n x n matrix representing the cost of performing task i after task j.

    Returns:
        nd.array: n x n matrix representing the optimized sequence of tasks.
    """
    return sa.sequence(matrix)


def sa_cluster(matrix):
    """
    Compute the optimal clustering of tasks using a simulated annealing technique.

    Args:
        matrix (nd.array): n x n matrix representing the cost of clustering task i with task j.

    Returns:
        nd.array: n x n matrix representing the optimized clustering of tasks.
    """
    return matrix


def optimize(matrix, strategy=None, technique=None):
    algorithm = {
        ("sequencing", "exact"): scip_sequence,
        ("clustering", "exact"): scip_cluster,
        ("sequencing", "heuristic"): sa_sequence,
        ("clustering", "heuristic"): sa_cluster,
    }
    if strategy is None or technique is None:
        return matrix
    return algorithm.get((strategy, technique))(matrix)


matrix = create_matrix()
print(matrix)
print()

optimized_matrix = optimize(matrix, strategy="sequencing", technique="exact")
print("sequencing, exact")
print(np.sum(np.tril(optimized_matrix, k=-1)))
print(optimized_matrix)

optimized_matrix = optimize(matrix, strategy="sequencing", technique="heuristic")
print("sequencing, heuristic")
print(np.sum(np.tril(optimized_matrix, k=-1)))
print(optimized_matrix)
