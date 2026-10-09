
import numpy as np


# ============================================================
# 1. Gaussian Elimination
# ============================================================

def gauss_elimination(A, b, tolerance=1e-12):
    """
    Solve A * x = b using Gaussian Elimination
    with partial pivoting and back substitution.

    Returns:
        x: solution vector, or None if the matrix is singular.
    """

    # Work on copies to preserve the original inputs
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]

    if b.ndim != 1 or len(b) != n:
        raise ValueError("b must have one entry per row of A.")

    for i in range(n):

        print(f"\nMatrix A({i}):")
        print(A)
        print(f"Vector b({i}):")
        print(b)

        # Partial pivoting: select the largest absolute pivot
        pivot_row = i + np.argmax(np.abs(A[i:, i]))

        if abs(A[pivot_row, i]) < tolerance:
            print("\nSingular matrix detected. No unique solution.")
            return None

        # Swap rows when necessary
        if pivot_row != i:
            A[[i, pivot_row]] = A[[pivot_row, i]]
            b[[i, pivot_row]] = b[[pivot_row, i]]
            print(f"Rows {i} and {pivot_row} swapped.")

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]

            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

        print("\nAfter elimination:")
        print(A)
        print("Updated b:")
        print(b)

    # Display the upper triangular system
    print("\n========== UPPER TRIANGULAR SYSTEM ==========")

    for i in range(n):
        terms = [
            f"({A[i, j]:.2f})x{j + 1}"
            for j in range(n)
            if abs(A[i, j]) > tolerance
        ]

        print(" + ".join(terms), f"= {b[i]:.2f}")

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        sum_values = np.dot(A[i, i + 1:], x[i + 1:])
        x[i] = (b[i] - sum_values) / A[i, i]

    return x


# ============================================================
# 2. Main Program
# ============================================================

if __name__ == "__main__":

    # Define the coefficient matrix and right-hand-side vector
    A = [
        [2, -1, 4, 0],
        [4, -1, 5, 1],
        [-2, 2, -2, 3],
        [0, 3, -9, 4]
    ]

    b = [1, 0, 0, 0]

    print("========== GAUSSIAN ELIMINATION ==========")

    # Solve using our implementation
    solution_gauss = gauss_elimination(A, b)

    # Solve using NumPy for verification
    A_np = np.array(A, dtype=float)
    b_np = np.array(b, dtype=float)

    try:
        solution_np = np.linalg.solve(A_np, b_np)

    except np.linalg.LinAlgError:
        solution_np = None
        print("\nNumPy could not find a unique solution.")

    # Display results
    print("\n========== RESULTS ==========")

    if solution_gauss is not None:
        print(
            "Gaussian Elimination:",
            np.round(solution_gauss, 6)
        )
    else:
        print("Gaussian Elimination: No unique solution found.")

    if solution_np is not None:
        print(
            "NumPy solution:",
            np.round(solution_np, 6)
        )
    else:
        print("NumPy: No unique solution found.")

    # Compare both solutions
    if solution_gauss is not None and solution_np is not None:

        difference = np.abs(solution_gauss - solution_np)

        print("\nAbsolute differences:")
        print(difference)

        print("\nMaximum absolute difference:")
        print(np.max(difference))

        if np.allclose(
            solution_gauss,
            solution_np,
            atol=1e-6,
            rtol=1e-6
        ):
            print("\nBoth solutions are numerically consistent.")
        else:
            print("\nThe solutions differ beyond the specified tolerance.")

        # Check the residual A*x - b
        residual = A_np @ solution_gauss - b_np

        print("\nResidual A*x - b:")
        print(residual)
        print("Residual norm:", np.linalg.norm(residual))
