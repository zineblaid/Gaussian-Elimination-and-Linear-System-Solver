# Gaussian Elimination and Linear System Solver

## Overview

This project implements Gaussian Elimination in Python to solve systems of linear equations.

The algorithm transforms a system into upper triangular form using elementary row operations, then calculates the unknowns through back substitution. Its results are compared with NumPy's built-in linear system solver.

## Features

- Gaussian Elimination algorithm.
- Row swapping when a pivot is zero.
- Transformation into upper triangular form.
- Back substitution to calculate the solution.
- Display of intermediate matrices and vectors.
- Comparison with `numpy.linalg.solve`.
- Absolute error calculation between the two solutions.
- Detection of singular or non-solvable cases.

## Example System

The project solves a system of four linear equations represented by:

\[
A x = b
\]

The coefficient matrix is:

\[
A =
\begin{bmatrix}
2 & -1 & 4 & 0\\
4 & -1 & 5 & 1\\
-2 & 2 & -2 & 3\\
0 & 3 & -9 & 4
\end{bmatrix}
\]

The right-hand-side vector is:

\[
b =
\begin{bmatrix}
1\\0\\0\\0
\end{bmatrix}
\]

## Technologies

- Python 3
- NumPy

## Installation

Install NumPy:

```bash
pip install numpy
```

## How to Run

```bash
python main.py
```

## Project Structure

```text
gaussian-elimination/
├── main.py
├── README.md
└── .gitignore
```

## Learning Objectives

- Understand Gaussian Elimination.
- Implement elementary row operations.
- Apply back substitution.
- Handle zero pivots through row swapping.
- Compare numerical algorithms with a trusted library.
- Analyze floating-point differences between computed solutions.

## Conclusion

Gaussian Elimination provides a systematic way to solve linear systems. Comparing its output with `numpy.linalg.solve` helps validate the implementation and observe small floating-point rounding differences.

## License

Add a license file if you want to specify the terms under which others can use, modify, and distribute this project.
