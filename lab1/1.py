import numpy as np


def modified_gram_schmidt_qr(A):
    A = np.array(A, dtype=float)
    m, n = A.shape

    if m < n:
        raise ValueError("Для QR-разложения методом МНК требуется m >= n")

    V = A.copy()
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for i in range(n):
        norm_vi = np.linalg.norm(V[:, i])
        if norm_vi < 1e-14:
            raise ValueError(
                "Столбцы матрицы линейно зависимы или почти линейно зависимы"
            )

        R[i, i] = norm_vi
        Q[:, i] = V[:, i] / R[i, i]

        for j in range(i + 1, n):
            # np.dot(Q[:, i], V[:, j])
            s = 0
            for k in range(len(Q[:, i])):
                s += Q[k, i] * V[k, j]
            R[i, j] = s  # попросили без готовых функций просто
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]

    return Q, R


def back_substitution(R, y):
    R = np.array(R, dtype=float)
    y = np.array(y, dtype=float)

    n = R.shape[0]

    if R.shape[0] != R.shape[1]:
        raise ValueError("Матрица R должна быть квадратной")

    if y.shape[0] != n:
        raise ValueError("Размеры R и y не согласованы")

    x = np.zeros(n, dtype=float)

    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += R[i, j] * x[j]

        if abs(R[i, i]) < 1e-14:
            raise ValueError("Нулевой или слишком маленький диагональный элемент в R")

        x[i] = (y[i] - s) / R[i, i]
    return x


def solve_least_squares_qr(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape

    if b.shape[0] != m:
        raise ValueError("Размер вектора b должен совпадать с числом строк матрицы A")

    Q, R = modified_gram_schmidt_qr(A)
    y = np.dot(Q.T, b) # по условию можно
    x = back_substitution(R, y)
    return Q, R, x


def print_matrix(name, M):
    print(f"{name} =")
    print(M, "\n")


def test_square_case():
    A = np.array([
        [4.0, 1.0, 2.0],
        [1.0, 3.0, 0.0],
        [2.0, 0.0, 5.0]
    ])

    b = np.array([7.0, 8.0, 3.0])

    Q, R, x_qr = solve_least_squares_qr(A, b)
    x_np = np.linalg.solve(A, b)

    print("КВАДРАТНАЯ НЕВЫРОЖДЕННАЯ МАТРИЦА")
    print_matrix("A", A)
    print_matrix("b", b)
    print_matrix("Q", Q)
    print_matrix("R", R)
    print_matrix("x (QR)", x_qr)
    print_matrix("x (numpy.linalg.solve)", x_np)

    diff_norm = np.linalg.norm(x_qr - x_np)
    print(f"Норма разности решений: {diff_norm:.12e} \n")

    reconstruction_error = np.linalg.norm(A - Q @ R)
    print(f"Норма ||A - QR||: {reconstruction_error:.12e} \n")

    orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
    print(f"Норма ||Q^T Q - I||: {orthogonality_error:.12e} \n")


def test_rectangular_case():
    A = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0]
    ])

    b = np.array([6.0, 5.0, 7.0, 10.0])

    Q, R, x_qr = solve_least_squares_qr(A, b)
    x_np, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    print("ПРЯМОУГОЛЬНАЯ МАТРИЦА (m > n)")
    print_matrix("A", A)
    print_matrix("b", b)
    print_matrix("Q", Q)
    print_matrix("R", R)
    print_matrix("x (QR)", x_qr)
    print_matrix("x (numpy.linalg.lstsq)", x_np)

    diff_norm = np.linalg.norm(x_qr - x_np)
    print(f"Норма разности решений: {diff_norm:.12e} \n")

    reconstruction_error = np.linalg.norm(A - Q @ R)
    print(f"Норма ||A - QR||: {reconstruction_error:.12e} \n")

    orthogonality_error = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
    print(f"Норма ||Q^T Q - I||: {orthogonality_error:.12e} \n")

    residual_norm = np.linalg.norm(A @ x_qr - b)
    print(f"Норма невязки ||Ax - b||: {residual_norm:.12e} \n")


def main():
    test_square_case()
    test_rectangular_case()


if __name__ == "__main__":
    main()
