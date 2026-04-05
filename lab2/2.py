import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


def generate_spd_system(n, seed=42):
    rng = np.random.default_rng(seed)

    S = rng.uniform(-0.05, 0.05, size=(n, n))
    S = 0.5 * (S + S.T)

    np.fill_diagonal(S, 0.0)
    row_abs_sums = np.sum(np.abs(S), axis=1)
    diag = row_abs_sums + rng.uniform(1.0, 2.0, size=n)
    A = S.copy()
    np.fill_diagonal(A, diag)

    # чтобы система была совместна
    x_true = rng.normal(size=n)
    b = A @ x_true
    return A, b


def gershgorin_bounds(A):
    diag = np.diag(A)
    r_ii = np.sum(np.abs(A), axis=1) - np.abs(diag)

    lambda_min_est = np.min(diag - r_ii)
    lambda_max_est = np.max(diag + r_ii)

    return lambda_min_est, lambda_max_est


def richardson_method(A, b, tau, max_iter=10000, tol=1e-8, x0 = None):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = A.shape[0]

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.array(x0, dtype=float).copy()

    residual_norms = []
    r = b - A @ x
    residual_norms.append(np.linalg.norm(r))
    for _ in range(max_iter):
        if residual_norms[-1] < tol:
            break

        # x_{k+1} = x_k + tau * (b - A x_k)
        x = x + tau * r
        r = b - A @ x
        residual_norms.append(np.linalg.norm(r))
    return x, np.array(residual_norms)


def run_with_timer(A, b, tau, max_iter, tol):
    start = perf_counter()
    x, residuals = richardson_method(A, b, tau, max_iter=max_iter, tol=tol)
    elapsed = perf_counter() - start
    return x, residuals, elapsed


def main():
    n = 150
    tol = 1e-10
    max_iter = 20000
    seed = 42

    A, b = generate_spd_system(n, seed=seed)
    lambda_min_est, lambda_max_est = gershgorin_bounds(A)
    eigvals = np.linalg.eigvalsh(A)
    lambda_min_exact = eigvals[0]
    lambda_max_exact = eigvals[-1]

    start = perf_counter()
    x_exact = np.linalg.solve(A, b)
    solve_time = perf_counter() - start

    tau_arbitrary = 1.0 / lambda_max_est
    tau_opt_est = 2.0 / (lambda_min_est + lambda_max_est)
    tau_opt_exact = 2.0 / (lambda_min_exact + lambda_max_exact)

    x1, res1, t1 = run_with_timer(A, b, tau_arbitrary, max_iter, tol)
    x2, res2, t2 = run_with_timer(A, b, tau_opt_est, max_iter, tol)
    x3, res3, t3 = run_with_timer(A, b, tau_opt_exact, max_iter, tol)

    err1 = np.linalg.norm(x1 - x_exact)
    err2 = np.linalg.norm(x2 - x_exact)
    err3 = np.linalg.norm(x3 - x_exact)

    print("ПАРАМЕТРЫ МАТРИЦЫ")
    print(f"Размер n = {n} \n")

    print("ОЦЕНКИ СОБСТВЕННЫХ ЧИСЕЛ")
    print(f"lambda_min (оценка Гершгорина) = {lambda_min_est:.12e}")
    print(f"lambda_max (оценка Гершгорина) = {lambda_max_est:.12e} \n")
    print(f"lambda_min (точное) = {lambda_min_exact:.12e}")
    print(f"lambda_max (точное) = {lambda_max_exact:.12e} \n")

    print("ДОПУСТИМАЯ ОБЛАСТЬ ДЛЯ tau")
    print(f"По точному lambda_max: 0 < tau < {2.0 / lambda_max_exact:.12e}")
    print(f"По оценке lambda_max: 0 < tau < {2.0 / lambda_max_est:.12e} \n")

    print("ВЫБРАННЫЕ tau")
    print(f"tau_arbitrary  = {tau_arbitrary:.12e}")
    print(f"tau_opt_est    = {tau_opt_est:.12e}")
    print(f"tau_opt_exact  = {tau_opt_exact:.12e} \n")

    print("ВРЕМЯ РАБОТЫ")
    print(f"numpy.linalg.solve: {solve_time:.12e} сек")
    print(f"Richardson (tau_arbitrary): {t1:.12e} сек")
    print(f"Richardson (tau_opt_est):   {t2:.12e} сек")
    print(f"Richardson (tau_opt_exact): {t3:.12e} сек \n")

    print("ЧИСЛО ИТЕРАЦИЙ")
    print(f"tau_arbitrary: {len(res1) - 1}")
    print(f"tau_opt_est:   {len(res2) - 1}")
    print(f"tau_opt_exact: {len(res3) - 1} \n")

    print("НОРМА РАЗНОСТИ С ТОЧНЫМ РЕШЕНИЕМ")
    print(f"||x1 - x_exact|| = {err1:.12e}")
    print(f"||x2 - x_exact|| = {err2:.12e}")
    print(f"||x3 - x_exact|| = {err3:.12e} \n")

    print("ФИНАЛЬНЫЕ НОРМЫ НЕВЯЗКИ")
    print(f"tau_arbitrary: {res1[-1]:.12e}")
    print(f"tau_opt_est:   {res2[-1]:.12e}")
    print(f"tau_opt_exact: {res3[-1]:.12e} \n")

    plt.figure(figsize=(10, 6))
    plt.plot(res1, label="tau arbitrary")
    plt.plot(res2, label="tau opt (Gershgorin)")
    plt.plot(res3, label="tau opt (exact eigenvalues)")
    plt.yscale("log")

    plt.xlabel("Номер итерации k")
    plt.ylabel("log(||r_k||)")
    plt.title("Метод Ричардсона: зависимость логарифма нормы невязки от номера итерации")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
