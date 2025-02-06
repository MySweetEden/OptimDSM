import neal
import numpy as np
from pyqubo import Array, Constraint, Placeholder


def sequence(matrix):
    n = len(matrix)
    # バイナリ変数 x_ij：タスク i がタスク j より先にスケジュールされるか
    x = Array.create("x", (n, n), "BINARY")

    # 制約条件
    relation_const = 0.0
    self_loop_const = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                # i が j より先にあるか、その逆かを強制
                # x[i, j] + x[j, i] = 1
                relation_const += Constraint(
                    (x[i, j] + x[j, i] - 1) ** 2, label=f"relation_{i}_{j}"
                )
            else:
                # 自己ループを禁止
                # x[i][j] = 0
                self_loop_const += Constraint(x[i][j] ** 2, label=f"self_loop_{i}")

    # サイクル防止制約
    no_cycle_const = 0.0
    dummy_const = 0.0
    c0 = Array.create("c", (n, n, n), "BINARY")
    c1 = Array.create("c", (n, n, n), "BINARY")
    c2 = Array.create("c", (n, n, n), "BINARY")
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and i != k:
                    # i -> j -> k ならば i -> k も成り立つべき
                    # x[i][j] + x[j][k] <= 1 + x[i][k]
                    no_cycle_const += Constraint(
                        # x[i][j] + x[j][k] + 0 * c0[i][j][k] + 1 * c1[i][j][k] + 2 * c2[i][j][k] = 1 + x[i][k]
                        (
                            1
                            + x[i][k]
                            - x[i][j]
                            - x[j][k]
                            - 0 * c0[i][j][k]
                            - 1 * c1[i][j][k]
                            - 2 * c2[i][j][k]
                        )
                        ** 2,
                        label=f"no_cycle_{i}_{j}_{k}",
                    )
                    dummy_const += Constraint(
                        # c0[i][j][k] + c1[i][j][k] + c2[i][j][k] = 1
                        (c0[i][j][k] + c1[i][j][k] + c2[i][j][k] - 1) ** 2,
                        label=f"dummy_{i}_{j}_{k}",
                    )

    # Objective function: Maximize the sum of the lower_sum triangular matrix (i > j)
    lower_sum = sum(matrix[i][j] * x[i][j] for i in range(j) for j in range(n))

    A = Placeholder("A")
    H = -lower_sum + A * (
        relation_const + self_loop_const + no_cycle_const + dummy_const
    )
    model = H.compile()
    feed_dict = {"A": n * (n - 1) / 2 + 1}
    bqm = model.to_bqm(feed_dict=feed_dict)
    sa = neal.SimulatedAnnealingSampler()
    sampleset = sa.sample(
        bqm,
        seed=1234,
        num_reads=1000,
        num_sweeps=100000,
        beta_range=[0.1, 10],
        beta_schedule_type="linear",
    )

    # Decode solution
    decoded_samples = model.decode_sampleset(sampleset, feed_dict=feed_dict)
    best_sample = min(decoded_samples, key=lambda x: x.energy)
    # num_broken = len(best_sample.constraints(only_broken=True))
    # print("number of broken constarint = {}".format(num_broken))

    result = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            result[i][j] = best_sample.sample[f"x[{i}][{j}]"]

    # 最適な順序を取得
    sequence_order = sorted(range(n), key=lambda i: sum(result[i][j] for j in range(n)))
    # 最適化後の行列
    optimized_matrix = matrix[np.ix_(sequence_order, sequence_order)]

    return optimized_matrix
