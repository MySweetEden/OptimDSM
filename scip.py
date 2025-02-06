import numpy as np
from pyscipopt import Model, quicksum


def sequence(matrix):
    """
    タスク間の依存関係を最適化し、下三角行列の形に近づける順序を計算する関数。

    この関数は、PySCIPOptを使用して、タスクの最適な順序を決定します。
    タスク間の依存関係を示す行列を入力とし、依存コストを最小化する順序を出力します。

    Args:
        matrix (nd.array): タスク間の依存関係を表すn×nの行列。
            - 行列の要素 matrix[i][j] はタスク i とタスク j 間の依存コストを示す。

    Returns:
        tuple:
            - nd.array: 最適な順序に基づいて並び替えられた行列。
            - list: 最適なタスクの順序を示すリスト（インデックスの順序）。
    """

    n = len(matrix)  # タスクの数
    model = Model("Sequencing")
    model.hideOutput(True)

    # バイナリ変数 x_ij：タスク i がタスク j より先にスケジュールされるか
    x = [
        [model.addVar(f"x_{i}_{j}", vtype="BINARY") for j in range(n)] for i in range(n)
    ]

    # 制約条件
    for i in range(n):
        for j in range(n):
            if i != j:
                # i が j より先にあるか、その逆かを強制
                model.addCons(x[i][j] + x[j][i] == 1, name=f"x_{i}_{j}_relation")
            else:
                # 自己ループを禁止
                model.addCons(x[i][j] == 0, name=f"x_{i}_{j}_self_loop")

    # サイクル防止制約
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and i != k:
                    # i -> j -> k ならば i -> k も成り立つべき
                    model.addCons(
                        x[i][j] + x[j][k] <= 1 + x[i][k], name=f"no_cycle_{i}_{j}_{k}"
                    )

    # 目的関数: タスク間の依存コストを最小化
    model.setObjective(
        quicksum(matrix[i][j] * x[i][j] for i in range(n) for j in range(n)),
        sense="maximize",
    )

    # 最適化の実行
    model.optimize()

    # 最適な順序を取得
    sequence_order = sorted(
        range(n), key=lambda i: sum(model.getVal(x[i][j]) for j in range(n))
    )

    # 最適化後の行列
    optimized_matrix = matrix[np.ix_(sequence_order, sequence_order)]

    return optimized_matrix
