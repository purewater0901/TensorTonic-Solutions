import numpy as np

def value_iteration(P, R, gamma, tol=1e-6, max_iters=1000):
    P = np.array(P)
    R = np.array(R)

    S, A, _ = P.shape

    values = np.zeros(S)
    pi = np.zeros(S, dtype=np.int32)

    for iteration in range(max_iters):
        old_values = values.copy()

        for state in range(S):
            max_val = -np.inf

            for action in range(A):
                target = 0.0

                for next_state in range(S):
                    p = P[state, action, next_state]
                    r = R[state, action, next_state]

                    target += p * (
                        r + gamma * old_values[next_state]
                    )

                if target > max_val:
                    max_val = target

            values[state] = max_val

        delta = np.max(np.abs(values - old_values))
        if delta < tol:
            break

    for state in range(S):
        max_val = -np.inf
        for action in range(A):
            target = 0.0
            for next_state in range(S):
                p = P[state, action, next_state]
                r = R[state, action, next_state]

                target += p * (
                        r + gamma * values[next_state]
                    )

            if target > max_val:
                max_val = target
                pi[state] = action

    return np.round(values, 4).tolist(), pi.tolist()