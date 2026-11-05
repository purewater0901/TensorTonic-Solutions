import numpy as np

def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    # Write code here
    num_s = len(values)
    num_a = len(rewards[0])

    values = np.array(values) # (S,)
    transitions = np.array(transitions) # (S, A, S)
    rewards = np.array(rewards) # (S, A)

    new_values = rewards + gamma * transitions @ values # (S, A)
    new_values = np.max(new_values, axis=-1)
    return new_values.tolist()