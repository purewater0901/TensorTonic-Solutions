def mc_policy_evaluation(episodes: list, gamma: float, num_states: int) -> list:
    """
    Returns first-visit state values rounded to four decimals; unvisited states are 0.0.
    """
    values = [0.0] * num_states
    counts = [0] * num_states

    for episode in episodes:
        rewards = []
        G = 0.0
        for state, reward in reversed(episode):
            G = reward + gamma * G
            rewards.append(G)
        rewards = rewards[::-1]

        visited = [False] * num_states
        for i in range(len(rewards)):
            state = episode[i][0]
            reward_to_go = rewards[i]
            if visited[state]:
                continue
            visited[state] = True
            values[state] += reward_to_go
            counts[state] += 1

    for i in range(num_states):
        if counts[i] > 0:
            values[i] /= counts[i]

    return values
            