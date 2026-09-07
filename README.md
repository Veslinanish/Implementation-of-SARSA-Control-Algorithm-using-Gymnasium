# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---

## Problem Statement

To implement the SARSA control algorithm using the Gymnasium FrozenLake-v1 environment and learn an optimal or near-optimal action-value function. The agent must learn to select appropriate actions to reach the goal while avoiding the hole states.

## Software Requirements

Python 3.x
Gymnasium
NumPy
Matplotlib
Jupyter Notebook / Google Colab


## Environment Description

FrozenLake-v1 is a discrete reinforcement-learning environment consisting of a 4×4 grid. The agent starts from the initial state and must reach the goal while avoiding holes. The environment has 16 states and 4 possible actions: Left, Down, Right, and Up. A reward of 1 is obtained for reaching the goal, while other transitions normally provide zero reward.

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm

1. Create the custom FrozenLake-v1 environment with start state 5 and goal state 10.
2. Initialize the Q-table with zeros and set α, γ, ε, ε_min, and ε_decay.
3. Reset the environment and obtain the starting state S.
4. Select action A using the epsilon-greedy policy.
5. Execute A and observe reward R and next state S'.
6. Select the next action A' using the current epsilon-greedy policy.
7. Update Q(S,A) using the SARSA update rule.
8. Set S = S' and A = A', and repeat until the episode ends.
9. Decrease epsilon using ε = max(ε_min, ε × ε_decay).
10. Repeat training for all episodes and obtain the final Q-table, policy, and rewards.


## Python Program

```python
# -------------------------------------------------
# SARSA Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()

    # Select initial action
    action = epsilon_greedy_action(
        state,
        epsilon
    )

    total_reward = 0

    for step in range(max_steps_per_episode):

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Terminal state update
        if terminated or truncated:

            Q[state, action] = Q[state, action] + alpha * (
                reward - Q[state, action]
            )

            break

        # Select next action using epsilon-greedy policy
        next_action = epsilon_greedy_action(
            next_state,
            epsilon
        )

        # SARSA update
        Q[state, action] = Q[state, action] + alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        # Move to next state and action
        state = next_state
        action = next_action

    # Store episode reward
    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )
```
---

## Output

<img width="487" height="552" alt="exp 6 op 1" src="https://github.com/user-attachments/assets/5b046009-e075-4b0b-bbba-45d33485e992" />

<img width="737" height="462" alt="exp 6 op 2" src="https://github.com/user-attachments/assets/42e31de9-815c-4956-ace2-458841dffe20" />

## Result

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

---

## Inference
The experiment demonstrates that SARSA can learn an effective policy through an epsilon-greedy exploration strategy. During training, the agent initially explores different actions and gradually exploits actions with higher Q-values as epsilon decreases.

---

