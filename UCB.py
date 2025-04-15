import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from CloudComputing import ServerAllocationEnv

# Initialize the Server Allocation Environment
env = ServerAllocationEnv()

# Hyperparameters (Fine-Tuned)
episodes = 40000  # Number of training iterations
epsilon = 1.0  # Exploration-exploitation balance (will decay)
delta = 0.7  # Regularization parameter for stable matrix updates
window_size = 500  # Sliding window size for reward averaging
d = 4  # Feature vector dimension (priority, job type, network use, processing time)

# Initialize LinUCB Matrices for each server allocation (1 to max servers)
A = {a: np.identity(d) for a in range(1, env.MaxServers + 1)}
b = {a: np.zeros(d) for a in range(1, env.MaxServers + 1)}

# Track rewards for analysis
rewards_history = []

# Training loop
for episode in range(episodes):
    state, _ = env.reset()

    # Convert job context into feature vectors
    if state:  # Check if there are jobs in the state
        feature_vectors = np.array([
            [job[0], {'A': 0, 'B': 1, 'C': 2}.get(job[1], -1), job[2], job[3]]
            for job in state
        ], dtype=np.float32)
        z_t = np.mean(feature_vectors, axis=0)
    else:
        z_t = np.zeros(d, dtype=np.float32)

    # Fine-Tuned Epsilon Decay for Smoother Learning
    epsilon = max(0.05, 1 - 0.4 * (episode / episodes))  

    # Compute Upper Confidence Bound (UCB) for each action
    ucb_values = {}
    for a in range(1, env.MaxServers + 1):
        inv_A_a = np.linalg.inv(A[a] + delta * np.identity(d))  # Regularization for stability
        theta_a = inv_A_a @ b[a]  # Estimate parameter θ
        uncertainty = np.sqrt(z_t.T @ inv_A_a @ z_t)  # Confidence bound
        ucb_values[a] = z_t.T @ theta_a + epsilon * uncertainty  # UCB-based action selection

    # Choose the action with the highest UCB value
    action = max(ucb_values, key=ucb_values.get)

    # Execute action and observe reward
    _, reward, _, truncated, _ = env.step(action)
    rewards_history.append(reward)

    # Update LinUCB Matrices
    A[action] += np.outer(z_t, z_t)  # Update matrix A
    b[action] += reward * z_t  # Update vector b

    # Stop early if environment truncates
    if truncated:
        break

# Compute moving average for time-averaged reward
window_avg_rewards = np.convolve(
    rewards_history, np.ones(window_size) / window_size, mode="valid"
)

# Plot Time-Averaged Reward vs. Time Step
plt.figure(figsize=(10, 5))
plt.plot(range(len(window_avg_rewards)), window_avg_rewards, label="Time-Averaged Reward")
plt.xlabel("Sliding Window Index (Time Step)")
plt.ylabel("Time-Averaged Reward")
plt.title("Fine-Tuned LinUCB: Time-Averaged Reward vs. Time Step")
plt.legend()
plt.grid()
plt.show()