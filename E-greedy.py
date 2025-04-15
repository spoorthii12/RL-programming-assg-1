import numpy as np
import matplotlib.pyplot as plt
from CloudComputing import ServerAllocationEnv

# ---------------------------
# 1. Hyperparameters
# ---------------------------
EPISODES = 30          # How many full episodes to run (each up to 1440 steps).
DIM = 4                # Context dimension: (priority, job_type, network_usage, processing_time)
ACTIONS = 8            # Discrete arms: 1..8 servers
ALPHA = 1.0            # Regularization parameter for A_a (controls A_a = alpha * I initially)
WINDOW_SIZE = 500      # Sliding window size for plotting
INITIAL_EPSILON = 1.0  # Starting exploration rate
FINAL_EPSILON = 0.01   # Minimum exploration rate
DECAY_RATE = 0.95      # Exponential decay for epsilon

# ---------------------------
# 2. Helper: Extract a 4D Context from the Env
# ---------------------------
def extract_context(env_observation):
    """
    env_observation = tuple of jobs, each job is (priority, job_type, network_usage, processing_time)
    We average them into a 4D vector.
    """
    if len(env_observation) == 0:
        return np.zeros(DIM, dtype=np.float32)

    job_vectors = []
    for job in env_observation:
        # job = (priority, job_type, net_usage, proc_time)
        if len(job) != 4:
            continue
        priority, job_type, net_usage, proc_time = job

        # Convert job_type from string to int if needed (A->0, B->1, C->2)
        if isinstance(job_type, str):
            job_type = {"A": 0, "B": 1, "C": 2}.get(job_type, -1)

        # Convert net_usage, proc_time to floats if they are arrays
        if isinstance(net_usage, (list, np.ndarray)):
            net_usage = float(net_usage[0])
        else:
            net_usage = float(net_usage)
        if isinstance(proc_time, (list, np.ndarray)):
            proc_time = float(proc_time[0])
        else:
            proc_time = float(proc_time)

        job_vectors.append([priority, job_type, net_usage, proc_time])

    if len(job_vectors) == 0:
        return np.zeros(DIM, dtype=np.float32)

    job_vectors = np.array(job_vectors, dtype=np.float32)
    return np.mean(job_vectors, axis=0)

# ---------------------------
# 3. Epsilon-Greedy for Linear Bandits
# ---------------------------
class LinearBanditEpsGreedy:
    """
    Maintains a separate linear model for each action:
      r_a(x) = theta_a^T x
    With A_a = alpha*I + sum(x_t x_t^T) for all t where action=a,
         b_a = sum(x_t * r_t)
         theta_a = A_a^{-1} b_a
    """
    def __init__(self, dim, n_actions, alpha=1.0):
        self.dim = dim
        self.n_actions = n_actions
        self.alpha = alpha

        # For each action a, we keep A_a (dxd), b_a (dx1), and theta_a (dx1)
        self.A = [alpha * np.eye(dim) for _ in range(n_actions)]
        self.b = [np.zeros(dim) for _ in range(n_actions)]
        self.theta = [np.zeros(dim) for _ in range(n_actions)]

    def update_theta(self, action):
        # Recompute theta_a = A_a^{-1} b_a
        invA = np.linalg.inv(self.A[action])
        self.theta[action] = invA @ self.b[action]

    def choose_action(self, x, epsilon):
        """
        Epsilon-greedy:
        With probability epsilon, choose random action.
        Otherwise, pick argmax of theta_a^T x.
        """
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)  # 0..(n_actions-1)
        else:
            # Evaluate r_hat_a = theta_a^T x for each arm
            estimates = [self.theta[a].dot(x) for a in range(self.n_actions)]
            return int(np.argmax(estimates))

    def update(self, action, x, reward):
        """
        A_a = A_a + x x^T
        b_a = b_a + x * reward
        Then recompute theta_a.
        """
        x = x.reshape(-1, 1)  # dx1
        self.A[action] += x @ x.T  # x x^T is dx1 * 1xd => dxd
        self.b[action] += (reward * x.flatten())  # add dx1
        self.update_theta(action)

# ---------------------------
# 4. Main Training Loop
# ---------------------------
def train_linear_bandit_epsilon_greedy():
    env = ServerAllocationEnv()
    bandit = LinearBanditEpsGreedy(dim=DIM, n_actions=ACTIONS, alpha=ALPHA)

    all_rewards = []  # Store immediate reward for each step
    total_steps = 0

    epsilon = INITIAL_EPSILON  # start exploration rate

    for episode in range(EPISODES):
        obs, _ = env.reset()
        done = False
        t = 0

        while not done:
            x = extract_context(obs)
            # Choose action in [0..7], environment expects [1..8]
            action_idx = bandit.choose_action(x, epsilon)
            action_env = action_idx + 1

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(int(action_env))
            all_rewards.append(reward)

            # Update bandit parameters
            bandit.update(action_idx, x, reward)

            obs = next_obs
            done = terminated or truncated
            t += 1
            total_steps += 1

        # Decay epsilon after each episode
        epsilon = max(FINAL_EPSILON, epsilon * DECAY_RATE)
        print(f"Episode {episode+1}/{EPISODES}, Steps in Episode: {t}, Epsilon: {epsilon:.3f}")

    # Plot receding window time-averaged reward
    all_rewards = np.array(all_rewards, dtype=np.float32)
    if len(all_rewards) < WINDOW_SIZE:
        print(f"Not enough steps collected ({len(all_rewards)}) to compute moving average with window={WINDOW_SIZE}")
        return

    moving_avg = np.convolve(all_rewards, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(moving_avg, marker='o', linestyle='-', color='b')
    plt.title(f"Epsilon-Greedy Linear Bandit\nReceding Window ({WINDOW_SIZE}) Time-Averaged Reward")
    plt.xlabel("Time Step (sliding window index)")
    plt.ylabel("Reward (Moving Average)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_linear_bandit_epsilon_greedy()
