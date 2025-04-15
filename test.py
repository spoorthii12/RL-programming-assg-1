import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import register_keras_serializable
from CloudComputing import ServerAllocationEnv

# ---------------------------
# Global Constants and Default Test Values
# ---------------------------
STATE_DIM = 5       # Updated: (transformed priority, job_type, network_usage, processing_time, job_count)
ACTION_DIM = 8      # Actions: 1 to 8 (internally indices 0..7)

# Default values for synthetic contexts
DEFAULT_PRIORITY = 1          # Default priority value (when not varying)
DEFAULT_NUM_JOBS = 4          # Default number of jobs in a context
DEFAULT_JOB_TYPE = "A"        # Default job type ("A", "B", or "C")
DEFAULT_NETWORK_USAGE = 0.5     # Default network usage value
DEFAULT_PROC_TIME = 20.0        # Default estimated processing time (seconds)

# ---------------------------
# Define the PolicyNetwork with Serialization Support
# ---------------------------
@register_keras_serializable()
class PolicyNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, **kwargs):
        super(PolicyNetwork, self).__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = tf.keras.layers.Dense(32, activation='relu', input_shape=(state_dim,))
        self.fc2 = tf.keras.layers.Dense(action_dim, activation='softmax')
    
    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)
    
    def get_config(self):
        config = super(PolicyNetwork, self).get_config()
        config.update({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        if "state_dim" not in config:
            config["state_dim"] = STATE_DIM
        if "action_dim" not in config:
            config["action_dim"] = ACTION_DIM
        return cls(**config)

# ---------------------------
# Load the Trained Model
# ---------------------------
loaded_model = tf.keras.models.load_model("pg_rl_agent.keras", custom_objects={'PolicyNetwork': PolicyNetwork}, compile=False)
# Force the model to build its layers by calling it on a dummy input.
dummy_input = tf.zeros((1, STATE_DIM))
_ = loaded_model(dummy_input)

# ---------------------------
# Helper: Extract State from an Observation
# ---------------------------
def extract_state(obs):
    """
    Given an observation (a tuple of job tuples) from ServerAllocationEnv,
    compute a state vector as follows:
      1. For each job, transform the priority using: transformed_priority = 4 - priority
      2. Encode the job type ("A":0, "B":1, "C":2)
      3. Convert network_usage and processing_time to floats.
      4. Compute the average of these four features.
      5. Append the number of jobs (as a float) as the fifth feature.
    Returns a 5D numpy vector.
    """
    if len(obs) == 0:
        return np.zeros(STATE_DIM, dtype=np.float32)
    
    job_vectors = []
    for job in obs:
        if len(job) != 4:
            continue
        priority, job_type, net_usage, proc_time = job
        
        # Transform priority: lower raw value (e.g., 1) becomes higher (3)
        transformed_priority = 4 - priority
        
        # Encode job_type
        if isinstance(job_type, str):
            job_type = {"A": 0, "B": 1, "C": 2}.get(job_type, -1)
        
        # Ensure net_usage and proc_time are floats
        if isinstance(net_usage, (list, np.ndarray)):
            net_usage = float(net_usage[0])
        else:
            net_usage = float(net_usage)
        if isinstance(proc_time, (list, np.ndarray)):
            proc_time = float(proc_time[0])
        else:
            proc_time = float(proc_time)
        
        job_vectors.append([transformed_priority, job_type, net_usage, proc_time])
    
    if len(job_vectors) == 0:
        return np.zeros(STATE_DIM, dtype=np.float32)
    
    avg_features = np.mean(np.array(job_vectors, dtype=np.float32), axis=0)
    job_count = float(len(obs))
    state_5d = np.concatenate([avg_features, [job_count]])
    return state_5d

# ---------------------------
# Compute Test Normalization Parameters
# ---------------------------
def compute_test_normalization(env, n_steps=1000):
    """
    Run the environment for n_steps using random actions to compute
    the mean and standard deviation of raw state vectors.
    """
    states = []
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < n_steps:
        s = extract_state(obs)
        states.append(s)
        action = np.random.randint(1, ACTION_DIM+1)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1
    states = np.array(states, dtype=np.float32)
    mean = np.mean(states, axis=0)
    std = np.std(states, axis=0) + 1e-8
    return mean, std

def normalize_state(x, mean, std):
    return (x - mean) / std

# ---------------------------
# Functions to Generate Synthetic Test Contexts
# ---------------------------
def generate_context_priority(priority, num_jobs=DEFAULT_NUM_JOBS):
    """
    Generate a context where every job has the given priority.
    Other features are fixed to default values.
    """
    context = []
    for _ in range(num_jobs):
        context.append((priority, DEFAULT_JOB_TYPE, DEFAULT_NETWORK_USAGE, DEFAULT_PROC_TIME))
    return tuple(context)

def generate_context_processing_time(proc_time, num_jobs=DEFAULT_NUM_JOBS):
    """
    Generate a context where every job has the given estimated processing time.
    Other features are fixed (using DEFAULT_PRIORITY for priority).
    """
    context = []
    for _ in range(num_jobs):
        context.append((DEFAULT_PRIORITY, DEFAULT_JOB_TYPE, DEFAULT_NETWORK_USAGE, proc_time))
    return tuple(context)

def generate_context_num_jobs(n):
    """
    Generate a context with exactly n jobs.
    Each job uses the default features.
    """
    context = []
    for _ in range(n):
        context.append((DEFAULT_PRIORITY, DEFAULT_JOB_TYPE, DEFAULT_NETWORK_USAGE, DEFAULT_PROC_TIME))
    return tuple(context)

# ---------------------------
# Action Query Function Using Loaded Model (Greedy)
# ---------------------------
def get_action_from_loaded_model(state_vec, temperature=1.0):
    state_tensor = tf.expand_dims(tf.convert_to_tensor(state_vec, dtype=tf.float32), axis=0)
    probs = loaded_model(state_tensor).numpy()[0]
    if temperature != 1.0:
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs ** (1.0 / temperature)
        probs /= np.sum(probs)
    action_idx = np.argmax(probs)
    return action_idx + 1

# ---------------------------
# Testing Functions
# ---------------------------
def test_vs_priority():
    priorities = [1, 2, 3]  # Lower number means high priority in environment; our transformation flips it.
    results = []
    env = ServerAllocationEnv()
    norm_mean, norm_std = compute_test_normalization(env, n_steps=1000)
    for p in priorities:
        context = generate_context_priority(p)
        state_vec = extract_state(context)
        norm_state = normalize_state(state_vec, norm_mean, norm_std)
        allocated_servers = get_action_from_loaded_model(norm_state, temperature=1.0)
        results.append((p, allocated_servers))
    return results

def test_vs_processing_time():
    proc_times = [10, 20, 30, 40, 50]  # in seconds
    results = []
    env = ServerAllocationEnv()
    norm_mean, norm_std = compute_test_normalization(env, n_steps=1000)
    for t in proc_times:
        context = generate_context_processing_time(t)
        state_vec = extract_state(context)
        norm_state = normalize_state(state_vec, norm_mean, norm_std)
        allocated_servers = get_action_from_loaded_model(norm_state, temperature=1.0)
        results.append((t, allocated_servers))
    return results

def test_vs_num_jobs():
    num_jobs_list = [2, 4, 8]
    results = []
    env = ServerAllocationEnv()
    norm_mean, norm_std = compute_test_normalization(env, n_steps=1000)
    for n in num_jobs_list:
        context = generate_context_num_jobs(n)
        state_vec = extract_state(context)
        norm_state = normalize_state(state_vec, norm_mean, norm_std)
        allocated_servers = get_action_from_loaded_model(norm_state, temperature=1.0)
        results.append((n, allocated_servers))
    return results

# ---------------------------
# Main Testing: Run Tests and Plot Graphs
# ---------------------------
def main_test():
    # Graph 1: Allocated Servers vs. Priority
    prio_results = test_vs_priority()
    prio_x, prio_y = zip(*prio_results)
    plt.figure(figsize=(8,5))
    plt.plot(prio_x, prio_y, marker='o', linestyle='-', color='b')
    plt.xlabel("Job Priority (1 = high, 3 = low)")
    plt.ylabel("Allocated Servers")
    plt.title("Allocated Servers vs. Job Priority")
    plt.grid(True)
    plt.show()
    
    # Graph 2: Allocated Servers vs. Estimated Processing Time
    proc_results = test_vs_processing_time()
    proc_x, proc_y = zip(*proc_results)
    plt.figure(figsize=(8,5))
    plt.plot(proc_x, proc_y, marker='o', linestyle='-', color='g')
    plt.xlabel("Estimated Processing Time (seconds)")
    plt.ylabel("Allocated Servers")
    plt.title("Allocated Servers vs. Processing Time")
    plt.grid(True)
    plt.show()
    
    # Graph 3: Allocated Servers vs. Number of Jobs
    jobs_results = test_vs_num_jobs()
    jobs_x, jobs_y = zip(*jobs_results)
    plt.figure(figsize=(8,5))
    plt.plot(jobs_x, jobs_y, marker='o', linestyle='-', color='r')
    plt.xlabel("Number of Jobs")
    plt.ylabel("Allocated Servers")
    plt.title("Allocated Servers vs. Number of Jobs")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main_test()
