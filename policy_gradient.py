import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from CloudComputing import ServerAllocationEnv
from tensorflow.keras.utils import register_keras_serializable


#Hyperparameters
BATCH_SIZE = 500             # Steps per batch (collects these many number of (x,a,r) samples)
NUM_BATCHES = 250            # Number of batches(N)
LEARNING_RATE = 0.001        # For policy network (gradient ascent)
STATE_DIM = 5                # (priority, job_type, network_usage, processing_time)
ACTION_DIM = 8               # Discrete actions between 1 to 8
INITIAL_TEMP = 1.0           # higher the initial temperature higher the exploration
FINAL_TEMP = 0.1             # the exploration shloud gradually decrease 
TEMP_DECAY = 0.95            # exploration is reduced by 5% each time it is updated
WINDOW_SIZE = 500            # Sliding window size for receding average

#Incremental Normalizer-Welford’s Algorithm for calculating the mean and variance 
def init_normalizer(state_dim):
    return {
        "mean": np.zeros(state_dim, dtype=np.float32),
        "var": np.zeros(state_dim, dtype=np.float32),
        "count": 0
    }

def update_normalizer(normalizer, x):
    x = np.array(x, dtype=np.float32)
    normalizer["count"] += 1
    delta = x - normalizer["mean"]
    normalizer["mean"] += delta / normalizer["count"]
    normalizer["var"] += delta * (x - normalizer["mean"])

def normalize_state(normalizer, x):
    if normalizer["count"] > 1:
        std = np.sqrt(normalizer["var"] / (normalizer["count"] - 1)) + 1e-8
        return (x - normalizer["mean"]) / std
    else:
        return x

#Policy Network
def create_policy_network(state_dim, action_dim):
    inputs = tf.keras.Input(shape=(state_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    outputs = layers.Dense(action_dim, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def policy_gradient_loss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0)
    log_likelihood = tf.math.log(y_pred)
    return -tf.reduce_sum(y_true * log_likelihood, axis=1)

#Policy Gradient Agent
def init_agent(state_dim, action_dim, lr=LEARNING_RATE):
    model = create_policy_network(state_dim, action_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=policy_gradient_loss
    )
    return {"model": model, "state_dim": state_dim, "action_dim": action_dim}

def get_action(agent, state_vec, temperature=1.0):
    state_tensor = tf.expand_dims(tf.convert_to_tensor(state_vec, dtype=tf.float32), axis=0)
    probs = agent["model"](state_tensor)[0].numpy()
    
    if temperature != 1.0:
        probs = np.clip(probs, 1e-10, 1.0)
        probs = probs ** (1.0 / temperature)
        probs /= np.sum(probs)
    
    action_idx = np.random.choice(agent["action_dim"], p=probs)
    return action_idx + 1, probs[action_idx]

def train_agent_on_batch(agent, states, actions, rewards):
    # Converting actions from [1-8] to one-hot encoding for [0-7]
    actions_idx = np.array(actions) - 1
    y_true = tf.keras.utils.to_categorical(actions_idx, num_classes=agent["action_dim"])
    agent["model"].train_on_batch(states, y_true, sample_weight=rewards)

#Extract State
def extract_state(obs):
    if len(obs) == 0:
        return np.zeros(STATE_DIM, dtype=np.float32)
    
    job_vectors = []
    for job in obs:
        if len(job) != 4:
            continue
        priority, job_type, net_usage, proc_time = job
        
         
        
        # Encode job_type if it's a string
        if isinstance(job_type, str):
            job_type = {"A": 0, "B": 1, "C": 2}.get(job_type, -1)
        
        # Convert net_usage and proc_time to floats
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
        return np.zeros(STATE_DIM, dtype=np.float32)
    
    avg_features = np.mean(np.array(job_vectors, dtype=np.float32), axis=0)
    
    job_count = float(len(obs))
    state_5d = np.concatenate([avg_features, [job_count]])
    return state_5d


#Training
def train_policy_gradient():
    env = ServerAllocationEnv()
    agent = init_agent(STATE_DIM, ACTION_DIM)
    normalizer = init_normalizer(STATE_DIM)

    all_rewards = []
    temperature = INITIAL_TEMP

    for batch_i in range(NUM_BATCHES):
        batch_states = []
        batch_actions = []
        batch_rewards = []
        steps_collected = 0

        # Collect BATCH_SIZE samples
        while steps_collected < BATCH_SIZE:
            obs, _ = env.reset()
            done = False

            while not done:
                # Extract and normalize state
                s_vec = extract_state(obs)
                update_normalizer(normalizer, s_vec)
                norm_state = normalize_state(normalizer, s_vec)

                # Sample action from policy
                action, action_prob = get_action(agent, norm_state, temperature=temperature)
                next_obs, reward, terminated, truncated, _ = env.step(int(action))

                # Store data
                batch_states.append(norm_state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                all_rewards.append(reward)

                obs = next_obs
                done = terminated or truncated
                steps_collected += 1
                if steps_collected >= BATCH_SIZE:
                    break
        
        batch_states_np = np.array(batch_states, dtype=np.float32)
        batch_actions_np = np.array(batch_actions, dtype=np.int32)
        batch_rewards_np = np.array(batch_rewards, dtype=np.float32)

        # Update the policy network using the batch data
        train_agent_on_batch(agent, batch_states_np, batch_actions_np, batch_rewards_np)

        # Decay temperature for next batch
        temperature = max(FINAL_TEMP, temperature * TEMP_DECAY)

        avg_reward = np.mean(batch_rewards_np)
        print(f"Batch {batch_i+1}/{NUM_BATCHES}, Steps: {steps_collected}, Temp: {temperature:.3f}, Avg Reward: {avg_reward:.2f}")
        
        agent["model"].save("pg_rl_agent.keras")
        
    print("Training completed!")

    #Plot of Receding Window Time-Averaged Reward
    all_rewards = np.array(all_rewards, dtype=np.float32)
    if len(all_rewards) < WINDOW_SIZE:
        print(f"Not enough steps for a sliding window of size {WINDOW_SIZE}.")
        return

    moving_avg = np.convolve(all_rewards, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')

    plt.figure(figsize=(15,8))
    plt.plot(moving_avg, linestyle='-', color='0')
    plt.xlabel("Time Step (Sliding Window Index)")
    plt.ylabel("Time-Averaged Reward")
    plt.title(f"Policy Gradient for Contextual Bandits\n(Receding Window = {WINDOW_SIZE})")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_policy_gradient()