# RL-programming-assg-1

---

# 🚀 Contextual Bandits for Server Allocation (CS4122 - RL Assignment 1)

This repository contains the implementation and experiments for our **Reinforcement Learning (RL) Programming Assignment 1**. The task was to solve a **Contextual Bandit problem** using three different strategies:  
- **ε-Greedy**  
- **Upper Confidence Bound (UCB)**  
- **Policy Gradient**

The environment is a custom Gymnasium-based simulator: `ServerAllocationEnv`, representing a data center job scheduling problem.

---

## 📁 Files in the Repository

| File | Description |
|------|-------------|
| `lin_greedy.py` | ε-Greedy implementation |
| `lin_ucb.py` | Upper Confidence Bound (UCB) implementation |
| `policy_gradient.py` | Policy Gradient algorithm |
| `CloudComputing.py` | Provided environment file (do not modify) |
| `report.pdf` | Final report detailing the analysis, code, and results |
| `README.md` | You are here! |

---

## 🧠 Problem Summary

The **Server Allocation Environment** simulates jobs arriving at a data center. Each job has features like:
- Priority
- Job Type (A, B, C)
- Network Usage
- Estimated Processing Time

The **goal** is to allocate the optimal number of servers to each batch of jobs, minimizing cost (or equivalently, maximizing reward).

### 🤖 Why Contextual Bandits?

This environment is a **stochastic contextual bandit** setup due to:
- Random number of jobs per time step
- Variable job characteristics
- Noisy rewards based on latency and energy consumption

---

## 🛠️ Feature Engineering

- States (contexts) are extracted by **averaging job features** into a **fixed-size 5D vector**.
- Features: `avg_priority`, `avg_job_type`, `avg_network`, `avg_processing_time`, `job_count`.

Normalization is performed using **Welford’s incremental mean-variance algorithm**.

---

## 📈 Methods Implemented

### 1. ε-Greedy (`lin_greedy.py`)
- Explores with probability ε, otherwise selects the best-known action.

### 2. UCB (`lin_ucb.py`)
- Balances exploration and exploitation using upper confidence bounds.
- Evaluation showed correct correlations between:
  - Priority and servers allocated
  - Processing time and servers allocated
  - Number of jobs and servers allocated

### 3. Policy Gradient (`policy_gradient.py`)
- Stochastic policy trained via gradient ascent
- Softmax output over 8 discrete actions
- Exploration managed by temperature scheduling (decayed over batches)

---

## 📊 Results

### ✅ UCB Logical Evaluation:
- More servers for higher priority or longer jobs.
- Dynamic adjustment based on number of jobs.

### ✅ Policy Gradient:
- Learns through reward-weighted updates.
- Final rewards plotted as **sliding-window averaged graphs**.

---

## 📌 How to Run

```bash
# Install dependencies
pip install tensorflow gymnasium matplotlib numpy

# Run Policy Gradient
python policy_gradient.py

# Run UCB
python lin_ucb.py

# Run ε-Greedy
python lin_greedy.py
```

> Make sure `CloudComputing.py` is present as it contains the environment logic.

---

## 🧪 Evaluation Metrics

- **Average reward** over batches
- **Receding window plots** for trend visualization
- **Logical consistency** in server allocation behavior
