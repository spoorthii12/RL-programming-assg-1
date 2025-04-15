import numpy as np
import gymnasium as gym
from scipy.stats import truncnorm, poisson, truncexpon

class ServerAllocationEnv(gym.Env):
    def __init__(self):
        self.Horizon = 60*24
        self.MaxServers = 8
        self.MaxJobs = 8
        self.MinPriority = 3
        self.NTypes = 3
        self.MaxProcessingTime = 30    # In seconds
        
        obsv1 = gym.spaces.Discrete(self.MinPriority, start=1)       # Priority level (low value means high priority)
        obsv2 = gym.spaces.Discrete(self.NTypes)                     # Type of job (three types)
        obsv3 = gym.spaces.Box(low=0,high=1)                         # Level of network usage (high value means more internet use)
        obsv4 = gym.spaces.Box(low=0,high=self.MaxProcessingTime)    # Predicted processing time (in seconds)
        self.observation_space = gym.spaces.Sequence(gym.spaces.Tuple([obsv1, obsv2, obsv3, obsv4]))
        
        self.action_space = gym.spaces.Discrete(self.MaxServers, start=1)
                
        self.NJobs_probability_mean = 5 # poisson
        
        self.type_probability = np.array([0.35, 0.1, 0.55])
        self.type_arr = np.arange(self.NTypes).astype(int)
        self.type_mapping = {0:'A', 1:'B', 2:'C'}
        
        self.priority_probability = np.zeros((3,3))
        self.priority_probability[0,:] = np.array([0.3, 0.18, 0.52])
        self.priority_probability[1,:] = np.array([0.0, 0.7, 0.3])
        self.priority_probability[2,:] = np.array([0.05,0.4,0.55])
        self.priority_arr = np.arange(1,self.MinPriority+1,1).astype(int)
        
        self.processing_time_mean = np.array([20.0, 24.0, 14.0]) # truncexpon
        
        self.network_paramater = np.array([[0.4, 0.2, 0.6, 2*0.4], [0.7, 0.55, 0.85, 2*0.7], [0.4, 0.2, 0.6, 0.8*0.4]]) # truncnorm
        
        self.processing_time_deviation_parameter = np.array([[0.6, 1, 1.5], [0.1, 0.8, 1.2], [1, 0.6, 1.1]])
        
        self.penalty = 300
        self.ServerCost = np.array([0.00, 14.84, 11.35, 5.04, 2.94, 2.07, 1.70, 1.22])
        self.weight = 1
        
        self.processing_time = None
        self.observation = None
        self.t = None
        
        
    def step(self, action):
        assert self.observation is not None, "Call reset before using step method!"
        assert self.t<self.Horizon, "The number of time slots exceeds the time horizon!"

        reward = self.generate_reward(action)
        
        self.t+=1
        
        self.generate_batch()
        
        terminated= False
        
        truncated = False
        if self.t>=self.Horizon:
            truncated = True
        
        return self.observation, reward, terminated, truncated, {}
    
    
    def reset(self):
        self.t = 0
        self.generate_batch()
        return self.observation, {}
        
    
    def generate_reward(self, action):        
        priority = np.array([val[0] for val in self.observation])
        priority_ix = np.argsort(priority)
        
        latency = np.zeros(action)
        total_latency = 0
        
        for ix in priority_ix:
            vm_ix = np.argmin(latency)
            
            total_latency+=(self.MinPriority-priority[ix]+1)*latency[vm_ix]
            if latency[vm_ix]>60:
                total_latency+=(self.MinPriority-priority[ix]+1)*self.penalty
                
            latency[vm_ix]+=self.processing_time[ix]
        
        
        vm_cost = 0
        for ix in range(action):
            mean_cost = self.ServerCost[ix]
            if mean_cost>0:
                loc = mean_cost
                scale = 0.8*mean_cost
                a = (0.8*loc-loc)/scale
                b = (1.2*loc-loc)/scale
                vm_cost+=truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
        
        reward = -(total_latency+self.weight*vm_cost)
        
        return reward
    
    
    def generate_batch(self):
        if self.t>=self.Horizon:
            self.observation = tuple()
        else:
            Njobs = max(1,min(poisson.rvs(self.NJobs_probability_mean), self.MaxJobs))
            self.processing_time = np.zeros(Njobs)
            observation = []
            for i in range(Njobs):
                job_type_ix = np.random.choice(self.type_arr, p=self.type_probability)
                job_type = self.type_mapping[job_type_ix]
                
                priority = np.random.choice(self.priority_arr, p=self.priority_probability[job_type_ix,:])
                
                processing_time = truncexpon.rvs(b=self.MaxProcessingTime, scale=self.processing_time_mean[job_type_ix])
                self.processing_time[i] = processing_time
                
                loc = self.network_paramater[job_type_ix,0]
                scale = self.network_paramater[job_type_ix,3]
                a = (self.network_paramater[job_type_ix,1] - loc)/scale
                b = (self.network_paramater[job_type_ix,2] - loc)/scale
                network_use = truncnorm.rvs(a=a, b=b, loc=loc, scale=scale)
                
                network_time = network_use*processing_time
                true_processing_time = processing_time - network_time
                
                prob = self.processing_time_deviation_parameter[job_type_ix,0]
                lb = self.processing_time_deviation_parameter[job_type_ix,1]
                ub = self.processing_time_deviation_parameter[job_type_ix,2]
                ratio = np.random.uniform(low=lb, high=ub)
                estimated_processing_time = ratio*true_processing_time
                if np.random.uniform()<=prob:
                    ratio = np.random.uniform(low=lb, high=ub)
                    estimated_processing_time+=ratio*network_time
                    
                
                observation.append(tuple([priority, job_type, network_use, estimated_processing_time]))
                                               
            self.observation = tuple(observation)