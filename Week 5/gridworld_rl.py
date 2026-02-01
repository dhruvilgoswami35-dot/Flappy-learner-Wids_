import numpy as np
import matplotlib.pyplot as plt
import random

class Gridworld:
    def __init__(self, size=4):
        self.size = size
        self.state = (0, 0)
        self.goal = (size - 1, size - 1)

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            y = min(self.size - 1, y + 1)
        elif action == 2:
            x = min(self.size - 1, x + 1)
        elif action == 3:
            y = max(0, y - 1)
            
        self.state = (x, y)
        
        if self.state == self.goal:
            return self.state, 10, True
        else:
            return self.state, -1, False

def get_action(Q_table, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        state_idx = state[0] * 4 + state[1]
        return np.argmax(Q_table[state_idx])

def run_sarsa(episodes=500):
    env = Gridworld()
    Q = np.zeros((16, 4))
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    decay = 0.995
    
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        state_idx = state[0] * 4 + state[1]
        
        action = get_action(Q, state, epsilon)
        
        done = False
        total_reward = 0
        
        while not done:
            next_state, reward, done = env.step(action)
            next_state_idx = next_state[0] * 4 + next_state[1]
            
            next_action = get_action(Q, next_state, epsilon)
            
            target = reward + gamma * Q[next_state_idx][next_action]
            Q[state_idx][action] += alpha * (target - Q[state_idx][action])
            
            state = next_state
            state_idx = next_state_idx
            action = next_action
            total_reward += reward
            
        rewards_history.append(total_reward)
        if epsilon > 0.01: epsilon *= decay
        
    return rewards_history

def run_q_learning(episodes=500):
    env = Gridworld()
    Q = np.zeros((16, 4))
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    decay = 0.995
    
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        state_idx = state[0] * 4 + state[1]
        
        done = False
        total_reward = 0
        
        while not done:
            action = get_action(Q, state, epsilon)
            
            next_state, reward, done = env.step(action)
            next_state_idx = next_state[0] * 4 + next_state[1]
            
            best_next_action = np.max(Q[next_state_idx])
            target = reward + gamma * best_next_action
            Q[state_idx][action] += alpha * (target - Q[state_idx][action])
            
            state = next_state
            state_idx = next_state_idx
            total_reward += reward
            
        rewards_history.append(total_reward)
        if epsilon > 0.01: epsilon *= decay
        
    return rewards_history

if __name__ == "__main__":
    print("Running SARSA...")
    sarsa_rewards = run_sarsa()
    
    print("Running Q-Learning...")
    q_rewards = run_q_learning()
    
    def smooth(data, window=20):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    plt.plot(smooth(sarsa_rewards), label='SARSA')
    plt.plot(smooth(q_rewards), label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Comparison: SARSA vs Q-Learning')
    plt.legend()
    plt.savefig('week5_results.png')
    plt.show()
    print("Done! Graph saved as week5_results.png")