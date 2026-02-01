import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
import os
import matplotlib.pyplot as plt # Needed for the graph
from flappy_env import FlappyBirdEnv

# --- CONFIGURATION ---
GAMES_TO_PLAY = 500
CHECKPOINT_INTERVAL = 50 # Save a backup every 50 games
LEARNING_RATE = 0.001

class SimpleBrain(nn.Module):
    def __init__(self):
        super(SimpleBrain, self).__init__()
        # Input: Vertical Dist, Velocity (2 inputs)
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.layers(x)

def train():
    env = FlappyBirdEnv()
    brain = SimpleBrain()
    optimizer = optim.Adam(brain.parameters(), lr=LEARNING_RATE)
    loss_func = nn.MSELoss()

    # Create folder for checkpoints
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    epsilon = 1.0
    epsilon_min = 0.01
    decay = 0.98 
    memory = []
    batch_size = 64
    all_scores = [] # To store data for the graph

    print(f"--- Starting Training for {GAMES_TO_PLAY} Games ---")

    for game in range(1, GAMES_TO_PLAY + 1):
        state_raw = env.reset()
        state = torch.FloatTensor(state_raw)
        
        game_over = False
        
        while not game_over:
            # OPTIONAL: Uncomment to watch training (slows it down)
            # env.render()
            pygame.event.pump() 

            # 1. Decide Action
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    q_values = brain(state)
                    action = torch.argmax(q_values).item()

            # 2. Step Environment
            next_state_raw, reward, game_over = env.step(action)
            next_state = torch.FloatTensor(next_state_raw)
            
            # 3. Save to Memory
            memory.append((state, action, reward, next_state, game_over))
            state = next_state

            if len(memory) > 10000:
                memory.pop(0)

            # 4. Train the Brain
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                
                states = torch.stack([x[0] for x in batch])
                actions = torch.tensor([x[1] for x in batch], dtype=torch.int64)
                rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                next_states = torch.stack([x[3] for x in batch])
                
                # --- THE FIX FOR YOUR ERROR IS HERE ---
                # We strictly convert the boolean 'game_over' to a float (1.0 or 0.0)
                dones = torch.tensor([1.0 if x[4] else 0.0 for x in batch], dtype=torch.float32)

                current_q = brain(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q = brain(next_states).max(1)[0]
                
                # Now math works because 'dones' is a float
                target_q = rewards + (0.99 * next_q * (1.0 - dones))

                loss = loss_func(current_q, target_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update Epsilon
        if epsilon > epsilon_min:
            epsilon *= decay

        # Track Score
        all_scores.append(env.score)
        print(f"Game: {game}, Score: {env.score}, Epsilon: {epsilon:.2f}")

        # SAVE CHECKPOINT (Task 5)
        if game % CHECKPOINT_INTERVAL == 0:
            filename = f"checkpoints/brain_game_{game}.pth"
            torch.save(brain.state_dict(), filename)
            print(f"Saved checkpoint: {filename}")

    # Save Final Model
    torch.save(brain.state_dict(), "final_model.pth")
    print("Training Complete!")
    
    return all_scores

# --- PLOTTING FUNCTION (Task 4) ---
def plot_learning_curve(scores):
    plt.figure(figsize=(10, 5))
    plt.plot(scores)
    plt.title("Flappy Bird AI Learning Curve")
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.grid(True)
    plt.savefig("learning_curve.png") # Save graph as image
    print("Graph saved as 'learning_curve.png'")
    plt.show()

if __name__ == "__main__":
    scores = train()
    plot_learning_curve(scores)