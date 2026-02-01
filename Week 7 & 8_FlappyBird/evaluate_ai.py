import torch
import pygame
import time
import os
from flappy_env import FlappyBirdEnv
from train_ai import SimpleBrain

def evaluate():
    env = FlappyBirdEnv()
    model = SimpleBrain()
    
    # Load the best brain you trained
    # You can change this to "checkpoints/brain_game_200.pth" to test specific versions
    model_path = "final_model.pth" 
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set to evaluation mode (no learning)
        print(f"Loaded {model_path}")
    else:
        print("Model not found! Run train_ai.py first.")
        return

    print("--- Starting Evaluation Run (Press ESC to Quit) ---")
    clock = pygame.time.Clock()
    
    for i in range(5): # Play 5 demo games
        state_raw = env.reset()
        state = torch.FloatTensor(state_raw)
        game_over = False
        
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
            
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item() # Always pick Best Action
            
            next_state_raw, reward, game_over = env.step(action)
            state = torch.FloatTensor(next_state_raw)
            
            env.render()
            clock.tick(30) # 30 FPS for viewing
            
        print(f"Demo Game {i+1} Score: {env.score}")
        time.sleep(1)

if __name__ == "__main__":
    evaluate()