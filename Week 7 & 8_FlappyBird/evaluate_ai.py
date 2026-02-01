import torch
import pygame
import time
import os
from flappy_env import FlappyBirdEnv
from train_ai import SimpleBrain

def evaluate():
    env = FlappyBirdEnv()
    model = SimpleBrain()
    
    model_path = "final_model.pth" 
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Loaded {model_path}")
    else:
        print("Model not found! Run train_ai.py first.")
        return

    print("--- Starting Evaluation Run (Press ESC to Quit) ---")
    clock = pygame.time.Clock()
    
    for i in range(5):
        state_raw = env.reset()
        state = torch.FloatTensor(state_raw)
        game_over = False
        
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: return
            
            with torch.no_grad():
                q_values = model(state)
                action = torch.argmax(q_values).item()
            
            next_state_raw, reward, game_over = env.step(action)
            state = torch.FloatTensor(next_state_raw)
            
            env.render()
            clock.tick(30)
            
        print(f"Demo Game {i+1} Score: {env.score}")
        time.sleep(1)

if __name__ == "__main__":

    evaluate()
