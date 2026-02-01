import pygame
import random
import numpy as np

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PIPE_GAP = 200  # HUGE GAP
PIPE_FREQUENCY = 1500

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 30)
        self.reset()

    def reset(self):
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_velocity = 0
        self.gravity = 0.25  # MOON GRAVITY (Very slow falling)
        self.jump_strength = -5 # Gentle jump
        
        self.pipes = []
        self.score = 0
        self.game_over = False
        
        # Spawn first pipe
        pipe_height = random.randint(150, 350)
        self.pipes.append({"x": SCREEN_WIDTH, "height": pipe_height, "passed": False})
        
        return self._get_state()

    def step(self, action):
        if action == 1:
            self.bird_velocity = self.jump_strength
            
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        
        # Move Pipes
        for pipe in self.pipes:
            pipe["x"] -= 4 
        
        if self.pipes and self.pipes[-1]["x"] < SCREEN_WIDTH - 250:
             pipe_height = random.randint(150, 350)
             self.pipes.append({"x": SCREEN_WIDTH, "height": pipe_height, "passed": False})

        if self.pipes and self.pipes[0]["x"] < -50:
            self.pipes.pop(0)

        self.game_over = False
        reward = 0.1
        
        # --- THE TEACHER REWARD ---
        # We hold the AI's hand. 
        # If it is below the gap, Reward for Jumping.
        # If it is above the gap, Reward for Waiting.
        target_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + 50 > 50:
                target_pipe = pipe
                break
        
        if target_pipe:
            gap_center = target_pipe["height"] + (PIPE_GAP // 2)
            
            # Distance Logic
            if self.bird_y > gap_center: # Bird is too low
                if action == 1: reward = 1.0 # GOOD BOY! You jumped!
                else: reward = -1.0 # BAD! You should have jumped!
            else: # Bird is too high
                if action == 0: reward = 1.0 # GOOD BOY! You waited!
                else: reward = -1.0 # BAD! Don't jump!

        # Collisions
        bird_rect = pygame.Rect(50, int(self.bird_y), 30, 30)
        
        if self.bird_y >= SCREEN_HEIGHT - 30 or self.bird_y <= 0:
            self.game_over = True
            reward = -10
            
        for pipe in self.pipes:
            top = pygame.Rect(pipe["x"], 0, 50, pipe["height"])
            bot = pygame.Rect(pipe["x"], pipe["height"] + PIPE_GAP, 50, SCREEN_HEIGHT)
            
            if bird_rect.colliderect(top) or bird_rect.colliderect(bot):
                self.game_over = True
                reward = -10

            if not pipe["passed"] and pipe["x"] < 50:
                self.score += 1
                reward = 10
                pipe["passed"] = True

        return self._get_state(), reward, self.game_over

    def _get_state(self):
        # We give the AI the exact vertical distance to the target
        target_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + 50 > 50:
                target_pipe = pipe
                break
        
        if not target_pipe:
             return np.array([0.0, 0.0], dtype=np.float32)

        gap_center = target_pipe["height"] + (PIPE_GAP // 2)
        # Positive = Bird is Below Gap (Needs Jump)
        # Negative = Bird is Above Gap (Needs Wait)
        diff_y = (self.bird_y - gap_center) / SCREEN_HEIGHT
        
        # Simplified Input: Only 2 numbers needed!
        # 1. Vertical Distance to Gap
        # 2. Bird Velocity
        return np.array([diff_y, self.bird_velocity/10.0], dtype=np.float32)

    def render(self):
        # Graphics enabled!
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit()
        self.screen.fill((20, 20, 40)) # Dark Blue Space Background
        pygame.draw.rect(self.screen, (255, 255, 0), (50, int(self.bird_y), 30, 30))
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe["x"], 0, 50, pipe["height"]))
            pygame.draw.rect(self.screen, (0, 255, 0), (pipe["x"], pipe["height"] + PIPE_GAP, 50, SCREEN_HEIGHT))
        score = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score, (10, 10))
        pygame.display.flip()