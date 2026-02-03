# WiDS Mentorship Program - Final Project Submission

**Student Name:** Goswami Dhruvilgiri 
**College:** IIT Bombay  
**Domain:** Artificial Intelligence / Reinforcement Learning  
**Project:** Autonomous Flappy Bird Agent using Deep Q-Learning (DQN)

---

## 📌 Project Overview
This repository contains the code and resources developed during the WiDS (Women in Data Science) Mentorship Program. The project journey spans from foundational Machine Learning concepts to building a fully autonomous Reinforcement Learning agent capable of playing Flappy Bird.

The core of this repository is the **Final Project (Week 7-8)**: A Deep Q-Network (DQN) agent trained from scratch to master a custom-built Flappy Bird environment.

---

## Repository Structure

The repository is organized by weekly milestones:

* **`Week_1_3_Basics/`**: 
  * Focus: Python Fundamentals & Data Analysis.
  * Contents: Jupyter Notebooks (`.ipynb`) covering Python basics, libraries (NumPy, Pandas), and exploratory data analysis tasks.
  
* **`Week_5_Code/`**: 
  * Focus: Tabular Reinforcement Learning.
  * Contents: Implementation of **Gridworld**, **SARSA**, and **Q-Learning** algorithms with performance comparison graphs.

* **`Week_7_8_Flappy_Bird/` (Final Project)**: 
  * Focus: Deep Reinforcement Learning (DQN).
  * Contents: The complete source code for the Flappy Bird AI, including the game engine, training loop, and trained models.

## Final Project: Flappy Bird AI

### Key Features
* **Custom Game Engine:** Built a lightweight clone of Flappy Bird using `pygame` optimized for high-speed training.
* **Deep Q-Network (DQN):** Implemented a Neural Network with Experience Replay to approximate Q-values.
* **Smart State Space:** The agent "sees" the world using 4 engineered features:
  1. Vertical Distance to Top Pipe (Headroom)
  2. Vertical Distance to Bottom Pipe (Legroom)
  3. Bird Velocity
  4. Horizontal Distance to Next Pipe
*https://github.com/dhruvilgoswami35-dot/Flappy-learner-Wids_/ **Reward Shaping:** Implemented a "Hot/Cold" reward system to accelerate convergence by 3x compared to sparse rewards.
* **Dual Modes:** Includes scripts for both **Training** (learning from scratch) and **Evaluation** (watching the trained agent play).

### 🛠️ Installation & Setup

1. Clone the repository:
   git clone https://github.com/dhruvilgoswami35-dot/Flappy-learner-Wids_.git
   cd Flappy-learner-Wids_/Week_7_8_Flappy_Bird
Install dependencies:

pip install -r requirements.txt
(Required libraries: torch, pygame, numpy, matplotlib)

🚀 How to Run
1. Train the Agent (from scratch)
To see the AI learn from trial and error:
python train_ai.py
This will run 500 episodes.

It saves checkpoints in the checkpoints/ folder.

It generates a learning_curve.png graph at the end.

2. Watch the Agent Play (Demo)
To see the fully trained agent in action:
python evaluate_ai.py
Loads the final_model.pth file.

Runs the game in standard speed (human-viewable).
