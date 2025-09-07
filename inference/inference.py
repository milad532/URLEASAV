import sys
import os


sys.path.append(os.path.expanduser("../training"))

from tqc import My_TQC
import torch
import os
from carenv import CarEnv,num_envs,World_Manager
import time
import argparse
import numpy as np
import shutil
from multiprocessing import shared_memory
from wandb.integration.sb3 import WandbCallback
import wandb
import copy
import pickle



import gym

# Assuming you already loaded your model and env
# model = My_TQC.load(..., env=env)
argparser = argparse.ArgumentParser()
argparser.add_argument('--models_dir', type=str, default=None)
args = argparser.parse_args()

# Set up log directory for TensorBoard
logdir = f"logs/{int(time.time())}/"
if not os.path.exists(logdir):
    os.makedirs(logdir)
models_dir = args.models_dir

world_manager = World_Manager()
# Initialize environment
world_manager.start_world()

env = CarEnv(world_manager)
TIMESTEPS = 20000  # Total number of steps per training iteration
LR = 7.3e-4

model = My_TQC.load(f"{models_dir}/trained.zip", env=env, verbose=1, learning_rate=LR, tensorboard_log=logdir)

# model.log_ent_coef = torch.log(torch.ones(1, device=model.device) * 0.1).requires_grad_(True)
# model.ent_coef_optimizer = torch.optim.Adam([model.log_ent_coef], lr=model.lr_schedule(1))


replay_buffer = pickle.load(open(f"{models_dir}/replay_buffer.pkl", "rb"))
model.replay_buffer = replay_buffer
# model.replay_buffer.avg_rewards = replay_buffer.avg_rewards
# model.replay_buffer.num_saved = replay_buffer.num_saved
del replay_buffer

obs = env.reset()
done = False
total_reward = 0.0
for i in range(100):
    try:
        
        while not done:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=True)  
            # deterministic=True → no exploration noise, good for evaluation
            
            # Step in the environment
            _ , reward, done, info, _ = env.step(action)
            total_reward += reward
            # env.render()  # optional, for visualization

        print("Episode reward:", total_reward)
    except:
        continue