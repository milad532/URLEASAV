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
import time
timestamp = int(time.time())


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


env = CarEnv(world_manager,evaluation=True,imagelog_stride=1)

os.environ['WANDB_BASE_URL'] = 'http://127.0.0.1:8080'
run = wandb.init(
    project="FAVRL-tests",
    entity='teimuri',
    name=f'{env.log_name}_test',
    id=f"tqc_{timestamp}", 
    # mode="offline",
    config=dict(
        algo="TQC",
    ),
    sync_tensorboard=True,  # picks up SB3's TB logs
    save_code=True,
)
wandb.define_metric("*", step_metric="global_step")

model = My_TQC.load(f"{models_dir}/best/best_model.zip", env=env, verbose=1)



for i in range(1000):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
        
    while not done:
        # Get action from the model
        action, _states = model.predict(obs, deterministic=True)  
        # deterministic=True → no exploration noise, good for evaluation
        
        # Step in the environment
        obs , reward, done, info, _ = env.step(action)
        total_reward += reward
        # env.render()  # optional, for visualization
    wandb.log({"Total Reward": total_reward,"global_step":i})
    print(f"Episode {i+1} reward:", total_reward)
wandb.finish()

