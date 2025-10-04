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
import copy
import pickle
import time
from datetime import datetime

timestamp = int(time.time())
dt_object = datetime.fromtimestamp(timestamp)
formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
logdir = f"logs/{formatted_time}/"


# Assuming you already loaded your model and env
# model = My_TQC.load(..., env=env)
argparser = argparse.ArgumentParser()
argparser.add_argument('--models_dir', type=str, default=None)
args = argparser.parse_args()

# Set up log directory for TensorBoard


os.makedirs(logdir,exist_ok=True)
models_dir = args.models_dir

world_manager = World_Manager()


env = CarEnv(world_manager,evaluation=True,imagelog_stride=5,logdir=logdir)


model = My_TQC.load(f"{models_dir}/best/best_model.zip", env=env, verbose=1,tensorboard_log=logdir)

last_iteration = 0
try:
    try:
        for i in range(last_iteration,2000):
            last_iteration = i
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
            print(f"Episode {i+1} reward:", total_reward)
    except Exception as e:
        print(f"Ran into an Error: {e}")
        world_manager.__init__()

except KeyboardInterrupt:
    print("Recived interruption, logging and exiting")
finally:

    print("Finall logging of the table. Rest assured if you see this.")

