from tqc import My_TQC,Obstacle1DCNNExtractor
# train.py
# from ppo import My_PPO,Obstacle1DCNNExtractor # <-- CHANGED
# from features_extractor import Obstacle1DCNNExtractor # <-- CHANGED
import torch
import os
from carenv import CarEnv, num_envs, World_Manager
import time
from datetime import datetime
import argparse
import numpy as np
import copy
import pickle
from pathlib import Path
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

if __name__ == '__main__':
    # ---- Setup Paths & Logging ----
    timestamp = int(time.time())
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    logdir = f"logs/{formatted_time}/"
    os.makedirs(logdir, exist_ok=True)

    # ---- Args ----
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--models_dir', type=str, default=None)
    args = argparser.parse_args()

    models_dir = args.models_dir or f"models/{timestamp}"
    os.makedirs(models_dir, exist_ok=True)
    best_dir = os.path.join(models_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    eval_log_dir = os.path.join(models_dir, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    
    # Paths for saving
    trained_zip = os.path.join(models_dir, "trained.zip")
    rb_path = os.path.join(models_dir, "replay_buffer.pkl") # TQC Needs Replay Buffer

    # ---- Environment ----
    world_manager = World_Manager()
    env = CarEnv(world_manager, imagelog_stride=20, logdir=logdir)
    eval_env = CarEnv(world_manager, evaluation=True)

    # ---- Hyperparams for TQC ----
    TIMESTEPS = 10_000
    TOTAL_ITERS = 100
    LR = 3e-4 # Standard TQC/SAC learning rate
    
    # Custom parameter for your My_TQC class
    KL_COEF = 0.1 

    # ---- Model Setup ----
    if not os.path.isfile(trained_zip):
        print("Creating new TQC model")
        model = My_TQC(
            "MultiInputPolicy", # or 'MlpPolicy' if inputs are flat, but use MultiInput for Dict obs
            env=env,
            learning_rate=LR,
            buffer_size=1_000_000,  # Important for off-policy
            batch_size=256,         # TQC usually benefits from larger batches (256 or 512)
            ent_coef="auto",        # Automatically tune entropy
            gamma=0.99,
            tau=0.005,              # Soft update coefficient
            train_freq=1,           # Update every step
            gradient_steps=1,       # One gradient step per env step
            learning_starts=1000,   # Warmup steps
            use_sde=True,           # gSDE is supported by TQC
            kl_coef=KL_COEF,        # <--- PASSING YOUR CUSTOM PARAMETER
            policy_kwargs=dict(
                log_std_init=-2,
                # TQC uses 'pi' (actor) and 'qf' (Q-function), NOT 'vf'
                net_arch=dict(pi=[256, 256], qf=[256, 256]), 
                features_extractor_class=Obstacle1DCNNExtractor,
                features_extractor_kwargs=dict(features_dim=48),
            ),
            verbose=1,
            tensorboard_log=logdir,
        )
    else:
        print("Loading existing TQC model")
        # Ensure you use custom_load if you overrode it to handle specific loading logic
        model = My_TQC.custom_load( 
            trained_zip,
            env=env,
            kl_coef=KL_COEF, # Pass this if it's not saved/loaded automatically by SB3 logic
            verbose=1,
            learning_rate=LR,
            tensorboard_log=logdir
        )
        
        # Load Replay Buffer (Crucial for TQC continuity)
        if os.path.isfile(rb_path):
            print("Loading replay buffer...")
            try:
                model.load_replay_buffer(rb_path)
                print(f"Replay buffer loaded with {model.replay_buffer.size()} transitions.")
            except Exception as e:
                print(f"Failed to load replay buffer: {e}")

    # ---- Callbacks ----
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=5_000,
        n_eval_episodes=10, # 50 might be too slow for frequent checks
        deterministic=True,
        render=False
    )

    # ---- Training Loop ----
    print(f'Starting training...')
    while model.num_timesteps < TOTAL_ITERS * TIMESTEPS:
        try:
            model.learn(
                total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                callback=[eval_cb],
            )
            
            # Save Model
            model.save(trained_zip)
            
            # Save Replay Buffer
            # Note: Replay buffers can be large (GBs). 
            # You might want to save this less frequently than the model.
            print("Saving replay buffer...")
            model.save_replay_buffer(rb_path)

        except Exception as e:
            print(f"Learn ran into a problem: {e}. Restarting world manager...")
            # Ideally, close the env properly before restarting
            try:
                env.close()
            except:
                pass
            
            # Re-initialize environment
            world_manager = World_Manager()
            env = CarEnv(world_manager, imagelog_stride=20, logdir=logdir)
            model.set_env(env)
            
        except KeyboardInterrupt:
            print("Interrupted by user. Saving and exiting.")
            model.save(trained_zip)
            model.save_replay_buffer(rb_path)
            break