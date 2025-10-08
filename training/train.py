# from tqc import My_TQC,Obstacle1DCNNExtractor
# train.py
from ppo import My_PPO,Obstacle1DCNNExtractor # <-- CHANGED
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
    global env
    timestamp = int(time.time())
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    logdir = f"logs/{formatted_time}/"

    world_manager = World_Manager()
    env = CarEnv(world_manager,imagelog_stride=20,logdir=logdir)

    eval_env = CarEnv(world_manager,evaluation=True)
    print("passing here 2")
    # ---- Args ----
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--models_dir', type=str, default=None)
    args = argparser.parse_args()

    # ---- Paths ----
    os.makedirs(logdir, exist_ok=True)
    models_dir = args.models_dir or f"models/{timestamp}"
    os.makedirs(models_dir, exist_ok=True)

    best_dir = os.path.join(models_dir, "best")
    ckpt_dir = os.path.join(models_dir, "checkpoints")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    eval_log_dir = os.path.join(models_dir, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    trained_zip = os.path.join(models_dir, "trained.zip")

    # ---- Training hyperparams ----
    TIMESTEPS = 10_000
    LR = 7.3e-4 # PPO is often more sensitive to LR, you might need to use a smaller value like 3e-4
    TOTAL_ITERS = 50
    
    # ---- Create or load model ----
    trained_zip = os.path.join(models_dir, "trained.zip")
    # rb_path = os.path.join(models_dir, "replay_buffer.pkl") # <-- REMOVED: PPO is on-policy and does not use a replay buffer

    if not os.path.isfile(trained_zip):
        print("Creating new PPO model") # <-- CHANGED
        # --- MODEL DEFINITION CHANGED FROM TQC to PPO ---
        model = My_PPO(
            'MlpPolicy',
            env=env,
            n_steps=2048,          # PPO: Number of steps to collect for each update
            batch_size=64,         # PPO: Minibatch size for training
            n_epochs=10,           # PPO: Number of optimization epochs per update
            learning_rate=LR,
            gamma=0.99,
            use_sde=True,
            sde_sample_freq=-1,    # PPO: -1 means sample new noise matrix at each update
            policy_kwargs=dict(
                log_std_init=-2,
                # PPO policies can have separate networks for actor (pi) and critic (vf)
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                # The feature extractor remains the same
                features_extractor_class=Obstacle1DCNNExtractor,
                features_extractor_kwargs=dict(features_dim=48),
            ),
            verbose=1,
            tensorboard_log=logdir,
        )
    else:
        print("Loading existing PPO model") # <-- CHANGED
        model = My_PPO.load( # <-- CHANGED
            trained_zip,
            env=env,
            verbose=1,
            learning_rate=LR,
            tensorboard_log=logdir
        )
        env.TOTAL_TIMESTEPS = model.num_timesteps
        # --- REPLAY BUFFER LOADING REMOVED ---
        if os.path.isfile(rb_path):
            with open(rb_path, "rb") as f:
                model.replay_buffer = pickle.load(f)

    eval_log_file = os.path.join(eval_log_dir, "evaluations.npz")
    if os.path.isfile(eval_log_file):
        data = np.load(eval_log_file)
        if "results" in data:
            prev_best = float(np.max(np.mean(data["results"], axis=1)))
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=1000,
        n_eval_episodes=50,
        deterministic=True,
        render=False
    )

    # ---- Training loop ----
    print(f'New iteration starting...')
    try:
        while model.num_timesteps < TOTAL_ITERS * TIMESTEPS:
            # try:
            model.learn(
                total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                callback=[eval_cb],
            )

            # Still save locally if you want
            model.save(trained_zip)
            # --- REPLAY BUFFER SAVING REMOVED ---
            model.save_replay_buffer(rb_path)

            # except Exception as e:
            #     print(f"Learn ran into a problem: {e}. Restarting world.")
            #     world_manager.__init__()
    except KeyboardInterrupt:
        print("Recived interruption, logging and exiting")
    finally:
        print("Finall logging of the table. Rest assured if you see this.")