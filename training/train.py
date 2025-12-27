# train.py
from trpo import My_TRPO, Obstacle1DCNNExtractor # <-- CHANGED
import torch
import os
from carenv import CarEnv, num_envs, World_Manager
import time
from datetime import datetime
import argparse
import numpy as np
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
    TIMESTEPS = 100_000
    LR = 1e-3 # Learning rate for the value function in TRPO
    TOTAL_ITERS = 1000
    
    # ---- Create or load model ----
    trained_zip = os.path.join(models_dir, "trained.zip")
    # TRPO is on-policy and does not use a replay buffer, so this remains commented out.
    # rb_path = os.path.join(models_dir, "replay_buffer.pkl") 

    if not os.path.isfile(trained_zip):
        print("Creating new TRPO model") # <-- CHANGED
        # --- MODEL DEFINITION CHANGED FROM PPO to TRPO ---
        model = My_TRPO(
            'MlpPolicy',
            device='cpu',
            env=env,
            n_steps=2048,          # Same as PPO: Number of steps to collect for each update
            gamma=0.99,            # Discount factor
            learning_rate=LR,      # TRPO: Learning rate for the value function optimizer
            target_kl=0.01,        # TRPO: Target KL divergence constraint
            use_sde=True,
            # sde_sample_freq is not a TRPO parameter
            policy_kwargs=dict(
                log_std_init=-2,
                # TRPO policies also have separate networks for actor (pi) and critic (vf)
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                # The feature extractor remains the same
                features_extractor_class=Obstacle1DCNNExtractor,
                features_extractor_kwargs=dict(features_dim=48),
            ),
            verbose=1,
            tensorboard_log=logdir,
        )
    else:
        print("Loading existing TRPO model") # <-- CHANGED
        model = My_TRPO.load( # <-- CHANGED
            trained_zip,
            device='cpu',
            env=env,
            verbose=1,
            learning_rate=LR,
            tensorboard_log=logdir
        )
        env.TOTAL_TIMESTEPS = model.num_timesteps
    
    # The evaluation callback remains unchanged.
    eval_log_file = os.path.join(eval_log_dir, "evaluations.npz")
    if os.path.isfile(eval_log_file):
        data = np.load(eval_log_file)
        if "results" in data:
            prev_best = float(np.max(np.mean(data["results"], axis=1)))
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=10_000,
        n_eval_episodes=50,
        deterministic=True,
        render=False
    )

    # ---- Training loop ----
    # The training loop remains the same.
    print(f'New iteration starting...')
    while model.num_timesteps < TOTAL_ITERS * TIMESTEPS:
        #try:
        model.learn(
                total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                callback=[eval_cb],
            )

            # Still save locally if you want
        model.save(trained_zip)

        #except Exception as e:
            
            #print(f"Learn ran into a problem: {e}. Restarting world.")
            #time.sleep(2.0)
            #world_manager.__init__()
            # try:
            # # Properly clean up the current environment
            #     env.close()
            #     eval_env.close()
            # except:
            #     pass
        
            # # Reinitialize world manager and environments
            # world_manager = World_Manager()
            # env = CarEnv(world_manager, imagelog_stride=20, logdir=logdir)
            # eval_env = CarEnv(world_manager, evaluation=True)
            
            # # Update the model's environment
            # model.set_env(env)
            
            # # Optional: wait a moment for CARLA to stabilize
            # time.sleep(2.0)
