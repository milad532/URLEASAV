from tqc import My_TQC,Obstacle1DCNNExtractor
import torch
import os
from carenv import CarEnv, num_envs, World_Manager
import time
import argparse
import numpy as np
import copy
import pickle
from pathlib import Path
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
import wandb
# wandb.login()



        
if __name__ == '__main__':
    global env
    world_manager = World_Manager()
    env = CarEnv(world_manager)
    eval_env = CarEnv(world_manager,evaluation=True)

    # ---- Args ----
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--models_dir', type=str, default=None)
    args = argparser.parse_args()

    # ---- Paths ----
    timestamp = int(time.time())
    logdir = f"logs/{timestamp}/"
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
    TIMESTEPS = 20_000
    LR = 7.3e-4
    TOTAL_ITERS = 50
    # ---- Start W&B run (metrics only, no model saving) ----
    os.environ['WANDB_BASE_URL'] = 'http://127.0.0.1:8080'
    run = wandb.init(
        project="FAVRL",
        entity='teimuri',
        name=env.log_name,
        id=f"tqc_{timestamp}", 
        # mode="offline",
        config=dict(
            algo="TQC",
            lr=LR,
            timesteps_per_iter=TIMESTEPS,
            total_iters=TOTAL_ITERS,
        ),
        sync_tensorboard=True,  # picks up SB3's TB logs
        save_code=True,
    )
    wandb.define_metric("*", step_metric="global_step")

    # ---- Create or load model ----
    trained_zip = os.path.join(models_dir, "trained.zip")
    rb_path = os.path.join(models_dir, "replay_buffer.pkl")

    if not os.path.isfile(trained_zip):
        print("Creating new model")
        model = My_TQC(
            'MlpPolicy',
            env=env,
            buffer_size=200_000,
            batch_size=256,
            train_freq=200,
            gradient_steps=256,
            learning_starts=500,
            use_sde=True,
            sde_sample_freq=16,
            policy_kwargs=dict(
                log_std_init=-3,
                net_arch=[256, 256],     # applies after the extractor output
                n_critics=2,
                use_expln=True,
                # 👇 add these 2 lines
                features_extractor_class=Obstacle1DCNNExtractor,
                features_extractor_kwargs=dict(features_dim=256),
            ),
            verbose=1,
            learning_rate=LR,
            gamma=0.99,
            tensorboard_log=logdir,
        )
    else:
        print("Loading existing model")
        model = My_TQC.load(
            trained_zip,
            env=env,
            verbose=1,
            learning_rate=LR,
            tensorboard_log=logdir
        )
        env.TOTAL_TIMESTEPS = model.num_timesteps
        if os.path.isfile(rb_path):
            with open(rb_path, "rb") as f:
                model.replay_buffer = pickle.load(f)

    # ---- W&B callback (no model saving) ----
    prev_best = -1000
    eval_log_file = os.path.join(eval_log_dir, "evaluations.npz")
    if os.path.isfile(eval_log_file):
        data = np.load(eval_log_file)
        # data["results"] shape: (n_evals, n_episodes)
        if "results" in data:
            prev_best = float(np.max(np.mean(data["results"], axis=1)))
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path=eval_log_dir,
        eval_freq=600,          # evaluate every N training steps
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    eval_cb.best_mean_reward = prev_best

    wandb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_freq=0,   # 🔑 disables checkpoint saving to W&B
        model_save_path = os.path.join(models_dir,"wandb_model"),
        verbose=1,
        log="all",
    )

    # ---- Training loop ----
    for iters in range(TOTAL_ITERS):
        if model.num_timesteps > TOTAL_ITERS * TIMESTEPS:
            break

        print(f'Iteration {iters} starting...')
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback=[eval_cb,wandb_cb],   # logs metrics only
        )

        # Still save locally if you want
        model.save(trained_zip)
        model.save_replay_buffer(rb_path)

        # Custom W&B logs
        wandb.log({
            "iteration": iters,
            "num_timesteps": model.num_timesteps,
        })

    wandb.finish()
