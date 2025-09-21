#!/bin/bash

start_processes() {
    rm -fr image_logs/*
    # rm -fr logs/*
    rm -fr wandb/*

    python3 ./inference.py --models_dir "../training/models" &
    inference_pid=$!

    # python3 ./Evaluator.py &
    # evaluation_pid=$!

    echo 2222222222

}

kill_processes() {
    
    pkill -f "./inference.py"

}
exting() {
    kill_processes
    # pkill -f "./file_keeper.py"
    exit 0
}

trap 'exting' SIGINT
kill_processes
pkill -f "./inference.py"

# rm -fr ./evaluation_logs/*
mkdir logs
mkdir image_logs
# Start the processes initially
rm -fr image_logs/*
# rm -fr logs/*
rm -fr wandb/*
python3 ./inference.py --models_dir "../training/models"
