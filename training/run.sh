#!/bin/bash

start_processes() {
    rm -fr image_logs/*

    python3 ./train.py --models_dir "./models" &
    train_pid=$!


    echo 2222222222

}

kill_processes() {
    

    pkill -f "./train.py"
    pkill -f "./Evaluator.py"


    pkill -f "wandb"

}
exting() {
    kill_processes
    exit 0
}

trap 'exting' SIGINT
kill_processes
pkill -f "./train.py"

cp ./models/trained.zip ./models/trained_backup.zip
rm -i ./models/trained.zip
cp ./models/my_buffer.pkl ./models/my_buffer_backup.pkl
mkdir logs
mkdir image_logs
rm -fr image_logs/*

python3 ./train.py --models_dir "./models"
# Start the processes initially
# start_processes


# while true; do
#     if ! ps -p $train_pid > /dev/null; then
#         kill_processes
#         sleep 5
#         echo "Trained stopped. Restarting Trained processes..."
#         start_processes
#     fi
#     sleep 5
# done
