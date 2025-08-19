#!/bin/bash

start_processes() {
    rm -fr image_logs/*
    # rm -fr logs/*
    rm -fr wandb/*
#    /media/carla/carla/CARLA_0.9.13/CarlaUE4.sh -carla-port=4732 -prefernvidia -windowed -vulkan -RenderOffScreen > /dev/null 2>&1 &
#     carla_pid=$!
    
    #/media/carla/carla/CARLA_0.9.13/CarlaUE4.sh -carla-port=4736 -prefernvidia -windowed -vulkan -RenderOffScreen > /dev/null 2>&1 &
    # carla_pid_eval=$!

    python3 ./train.py --models_dir "./models" &
    train_pid=$!

    # python3 ./Evaluator.py &
    # evaluation_pid=$!

    echo 2222222222
    # python3 ./Evaluator.py &
    # evaluation_pid=$!

}

kill_processes() {
    
    # pkill -f "/media/carla/carla/CARLA_0.9.13/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4"

    pkill -f "./train.py"
    pkill -f "./Evaluator.py"


    pkill -f "wandb"

}
exting() {
    kill_processes
    # pkill -f "./file_keeper.py"
    exit 0
}
# check_python() {
    
# }
# check_carla() {
#     while true; do
#         if ! ps -p $carla_pid > /dev/null; then
#             echo "Carla stopped. Restarting Carla processes..."
#             /media/carla/carla/CARLA_0.9.13/CarlaUE4.sh -carla-port=4732 -prefernvidia -windowed -vulkan -RenderOffScreen > /dev/null 2>&1 &
#             carla_pid=$!
#         fi
#         sleep 5
#     done
# }
# check_eval_carla() {
#     while true; do
#         if  then
#             echo "Eval Carla stopped. Restarting Eval Carla processes..."
#             /media/carla/carla/CARLA_0.9.13/CarlaUE4.sh -carla-port=4736 -prefernvidia -windowed -vulkan -RenderOffScreen > /dev/null 2>&1 &
#             carla_pid_eval=$!
#         fi
#         sleep 5
#     done
# }
trap 'exting' SIGINT
kill_processes
pkill -f "./train.py"

# rm -fr ./evaluation_logs/*
cp ./models/trained.zip ./models/trained_backup.zip
rm -i ./models/trained.zip
cp ./models/my_buffer.pkl ./models/my_buffer_backup.pkl
mkdir logs
mkdir image_logs
# Start the processes initially
start_processes


# python3 ./file_keeper.py &
while true; do
    if ! ps -p $train_pid > /dev/null; then
        kill_processes
        sleep 5
        echo "Trained stopped. Restarting Trained processes..."
        start_processes
    fi
    sleep 5
done
