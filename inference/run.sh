#!/bin/bash
port=${1:-5050}
start_processes() {
    /media/carla/carla/CARLA_0.9.14/CarlaUE4.sh -carla-port=$port -prefernvidia -windowed -vulkan -RenderOffScreen -fps=20 > /dev/null 2>&1 &
    carla_pid=$!
    
    python3 /media/carla/AVRL/data_collect/worker_fairness.py --port=$port &
    train_pid=$!


}

kill_processes() {
    echo "/media/carla/carla/CARLA_0.9.14/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -carla-port=${port}"
    pkill -f "/media/carla/carla/CARLA_0.9.14/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 -carla-port=${port}"

    pkill -f "worker_fairness.py --port=${port}"
 
}
exting() {
    kill_processes
    exit 0
}

trap 'exting' SIGINT
kill_processes

start_processes


while true; do
    if ! ps -p $train_pid > /dev/null || ! ps -p $carla_pid > /dev/null; then
        kill_processes
        sleep 5
        echo "Trained stopped. Restarting Trained processes..."
        start_processes
    fi
    sleep 5
done

