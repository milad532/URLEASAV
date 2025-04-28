#!/bin/bash

# Path to the Python script
PYTHON_SCRIPT="/media/external_10T/milad/carla_project/inference/worker_fairness.py"
# Log file to track errors or crashes
LOG_FILE="/media/external_10T/milad/carla_project/inference/running_log/logfile.log"
# Set the timer interval (in seconds) for checking if the script is running
CHECK_INTERVAL=60
# Maximum retry attempts after a crash (to avoid infinite loop)
MAX_RETRIES=5
RETRY_COUNT=0
# Port to check for processes
PORT=4040

# Function to check if Python script is running
is_script_running() {
  pgrep -f "$PYTHON_SCRIPT" > /dev/null 2>&1
}

# Function to kill processes using the specified port
kill_processes_using_port() {
  echo "Checking for processes using port $PORT..." >> "$LOG_FILE"
  # Find the process ID (PID) using port 4040
  PID=$(lsof -t -i :$PORT)

  if [ -n "$PID" ]; then
    echo "Killing process(es) with PID: $PID using port $PORT..." >> "$LOG_FILE"
    kill -9 $PID
    echo "Processes on port $PORT have been killed." >> "$LOG_FILE"
  else
    echo "No processes found using port $PORT." >> "$LOG_FILE"
  fi
}

# Function to check for segmentation fault
check_segfault() {
  # Check the exit status of the last run process (Python script)
  if [ $? -eq 139 ]; then
    echo "Segmentation fault (Signal 11) detected." >> "$LOG_FILE"
    return 1  # Indicate a crash occurred
  fi
  return 0  # No crash
}

# Function to run the Python script and handle crashes
run_python_script() {
  kill_processes_using_port  # Kill processes using port 4040 before starting
  python3 "$PYTHON_SCRIPT" &  # Run the Python script in the background
  PYTHON_PID=$!  # Capture the PID of the Python process
  echo "Started Python script with PID $PYTHON_PID at $(date)" >> "$LOG_FILE"
}

# Main loop
while true; do
  # Check if the Python script is running and log the status
  if ! is_script_running; then
    if [[ $RETRY_COUNT -ge $MAX_RETRIES ]]; then
      echo "Python script has crashed $MAX_RETRIES times. Stopping attempts." >> "$LOG_FILE"
      break  # Exit if we reach max retries
    fi
    echo "Python script not running. Attempting to start..." >> "$LOG_FILE"
    run_python_script  # Start the Python script
    RETRY_COUNT=$((RETRY_COUNT + 1))
  else
    echo "Python script is running normally." >> "$LOG_FILE"
  fi

  # Wait for the Python script to complete or crash
  wait $PYTHON_PID  # Wait for the Python process to exit

  # Check if the Python script ended due to a segmentation fault
  if ! check_segfault; then
    echo "Segmentation fault detected. Restarting Python script..." >> "$LOG_FILE"
    kill_processes_using_port  # Kill processes using port 4040
    run_python_script  # Restart the Python script
  fi

  # Wait for the next check
  sleep "$CHECK_INTERVAL"
done