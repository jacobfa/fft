#!/bin/bash
# launch.sh - Launch distributed training for LRA benchmarks inside a tmux session,
# recording terminal output to a log file using the 'script' command.
#
# Usage: ./launch.sh <dataset>
#   where <dataset> is one of: imdb, listops, cifar10

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset>"
    echo "Datasets: imdb, listops, cifar10"
    exit 1
fi

DATASET=$1
SESSION="ddp_session_${DATASET}"
OUTPUT_FILE="terminal_${DATASET}.txt"

# Kill any existing tmux session with the same name.
tmux kill-session -t "$SESSION" 2>/dev/null

# (Optional) Set additional environment variables for distributed training.
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"

# Build the torchrun command with dataset-specific arguments.
if [ "$DATASET" == "imdb" ]; then
    CMD="torchrun --nproc_per_node=6 train.py --dataset imdb --max_length 1024 --seq_len 1024 --input_dim 1 --num_classes 2"
elif [ "$DATASET" == "listops" ]; then
    CMD="torchrun --nproc_per_node=6 train.py --dataset listops --max_length 1024 --seq_len 1024 --input_dim 1"
elif [ "$DATASET" == "cifar10" ]; then
    CMD="torchrun --nproc_per_node=6 train.py --dataset cifar10 --max_length 1024 --seq_len 1024 --input_dim 1 --num_classes 10"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

# Launch a new detached tmux session running the command under 'script'.
# The -q flag makes script quiet and -f forces flushing after each write.
tmux new-session -d -s "$SESSION" "script -q -f ${OUTPUT_FILE} -c '${CMD}'"

echo "Launched tmux session '$SESSION' with command:"
echo "${CMD}"
echo "To attach: tmux attach-session -t $SESSION"
echo "Terminal output is being logged to ${OUTPUT_FILE}."
