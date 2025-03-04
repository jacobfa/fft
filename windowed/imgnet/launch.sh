#!/bin/bash
# launch.sh - Launch the distributed training inside a tmux session,
# recording terminal output to terminal.txt using the 'script' command.

SESSION="ddp_session"
OUTPUT_FILE="terminal.txt"

# Kill any existing session with the same name.
tmux kill-session -t "$SESSION" 2>/dev/null


# Launch a new detached tmux session that runs the command under 'script'.
# The -q flag makes script quiet and -f forces flushing after each write.
tmux new-session -d -s "$SESSION" "script -q -f ${OUTPUT_FILE} -c 'torchrun --nproc_per_node=8 train.py'"

echo "Launched tmux session '$SESSION'."
echo "To attach: tmux attach-session -t $SESSION"
echo "Terminal output is being logged to ${OUTPUT_FILE}."
