#!/bin/bash
# record_demo.sh — 397B inference demo recorder
# Usage: bash record_demo.sh
# Requirements: ffmpeg, tmux
# Output: demo.gif (for GitHub README)

set -e

SESSION="flash_moe_demo"
OUTPUT_MP4="/tmp/flash_moe_demo.mp4"
OUTPUT_GIF="$(dirname "$0")/demo.gif"
PYTHON="/home/mh/ocstorage/workspace/ml_env/bin/python3"
SCRIPT="$(dirname "$0")/src/run_397b_ssd.py"
RECORD_SECONDS=120  # adjust as needed

INFERENCE_CMD="$PYTHON $SCRIPT \
  --2bit --float8-nonexpert \
  --gpu-hot-gb 10 --gpu-margin-gb 2.5 \
  --hot-pct 1.0 --cache-format raw \
  --tokens 50 \
  --prompt 'Explain how a Mixture of Experts model works.'"

echo "=== flash-moe demo recorder ==="
echo "Output: $OUTPUT_GIF"
echo ""

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create tmux session: left pane (inference) | right pane (nvidia-smi)
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux split-window -h -t "$SESSION"

# Right pane: GPU monitor
tmux send-keys -t "$SESSION:0.1" \
  "watch -n 0.5 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk -F\",\" \"{printf \\\"GPU%s | util:%3s%% | mem:%5sMB/%5sMB | temp:%s°C\\\\n\\\", \\\$1,\\\$3,\\\$4,\\\$5,\\\$6}\"'" Enter

sleep 2

# Left pane: inference
tmux send-keys -t "$SESSION:0.0" \
  "cd /home/mh/llm/flash-moe/ses && clear && echo '=== 397B MoE on 2x RTX 3060 ===' && sleep 1 && $INFERENCE_CMD" Enter

echo ""
echo "tmux session '$SESSION' started."
echo ""
echo "Now record your screen with one of these:"
echo ""
echo "  [ffmpeg — record display :0]"
echo "  ffmpeg -video_size 1920x1080 -framerate 15 -f x11grab -i :0 \\"
echo "    -t $RECORD_SECONDS $OUTPUT_MP4"
echo ""
echo "  [OBS / other screen recorder]"
echo "  Attach to tmux first: tmux attach -t $SESSION"
echo ""
echo "After recording, convert to GIF:"
echo "  bash $(dirname "$0")/make_gif.sh $OUTPUT_MP4 $OUTPUT_GIF"
echo ""
echo "Attach to session to watch live:"
echo "  tmux attach -t $SESSION"
