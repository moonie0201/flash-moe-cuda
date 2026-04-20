#!/bin/bash
# make_gif.sh — convert MP4 recording to optimized GIF for GitHub
# Usage: bash make_gif.sh input.mp4 output.gif [width=1200]

INPUT="${1:-/tmp/flash_moe_demo.mp4}"
OUTPUT="${2:-demo.gif}"
WIDTH="${3:-1200}"

set -e

if [ ! -f "$INPUT" ]; then
  echo "Error: input file not found: $INPUT"
  exit 1
fi

echo "Converting $INPUT → $OUTPUT (width=${WIDTH}px)..."

# Two-pass: generate palette first for better GIF color quality
PALETTE="/tmp/gif_palette.png"

ffmpeg -y -i "$INPUT" \
  -vf "fps=10,scale=${WIDTH}:-1:flags=lanczos,palettegen=stats_mode=diff" \
  "$PALETTE"

ffmpeg -y -i "$INPUT" -i "$PALETTE" \
  -filter_complex "fps=10,scale=${WIDTH}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" \
  "$OUTPUT"

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "Done: $OUTPUT ($SIZE)"
echo ""
echo "Add to README.md:"
echo "  ![demo]($OUTPUT)"
