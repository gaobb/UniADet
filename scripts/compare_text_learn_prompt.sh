# Compare WinCLIP text-prompt deltas with UniADet learned cls/seg weights.
source "$(dirname "$0")/env.sh"

python tools/compare_text_learn_prompt.py
