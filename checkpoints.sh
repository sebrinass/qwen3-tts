#!/bin/bash
# Download Qwen3-TTS checkpoints from HuggingFace.
# Usage: ./checkpoints.sh [variant]
#   variant : tokenizer | 0.6b-base | 0.6b-customvoice | 1.7b-base
#             1.7b-customvoice | 1.7b-voicedesign | all (default)

set -eu

DIR="checkpoints"
mkdir -p "$DIR"

HF="hf download --quiet"

dl_repo() {
    local name="$1" repo="$2"
    local target="$DIR/$name"
    if [ -d "$target" ] && [ "$(ls "$target"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[OK] $name"
        return
    fi
    echo "[Download] $name <- $repo"
    $HF "$repo" --local-dir "$target"
}

variant="${1:-all}"

case "$variant" in
    tokenizer|all)
        dl_repo "Qwen3-TTS-Tokenizer-12Hz" "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        ;;&
    0.6b-base|all)
        dl_repo "Qwen3-TTS-12Hz-0.6B-Base" "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
        ;;&
    0.6b-customvoice|all)
        dl_repo "Qwen3-TTS-12Hz-0.6B-CustomVoice" "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        ;;&
    1.7b-base|all)
        dl_repo "Qwen3-TTS-12Hz-1.7B-Base" "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ;;&
    1.7b-customvoice|all)
        dl_repo "Qwen3-TTS-12Hz-1.7B-CustomVoice" "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ;;&
    1.7b-voicedesign|all)
        dl_repo "Qwen3-TTS-12Hz-1.7B-VoiceDesign" "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        ;;
    *)
        echo "Unknown variant: $variant"
        echo "Valid: tokenizer | 0.6b-base | 0.6b-customvoice | 1.7b-base | 1.7b-customvoice | 1.7b-voicedesign | all"
        exit 1
        ;;
esac
