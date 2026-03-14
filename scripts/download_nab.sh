#!/bin/bash
# Download NAB (Numenta Anomaly Benchmark) dataset
set -e

DATA_DIR="data/raw/NAB"
REPO_DIR="data/raw/NAB_repo"

if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR/*.csv 2>/dev/null)" ]; then
    echo "NAB data already exists at $DATA_DIR, skipping."
    exit 0
fi

echo "Downloading NAB dataset..."

# Clone the NAB repository
if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/numenta/NAB.git "$REPO_DIR"
fi

# Copy relevant files
mkdir -p "$DATA_DIR"
cp "$REPO_DIR"/data/realKnownCause/*.csv "$DATA_DIR/"
cp "$REPO_DIR"/labels/combined_labels.json "$DATA_DIR/labels.json"

echo "NAB dataset downloaded to $DATA_DIR"
echo "  $(ls "$DATA_DIR"/*.csv | wc -l) CSV files"
