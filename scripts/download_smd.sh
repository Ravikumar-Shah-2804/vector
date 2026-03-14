#!/bin/bash
# Download SMD (Server Machine Dataset)
set -e

DATA_DIR="data/raw/SMD"
REPO_DIR="data/raw/OmniAnomaly_repo"

if [ -d "$DATA_DIR" ] && [ "$(ls -A $DATA_DIR/ 2>/dev/null)" ]; then
    echo "SMD data already exists at $DATA_DIR, skipping."
    exit 0
fi

echo "Downloading SMD dataset..."

# Clone OmniAnomaly repo which contains SMD
if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/NetManAIOps/OmniAnomaly.git "$REPO_DIR"
fi

mkdir -p "$DATA_DIR"

# Copy ServerMachineDataset contents
if [ -d "$REPO_DIR/ServerMachineDataset" ]; then
    cp -r "$REPO_DIR/ServerMachineDataset/"* "$DATA_DIR/"
    echo "SMD dataset downloaded to $DATA_DIR"
    echo "  $(ls "$DATA_DIR"/train/ 2>/dev/null | wc -l) train files"
    echo "  $(ls "$DATA_DIR"/test/ 2>/dev/null | wc -l) test files"
else
    echo "ERROR: ServerMachineDataset not found in OmniAnomaly repo."
    echo "Please check the repository structure."
    exit 1
fi
