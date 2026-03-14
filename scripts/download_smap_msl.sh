#!/bin/bash
# Download SMAP and MSL datasets (NASA telemetry)
set -e

DATA_DIR="data/raw/SMAP_MSL"
REPO_DIR="data/raw/telemanom_repo"

if [ -d "$DATA_DIR/train" ] && [ -d "$DATA_DIR/test" ]; then
    echo "SMAP/MSL data already exists at $DATA_DIR, skipping."
    exit 0
fi

echo "Downloading SMAP and MSL datasets..."

# Clone telemanom for labeled_anomalies.csv
if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/khundman/telemanom.git "$REPO_DIR"
fi

mkdir -p "$DATA_DIR/train" "$DATA_DIR/test"

# Copy labels
cp "$REPO_DIR/labeled_anomalies.csv" "$DATA_DIR/"

# Download .npy files from the telemanom data directory
# SMAP and MSL data are stored in the repo's data/ folder
if [ -d "$REPO_DIR/data/train" ]; then
    cp "$REPO_DIR"/data/train/*.npy "$DATA_DIR/train/" 2>/dev/null || true
    cp "$REPO_DIR"/data/test/*.npy "$DATA_DIR/test/" 2>/dev/null || true
fi

echo "SMAP/MSL dataset downloaded to $DATA_DIR"
echo "  Train files: $(ls "$DATA_DIR/train/"*.npy 2>/dev/null | wc -l)"
echo "  Test files: $(ls "$DATA_DIR/test/"*.npy 2>/dev/null | wc -l)"
echo ""
echo "Note: If .npy files are missing, download them from:"
echo "  https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
echo "  Extract train/ and test/ folders into $DATA_DIR/"
