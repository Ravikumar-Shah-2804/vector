#!/bin/bash
# Download UCR Time Series Anomaly Archive
set -e

DATA_DIR="data/raw/UCR"
mkdir -p "$DATA_DIR"

echo "=== UCR Time Series Anomaly Archive ==="
echo ""
echo "The full UCR archive requires manual download from:"
echo "  https://www.cs.ucr.edu/~eamonn/time_series_data_2018/"
echo ""
echo "After downloading, extract selected IDs (135-138) to:"
echo "  $DATA_DIR/"
echo ""
echo "Expected structure:"
echo "  $DATA_DIR/135_*.txt"
echo "  $DATA_DIR/136_*.txt"
echo "  $DATA_DIR/137_*.txt"
echo "  $DATA_DIR/138_*.txt"
echo ""

if [ "$(ls -A $DATA_DIR/ 2>/dev/null)" ]; then
    echo "Some UCR data already exists at $DATA_DIR."
else
    echo "No UCR data found. Please download manually."
fi
