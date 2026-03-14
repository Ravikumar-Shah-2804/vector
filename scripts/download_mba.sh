#!/bin/bash
# Download MBA (MIT-BIH Arrhythmia) dataset
set -e

DATA_DIR="data/raw/MBA"
mkdir -p "$DATA_DIR"

echo "=== MBA (MIT-BIH Arrhythmia) Dataset ==="
echo ""
echo "MBA dataset requires manual download."
echo ""
echo "Expected files in $DATA_DIR/:"
echo "  train.xlsx  - Training ECG data"
echo "  test.xlsx   - Test ECG data with anomalies"
echo "  labels.xlsx - Anomaly labels for test data"
echo ""

if [ "$(ls -A $DATA_DIR/ 2>/dev/null)" ]; then
    echo "Some MBA data already exists at $DATA_DIR."
else
    echo "No MBA data found. Please download manually."
fi
