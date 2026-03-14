#!/bin/bash
# Master download script: fetch all datasets and generate dummy data
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================"
echo "  Dataset Download & Generation"
echo "======================================"
echo ""

# Download publicly available datasets
echo "--- Downloading NAB ---"
bash "$SCRIPT_DIR/download_nab.sh" || echo "WARNING: NAB download failed, skipping."
echo ""

echo "--- Downloading UCR ---"
bash "$SCRIPT_DIR/download_ucr.sh" || echo "WARNING: UCR download failed, skipping."
echo ""

echo "--- Downloading SMAP/MSL ---"
bash "$SCRIPT_DIR/download_smap_msl.sh" || echo "WARNING: SMAP/MSL download failed, skipping."
echo ""

echo "--- Downloading SMD ---"
bash "$SCRIPT_DIR/download_smd.sh" || echo "WARNING: SMD download failed, skipping."
echo ""

echo "--- MBA Info ---"
bash "$SCRIPT_DIR/download_mba.sh" || echo "WARNING: MBA info failed, skipping."
echo ""

# Generate dummy data for restricted datasets
echo "--- Generating Dummy SWaT ---"
python "$SCRIPT_DIR/generate_dummy_swat.py" || echo "WARNING: SWaT dummy generation failed."
echo ""

echo "--- Generating Dummy WADI ---"
python "$SCRIPT_DIR/generate_dummy_wadi.py" || echo "WARNING: WADI dummy generation failed."
echo ""

echo "======================================"
echo "  Summary"
echo "======================================"
echo ""
echo "Available datasets:"
for dir in data/raw/*/; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f | wc -l)
        echo "  $(basename $dir): $count files"
    fi
done
echo ""
echo "For restricted datasets (SWaT, WADI), register at iTrust:"
echo "  https://itrust.sutd.edu.sg/itrust-labs_datasets/"
