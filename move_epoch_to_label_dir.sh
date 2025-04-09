#!/bin/bash

SOURCE_DIR="/Users/wachiii/Workschii/brain-mdd/data/dataset/predct/eyeopen_16Channels/epoch20sFiles"

HC_DIR="$SOURCE_DIR/hc"
MDD_DIR="$SOURCE_DIR/mdd"

mkdir -p "$HC_DIR"
mkdir -p "$MDD_DIR"

for file in "$SOURCE_DIR"/*_HC*; do
    if [ -f "$file" ]; then
        echo "Moving $file to $HC_DIR"
        mv "$file" "$HC_DIR"
    fi
done

for file in "$SOURCE_DIR"/*_MDD*; do
    if [ -f "$file" ]; then
        echo "Moving $file to $MDD_DIR"
        mv "$file" "$MDD_DIR"
    fi
done

echo "File organization complete."