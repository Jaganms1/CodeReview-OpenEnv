#!/bin/bash
echo "====== Running Code Review Environment ======"
python inference.py 2>&1 | tee /app/results.txt
echo ""
echo "====== Inference Complete ======"
echo "Results saved to /app/results.txt"
echo "Container staying alive for Space..."

# Keep container alive so HF Spaces shows "Running"
tail -f /dev/null
