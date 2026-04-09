#!/bin/bash
set -e

echo "====== Code Review Environment ======"

# Start the FastAPI server in the background on port 8000
echo "Starting FastAPI server on port 8000..."
python server.py &
SERVER_PID=$!

# Wait for the server to be ready
echo "Waiting for server to start..."
for i in $(seq 1 30); do
    if python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; then
        echo "Server is ready!"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "WARNING: Server process exited. Continuing with direct inference..."
        break
    fi
    sleep 1
done

# Run inference
echo "Running inference.py..."
python inference.py
EXIT_CODE=$?

# Cleanup
if kill -0 $SERVER_PID 2>/dev/null; then
    kill $SERVER_PID 2>/dev/null || true
fi

exit $EXIT_CODE
