#!/bin/bash
# =============================================================================
# setup.sh - Set up the development environment
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "Setting up YOLO Dashboard development environment..."
echo ""

# Check for system dependencies
echo "Checking system dependencies..."

# Check for ffmpeg (required for browser-compatible video encoding)
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg not found (required for video playback in Chrome/Firefox)"
    
    # Try to install based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            echo "   Installing ffmpeg via Homebrew..."
            brew install ffmpeg
            echo "✓ Installed ffmpeg"
        else
            echo "   Please install ffmpeg: brew install ffmpeg"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "   Installing ffmpeg via apt..."
            sudo apt-get update && sudo apt-get install -y ffmpeg
            echo "✓ Installed ffmpeg"
        elif command -v dnf &> /dev/null; then
            echo "   Installing ffmpeg via dnf..."
            sudo dnf install -y ffmpeg
            echo "✓ Installed ffmpeg"
        else
            echo "   Please install ffmpeg manually"
        fi
    else
        echo "   Please install ffmpeg for your system"
    fi
else
    echo "✓ ffmpeg found: $(which ffmpeg)"
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]] && [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Created venv/"
fi

# Activate venv
if [[ -d "venv" ]]; then
    source venv/bin/activate
elif [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

# Install requirements
if [[ -f "requirements.txt" ]]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    echo "✓ Installed requirements"
fi

# Download YOLO model if not present
if [[ ! -f "yolo11n.pt" ]]; then
    echo "Downloading YOLO v11 nano model..."
    if command -v wget &> /dev/null; then
        wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
    elif command -v curl &> /dev/null; then
        curl -sLO https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
    else
        echo "⚠️  Please download yolo11n.pt manually from:"
        echo "   https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    fi
    if [[ -f "yolo11n.pt" ]]; then
        echo "✓ Downloaded yolo11n.pt"
    fi
else
    echo "✓ YOLO model found: yolo11n.pt"
fi

# Create data directories (gitignored)
mkdir -p data/videos
mkdir -p data/outputs
mkdir -p logs

echo ""
echo "✓ Setup complete!"
echo ""
echo "To run the dashboard:"
echo "  ./scripts/run_dashboard.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  export PYTHONPATH=\"\$PWD/src\""
echo "  python -m moe_yolo_pipeline.moe_yolo_pipeline.web_video_bridge"
