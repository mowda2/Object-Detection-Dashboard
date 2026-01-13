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
