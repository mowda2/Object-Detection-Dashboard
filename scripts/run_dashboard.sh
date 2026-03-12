#!/bin/bash
# =============================================================================
# run_dashboard.sh - Start the YOLO Dashboard (Offline or ROS mode)
# =============================================================================
# Usage:
#   ./scripts/run_dashboard.sh          # Auto-detect mode
#   ./scripts/run_dashboard.sh --help   # Show help
#
# On macOS: Runs in offline-only mode (no ROS)
# On Linux with ROS: Runs with full ROS 2 support
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_help() {
    echo "YOLO Dashboard - Start Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --port PORT    Set Flask port (default: 5000)"
    echo "  --debug        Enable Flask debug mode"
    echo ""
    echo "Environment:"
    echo "  The script will automatically:"
    echo "  - Activate venv if present"
    echo "  - Set PYTHONPATH to src/"
    echo "  - Detect ROS 2 availability"
    echo ""
    echo "URLs after startup:"
    echo "  Dashboard:     http://localhost:5000"
    echo "  Offline:       http://localhost:5000/offline"
    echo "  Library:       http://localhost:5000/library"
}

# Parse arguments
PORT=5000
DEBUG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            print_help
            exit 0
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       YOLO Dashboard Startup Script        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Check for virtual environment
if [[ -d "venv" ]]; then
    echo -e "${GREEN}✓${NC} Found venv/, activating..."
    source venv/bin/activate
elif [[ -d ".venv" ]]; then
    echo -e "${GREEN}✓${NC} Found .venv/, activating..."
    source .venv/bin/activate
else
    echo -e "${YELLOW}⚠${NC} No virtual environment found (venv/ or .venv/)"
    echo "  Proceeding with system Python..."
fi

# Set PYTHONPATH
export PYTHONPATH="$REPO_ROOT/src:$PYTHONPATH"
echo -e "${GREEN}✓${NC} PYTHONPATH set to include src/"

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1)
echo -e "${GREEN}✓${NC} Python: $PYTHON_VERSION"

# Check for ROS 2
if python3 -c "import rclpy" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} ROS 2 detected - running in FULL mode (offline + live)"
    MODE="ros"
else
    echo -e "${YELLOW}⚠${NC} ROS 2 not available - running in OFFLINE-ONLY mode"
    MODE="offline"
fi

echo ""
echo -e "${BLUE}Starting dashboard on port $PORT...${NC}"
echo -e "  Main:     ${GREEN}http://localhost:$PORT${NC}"
echo -e "  Offline:  ${GREEN}http://localhost:$PORT/offline${NC}"
echo -e "  Library:  ${GREEN}http://localhost:$PORT/library${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo ""

# Run the dashboard
if [[ -n "$DEBUG" ]]; then
    export FLASK_DEBUG=1
fi

python3 -m moe_yolo_pipeline.moe_yolo_pipeline.web_video_bridge
