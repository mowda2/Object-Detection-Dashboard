# MoeWS - Multiview YOLO Dashboard

A local web-based analysis interface for computer vision pipelines, supporting both **offline video analysis** and **live ROS 2 streams**.

![Dashboard](docs/screenshots/dashboard.png)

## Features

- **Offline Analysis Mode** (macOS & Linux)

  - Upload video files for YOLO object detection
  - Object tracking with ByteTrack
  - Speed estimation (configurable meters/pixel)
  - Export results as CSV and JSON
  - Analysis library with search and playback

- **Roboflow Hosted Inference** (NEW)

  - Use Roboflow's cloud API for YOLO inference
  - No local GPU required - inference happens on Roboflow servers
  - Frame-by-frame video analysis with progress tracking
  - Interactive viewer with detection overlays
  - Export detections to JSON and CSV formats
  - Configurable frame skip for usage optimization

- **Live ROS 2 Mode** (Linux only)
  - Multi-camera support with auto-discovery
  - Real-time YOLO inference overlay
  - Stream annotated frames to browser
  - Launch manager for ROS nodes

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/mowda2/MoeWS.git
cd MoeWS

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard

**Option A: Using the run script (recommended)**

```bash
./scripts/run_dashboard.sh
```

**Option B: Manual startup**

```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m moe_yolo_pipeline.moe_yolo_pipeline.web_video_bridge
```

### 3. Open in Browser

- **Main Dashboard**: http://localhost:5000
- **Offline Analysis**: http://localhost:5000/offline
- **Roboflow Inference**: http://localhost:5000/roboflow
- **Analysis Library**: http://localhost:5000/library

## Project Structure

```
MoeWS/
├── src/
│   └── moe_yolo_pipeline/           # ROS 2 package root
│       ├── moe_yolo_pipeline/       # Python package
│       │   ├── web_video_bridge.py  # Flask app (ENTRY POINT)
│       │   ├── offline_routes.py    # Offline analysis routes
│       │   ├── offline_analyzer.py  # YOLO + tracking logic
│       │   ├── roboflow_routes.py   # Roboflow hosted inference
│       │   ├── roboflow_client.py   # Roboflow API client
│       │   ├── library_db.py        # SQLite database
│       │   ├── yolo_inference_node.py    # ROS node
│       │   ├── templates/           # HTML templates
│       │   └── static/              # CSS/JS assets
│       ├── launch/                  # ROS 2 launch files
│       ├── package.xml              # ROS 2 package manifest
│       └── setup.py                 # ROS 2 setup
├── scripts/
│   ├── run_dashboard.sh             # Start script
│   └── setup.sh                     # Environment setup
├── requirements.txt
├── yolo11n.pt                       # YOLO model (download separately)
└── README.md
```

## Runtime Modes

### Offline Mode (macOS & Linux)

Automatically activated when ROS 2 is not installed. Features:

- Video file upload and analysis
- Frame-by-frame YOLO inference
- Object tracking and speed estimation
- Results saved to SQLite library

### ROS 2 Mode (Linux only)

Activated when `rclpy` is available. Additional features:

- Live camera discovery (`/dev/video*`)
- Multi-camera YOLO inference nodes
- Real-time overlay streaming
- Launch manager integration

## Configuration

### Offline Analysis Parameters

| Parameter     | Default      | Description                        |
| ------------- | ------------ | ---------------------------------- |
| Model         | `yolo11n.pt` | YOLO model file                    |
| Confidence    | `0.25`       | Detection confidence threshold     |
| Meters/pixel  | `0.05`       | Scale factor for speed calculation |
| Device        | `cpu`        | Inference device (`cpu` or `cuda`) |
| Track classes | Vehicles     | Which object classes to track      |

### Environment Variables

```bash
PYTHONPATH="$PWD/src"  # Required for module imports
FLASK_DEBUG=1          # Enable debug mode (optional)

# Roboflow Hosted Inference (optional)
ROBOFLOW_API_KEY="your_api_key"    # Get from roboflow.com/settings
ROBOFLOW_MODEL="your-model-id"     # e.g., "coco/1" or "your-project/2"
ROBOFLOW_VERSION="1"               # Model version number
```

### Roboflow Setup

To use the Roboflow Hosted Inference feature:

1. **Create a Roboflow account** at [roboflow.com](https://roboflow.com)
2. **Get your API key** from [Settings > API Keys](https://app.roboflow.com/settings/api)
3. **Choose a model**:
   - Use a public model like `coco/1` (COCO-trained YOLO)
   - Or train your own model on Roboflow and use your project ID
4. **Set environment variables**:
   ```bash
   export ROBOFLOW_API_KEY="your_api_key_here"
   export ROBOFLOW_MODEL="coco"  # or your project name
   export ROBOFLOW_VERSION="1"
   ```
5. **Restart the dashboard** and navigate to `/roboflow`

**Usage Notes:**

- Roboflow API has usage limits based on your plan
- Use "Frame Skip" to reduce API calls (e.g., skip 5 = process every 5th frame)
- Results are cached locally in `runs/roboflow/` directory

## Dependencies

Core requirements (see `requirements.txt`):

- `ultralytics` - YOLO inference
- `opencv-python` - Video processing
- `supervision` - Object tracking (ByteTrack)
- `flask` - Web server
- `numpy` - Array operations

Optional (for ROS 2 mode):

- `rclpy` - ROS 2 Python client
- `cv_bridge` - ROS/OpenCV bridge
- `sensor_msgs` - Image messages

## Development

### Running Tests

```bash
source venv/bin/activate
export PYTHONPATH="$PWD/src"
pytest tests/ -v
```

### Code Style

```bash
# Check with flake8
flake8 src/moe_yolo_pipeline/moe_yolo_pipeline/

# Format with black
black src/moe_yolo_pipeline/moe_yolo_pipeline/
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'moe_yolo_pipeline'"

Make sure `PYTHONPATH` includes the `src/` directory:

```bash
export PYTHONPATH="$PWD/src"
```

### "ImportError: attempted relative import with no known parent package"

Don't run the script directly. Use the module syntax:

```bash
# Wrong:
python src/moe_yolo_pipeline/moe_yolo_pipeline/web_video_bridge.py

# Correct:
python -m moe_yolo_pipeline.moe_yolo_pipeline.web_video_bridge
```

### Dashboard starts but ROS features don't work

This is expected on macOS. ROS 2 mode requires Linux with ROS 2 Humble installed.

## License

MIT License - See LICENSE file for details.
