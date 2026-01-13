"""
Roboflow Hosted Inference Client

This module provides a clean interface to Roboflow's hosted inference API.
It is completely isolated from the existing Ultralytics/Supervision pipeline.

Environment variables required:
    ROBOFLOW_API_KEY    - Your Roboflow API key
    ROBOFLOW_MODEL      - Model ID (e.g., "coco/1" or "workspace/model")
    
Optional environment variables:
    ROBOFLOW_CONFIDENCE - Confidence threshold (default: 0.4)
    ROBOFLOW_OVERLAP    - IOU/overlap threshold (default: 0.3)
"""

import os
import io
import base64
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Load .env file if python-dotenv is available, otherwise try manual loading
def _load_dotenv():
    """Load .env file from project root."""
    # Try python-dotenv first
    try:
        from dotenv import load_dotenv
        # Look for .env in project root (several levels up from this file)
        here = os.path.dirname(__file__)
        for _ in range(5):  # Go up to 5 levels
            env_path = os.path.join(here, ".env")
            if os.path.exists(env_path):
                load_dotenv(env_path)
                return
            here = os.path.dirname(here)
    except ImportError:
        pass
    
    # Fallback: manually parse .env file
    here = os.path.dirname(__file__)
    for _ in range(5):
        env_path = os.path.join(here, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ.setdefault(key.strip(), value.strip())
            return
        here = os.path.dirname(here)

_load_dotenv()

# Lazy import requests to avoid adding hard dependency
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

import cv2
import numpy as np


@dataclass
class Detection:
    """Normalized detection format for internal use."""
    class_name: str
    class_id: Optional[int]
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }


class RoboflowClientError(Exception):
    """Base exception for Roboflow client errors."""
    pass


class RoboflowConfigError(RoboflowClientError):
    """Raised when configuration is missing or invalid."""
    pass


class RoboflowAPIError(RoboflowClientError):
    """Raised when API call fails."""
    pass


class RoboflowClient:
    """
    Client for Roboflow Hosted Inference API.
    
    Usage:
        client = RoboflowClient()
        if client.is_configured():
            detections = client.infer_image(frame_bgr)
    """
    
    # Roboflow Serverless API base URL (works with Universe models)
    API_BASE = "https://serverless.roboflow.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        confidence: Optional[float] = None,
        overlap: Optional[float] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        """
        Initialize Roboflow client.
        
        Args:
            api_key: Roboflow API key (or set ROBOFLOW_API_KEY env var)
            model: Model ID like "coco/1" (or set ROBOFLOW_MODEL env var)
            confidence: Confidence threshold 0-1 (default: 0.4)
            overlap: IOU overlap threshold 0-1 (default: 0.3)
            timeout: Request timeout in seconds
            max_retries: Number of retries on transient failures
        """
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "")
        self.model = model or os.environ.get("ROBOFLOW_MODEL", "")
        self.confidence = confidence or float(os.environ.get("ROBOFLOW_CONFIDENCE", "0.4"))
        self.overlap = overlap or float(os.environ.get("ROBOFLOW_OVERLAP", "0.3"))
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Statistics
        self._total_calls = 0
        self._total_errors = 0
        self._last_call_time = 0.0
    
    def is_configured(self) -> bool:
        """Check if client has required configuration."""
        return bool(self.api_key and self.model and REQUESTS_AVAILABLE)
    
    def get_config_status(self) -> Dict[str, Any]:
        """Get detailed configuration status for UI display."""
        return {
            "requests_available": REQUESTS_AVAILABLE,
            "api_key_set": bool(self.api_key),
            "model_set": bool(self.model),
            "model": self.model if self.model else None,
            "confidence": self.confidence,
            "overlap": self.overlap,
            "is_configured": self.is_configured(),
            "total_calls": self._total_calls,
            "total_errors": self._total_errors,
        }
    
    def get_missing_config(self) -> List[str]:
        """Get list of missing configuration items."""
        missing = []
        if not REQUESTS_AVAILABLE:
            missing.append("requests library not installed (pip install requests)")
        if not self.api_key:
            missing.append("ROBOFLOW_API_KEY environment variable not set")
        if not self.model:
            missing.append("ROBOFLOW_MODEL environment variable not set")
        return missing
    
    def _encode_image(self, image_bgr: np.ndarray, quality: int = 90) -> str:
        """Encode OpenCV BGR image to base64 JPEG string."""
        # Encode to JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image_bgr, encode_params)
        if not success:
            raise RoboflowClientError("Failed to encode image to JPEG")
        
        # Convert to base64
        return base64.b64encode(buffer).decode("utf-8")
    
    def _parse_response(self, response_json: Dict[str, Any], img_width: int, img_height: int) -> List[Detection]:
        """Parse Roboflow API response into normalized Detection objects."""
        detections = []
        
        predictions = response_json.get("predictions", [])
        for pred in predictions:
            # Roboflow returns center x, y, width, height
            cx = pred.get("x", 0)
            cy = pred.get("y", 0)
            w = pred.get("width", 0)
            h = pred.get("height", 0)
            
            # Convert to x1, y1, x2, y2
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            detection = Detection(
                class_name=pred.get("class", "unknown"),
                class_id=pred.get("class_id"),
                confidence=float(pred.get("confidence", 0)),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            detections.append(detection)
        
        return detections
    
    def infer_image(self, image_bgr: np.ndarray) -> Tuple[List[Detection], Dict[str, Any]]:
        """
        Run inference on a single image using Roboflow Hosted API.
        
        Args:
            image_bgr: OpenCV BGR image (numpy array)
            
        Returns:
            Tuple of (list of Detection objects, metadata dict)
            
        Raises:
            RoboflowConfigError: If client is not configured
            RoboflowAPIError: If API call fails
        """
        if not REQUESTS_AVAILABLE:
            raise RoboflowConfigError("requests library not installed")
        if not self.api_key:
            raise RoboflowConfigError("ROBOFLOW_API_KEY not set")
        if not self.model:
            raise RoboflowConfigError("ROBOFLOW_MODEL not set")
        
        img_height, img_width = image_bgr.shape[:2]
        
        # Build API URL
        url = f"{self.API_BASE}/{self.model}"
        params = {
            "api_key": self.api_key,
            "confidence": self.confidence,
            "overlap": self.overlap,
        }
        
        # Encode image
        image_b64 = self._encode_image(image_bgr)
        
        # Make request with retries
        last_error = None
        start_time = time.time()
        
        for attempt in range(self.max_retries + 1):
            try:
                self._total_calls += 1
                
                response = requests.post(
                    url,
                    params=params,
                    data=image_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=self.timeout,
                )
                
                self._last_call_time = time.time() - start_time
                
                if response.status_code == 401:
                    raise RoboflowAPIError("Invalid API key (401 Unauthorized)")
                elif response.status_code == 404:
                    raise RoboflowAPIError(f"Model not found: {self.model} (404)")
                elif response.status_code == 429:
                    raise RoboflowAPIError("Rate limit exceeded (429). Please wait and try again.")
                elif response.status_code != 200:
                    raise RoboflowAPIError(f"API error: {response.status_code} - {response.text[:200]}")
                
                response_json = response.json()
                detections = self._parse_response(response_json, img_width, img_height)
                
                metadata = {
                    "inference_time": response_json.get("time", 0),
                    "image_width": img_width,
                    "image_height": img_height,
                    "model": self.model,
                    "num_detections": len(detections),
                }
                
                return detections, metadata
                
            except requests.exceptions.Timeout:
                last_error = RoboflowAPIError(f"Request timed out after {self.timeout}s")
                self._total_errors += 1
            except requests.exceptions.ConnectionError as e:
                last_error = RoboflowAPIError(f"Connection error: {e}")
                self._total_errors += 1
            except requests.exceptions.RequestException as e:
                last_error = RoboflowAPIError(f"Request failed: {e}")
                self._total_errors += 1
            
            # Brief delay before retry
            if attempt < self.max_retries:
                time.sleep(0.5 * (attempt + 1))
        
        raise last_error or RoboflowAPIError("Unknown error")
    
    def estimate_api_calls(self, total_frames: int, frame_skip: int = 1) -> int:
        """Estimate number of API calls for a video."""
        return (total_frames + frame_skip - 1) // frame_skip


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Detection],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw detection bounding boxes on an image.
    
    Args:
        image_bgr: OpenCV BGR image
        detections: List of Detection objects
        box_color: BGR color for boxes
        text_color: BGR color for text
        thickness: Line thickness
        font_scale: Font scale for labels
        
    Returns:
        Annotated image (copy of original)
    """
    annotated = image_bgr.copy()
    
    for det in detections:
        x1, y1 = int(det.x1), int(det.y1)
        x2, y2 = int(det.x2), int(det.y2)
        
        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
        
        # Prepare label
        label = f"{det.class_name} {det.confidence:.2f}"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 4, y1),
            box_color,
            -1,  # Filled
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
        )
    
    return annotated


# Singleton instance for convenience
_default_client: Optional[RoboflowClient] = None


def get_client() -> RoboflowClient:
    """Get or create the default Roboflow client instance."""
    global _default_client
    if _default_client is None:
        _default_client = RoboflowClient()
    return _default_client
