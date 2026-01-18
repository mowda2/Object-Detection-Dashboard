"""
Smoke tests for MoeWS YOLO Dashboard

These tests verify that:
1. Core modules can be imported
2. Flask app initializes correctly
3. Offline mode works without ROS
4. Database operations work
"""

import sys
import os

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_package_import():
    """Test that the main package can be imported."""
    import moe_yolo_pipeline
    assert moe_yolo_pipeline is not None


def test_web_video_bridge_import():
    """Test that web_video_bridge module can be imported."""
    from moe_yolo_pipeline.moe_yolo_pipeline import web_video_bridge
    assert web_video_bridge is not None
    assert hasattr(web_video_bridge, 'app')


def test_flask_app_exists():
    """Test that Flask app is created."""
    from moe_yolo_pipeline.moe_yolo_pipeline.web_video_bridge import app
    assert app is not None
    assert app.name == 'moe_yolo_pipeline.moe_yolo_pipeline.web_video_bridge'


def test_offline_routes_import():
    """Test that offline routes can be imported."""
    from moe_yolo_pipeline.moe_yolo_pipeline import offline_routes
    assert offline_routes is not None
    assert hasattr(offline_routes, 'offline_bp')


def test_offline_analyzer_import():
    """Test that offline analyzer can be imported."""
    from moe_yolo_pipeline.moe_yolo_pipeline import offline_analyzer
    assert offline_analyzer is not None
    assert hasattr(offline_analyzer, 'run_offline_speed_job')


def test_offline_analyzer_helpers():
    """Test that offline analyzer helper functions work correctly."""
    from moe_yolo_pipeline.moe_yolo_pipeline import offline_analyzer
    
    # Test _format_label produces clean ASCII-only output
    label = offline_analyzer._format_label("person", 42, 0.87, 25.5)
    assert "person" in label
    assert "#42" in label
    assert "0.87" in label
    assert "25.5" in label or "km/h" in label
    # Ensure no weird characters (bullet points, etc.)
    assert "•" not in label
    assert all(ord(c) < 128 for c in label), "Label should be ASCII-only"
    
    # Test _box_area calculation
    area = offline_analyzer._box_area([0, 0, 100, 50])
    assert area == 5000
    
    # Test _center_xyxy calculation
    cx, cy = offline_analyzer._center_xyxy([10, 20, 30, 40])
    assert cx == 20.0
    assert cy == 30.0
    
    # Test env var helpers exist and return defaults
    assert offline_analyzer._env_float("NONEXISTENT_VAR", 1.5) == 1.5
    assert offline_analyzer._env_int("NONEXISTENT_VAR", 10) == 10
    assert offline_analyzer._env_bool("NONEXISTENT_VAR", False) == False


def test_library_db_import():
    """Test that library database module can be imported."""
    from moe_yolo_pipeline.moe_yolo_pipeline import library_db
    assert library_db is not None
    assert hasattr(library_db, 'LibraryDB')


def test_roboflow_client_import():
    """Test that Roboflow client module can be imported."""
    from moe_yolo_pipeline.moe_yolo_pipeline import roboflow_client
    assert roboflow_client is not None
    assert hasattr(roboflow_client, 'RoboflowClient')
    assert hasattr(roboflow_client, 'Detection')


def test_roboflow_routes_import():
    """Test that Roboflow routes can be imported."""
    from moe_yolo_pipeline.moe_yolo_pipeline import roboflow_routes
    assert roboflow_routes is not None
    assert hasattr(roboflow_routes, 'roboflow_bp')


def test_ros_availability_flag():
    """Test that ROS availability is correctly detected."""
    from moe_yolo_pipeline.moe_yolo_pipeline import web_video_bridge
    # On macOS or without ROS, this should be False
    # On Linux with ROS, this should be True
    assert isinstance(web_video_bridge.ROS_AVAILABLE, bool)


def test_flask_routes_registered():
    """Test that expected routes are registered in Flask app."""
    from moe_yolo_pipeline.moe_yolo_pipeline.web_video_bridge import app
    
    # Get list of registered routes
    rules = [rule.rule for rule in app.url_map.iter_rules()]
    
    # Check core routes exist
    assert '/' in rules
    assert '/offline' in rules
    assert '/library' in rules
    assert '/api/library' in rules
    # Roboflow routes (Blueprint adds trailing slash)
    assert '/roboflow/' in rules or '/roboflow' in rules


def test_templates_exist():
    """Test that required template files exist."""
    from moe_yolo_pipeline.moe_yolo_pipeline import web_video_bridge
    
    template_dir = os.path.join(os.path.dirname(web_video_bridge.__file__), 'templates')
    
    required_templates = [
        'index.html', 
        'offline.html', 
        'library.html',
        'roboflow/index.html',
        'roboflow/viewer.html',
    ]
    
    for template in required_templates:
        template_path = os.path.join(template_dir, template)
        assert os.path.exists(template_path), f"Missing template: {template}"


def test_static_files_exist():
    """Test that static files directory exists."""
    from moe_yolo_pipeline.moe_yolo_pipeline import web_video_bridge
    
    static_dir = os.path.join(os.path.dirname(web_video_bridge.__file__), 'static')
    assert os.path.isdir(static_dir), "Static directory missing"


def test_library_db_operations():
    """Test basic LibraryDB operations with in-memory database."""
    import tempfile
    from moe_yolo_pipeline.moe_yolo_pipeline.library_db import LibraryDB
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False) as f:
        db_path = f.name
    
    try:
        db = LibraryDB(db_path)
        
        # Test file upsert
        db.upsert_file('testhash123', 'test.mp4', 1024, '/tmp/test.mp4')
        
        # Test file retrieval
        row = db.get_file('testhash123')
        assert row is not None
        assert row['filename'] == 'test.mp4'
        
        # Test listing (should be empty for analyses)
        analyses = db.list_analyses()
        assert isinstance(analyses, list)
        
    finally:
        os.unlink(db_path)


def test_pass2_track_stats():
    """Test PASS 2 TrackStats collector functionality."""
    from moe_yolo_pipeline.moe_yolo_pipeline.offline_analyzer import TrackStats
    
    ts = TrackStats()
    
    # Initially empty
    assert ts.unique_tracks == 0
    
    # Add some track updates
    ts.update(track_id=1, frame_idx=1, class_name="car", confidence=0.9, box_area=5000)
    ts.update(track_id=2, frame_idx=1, class_name="person", confidence=0.8, box_area=3000)
    ts.update(track_id=1, frame_idx=2, class_name="car", confidence=0.85, box_area=5100)
    
    # Check counts
    assert ts.unique_tracks == 2
    assert ts.active_tracks_in_frame([1, 2]) == 2
    assert ts.active_tracks_in_frame([1, 999]) == 1  # 999 doesn't exist
    
    # Export summary
    summary = ts.export_summary("test.mp4", 30.0, 100)
    assert summary["video"] == "test.mp4"
    assert summary["fps"] == 30.0
    assert summary["frames_processed"] == 100
    assert summary["unique_tracks"] == 2
    assert len(summary["tracks"]) == 2
    
    # Check track 1 stats (appeared twice)
    track1 = next(t for t in summary["tracks"] if t["track_id"] == 1)
    assert track1["class"] == "car"
    assert track1["start_frame"] == 1
    assert track1["end_frame"] == 2
    assert track1["frames_seen"] == 2
    assert track1["max_conf"] == 0.9  # max of 0.9, 0.85


def test_pass2_hud_env_var():
    """Test PASS 2 HUD env var functions exist."""
    from moe_yolo_pipeline.moe_yolo_pipeline import offline_analyzer
    
    # These should exist and be callable
    assert callable(offline_analyzer.MOE_SHOW_HUD)
    assert callable(offline_analyzer.MOE_WRITE_CSV_V2)
    
    # Default should be False (disabled by default)
    assert offline_analyzer.MOE_SHOW_HUD() == False
    assert offline_analyzer.MOE_WRITE_CSV_V2() == False
    
    # _draw_hud function should exist
    assert hasattr(offline_analyzer, '_draw_hud')
    assert callable(offline_analyzer._draw_hud)


if __name__ == '__main__':
    # Run tests manually
    import traceback
    
    tests = [
        test_package_import,
        test_web_video_bridge_import,
        test_flask_app_exists,
        test_offline_routes_import,
        test_offline_analyzer_import,
        test_library_db_import,
        test_roboflow_client_import,
        test_roboflow_routes_import,
        test_ros_availability_flag,
        test_flask_routes_registered,
        test_templates_exist,
        test_static_files_exist,
        test_library_db_operations,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
