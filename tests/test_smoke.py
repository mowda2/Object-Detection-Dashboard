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


def test_library_db_import():
    """Test that library database module can be imported."""
    from moe_yolo_pipeline.moe_yolo_pipeline import library_db
    assert library_db is not None
    assert hasattr(library_db, 'LibraryDB')


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


def test_templates_exist():
    """Test that required template files exist."""
    from moe_yolo_pipeline.moe_yolo_pipeline import web_video_bridge
    
    template_dir = os.path.join(os.path.dirname(web_video_bridge.__file__), 'templates')
    
    required_templates = ['index.html', 'offline.html', 'library.html']
    
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
