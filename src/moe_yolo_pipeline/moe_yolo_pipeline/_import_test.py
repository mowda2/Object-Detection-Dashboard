"""Temporary import test — delete after verification."""
try:
    from .counting import CountingLine, CrossingEvent, LineCrossingDetector
    print("  counting.py: OK")
except Exception as e:
    print(f"  counting.py: FAIL - {e}")

try:
    from .turn_movements import TurnMovement, CompletedMovement, TurnMovementClassifier
    print("  turn_movements.py: OK")
except Exception as e:
    print(f"  turn_movements.py: FAIL - {e}")

try:
    from .interval_binning import IntervalAggregator, TrafficStudyResult
    print("  interval_binning.py: OK")
except Exception as e:
    print(f"  interval_binning.py: FAIL - {e}")

try:
    from .report_generator import generate_tmc_pdf, generate_utdf_csv
    print("  report_generator.py: OK")
except Exception as e:
    print(f"  report_generator.py: FAIL - {e}")

try:
    from .speed_analyzer import run_speed_job, get_default_settings
    print("  speed_analyzer.py: OK")
except Exception as e:
    print(f"  speed_analyzer.py: FAIL - {e}")

try:
    from .web_video_bridge import app
    print("  web_video_bridge.py: OK")
except Exception as e:
    print(f"  web_video_bridge.py: FAIL - {e}")
