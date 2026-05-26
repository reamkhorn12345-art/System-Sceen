#!/usr/bin/env python3
"""
Test script to verify that all modules can be imported and initialized correctly.
"""

def test_imports():
    """Test that all modules can be imported."""
    try:
        import camera
        import face_detector
        import hud
        import config
        print("+ All modules imported successfully")
        return True
    except Exception as e:
        print(f"- Import error: {e}")
        return False

def test_classes():
    """Test that classes can be instantiated."""
    try:
        import config
        import camera
        import face_detector
        # Test config
        assert hasattr(config, 'CAMERA_WIDTH')
        assert hasattr(config, 'CAMERA_HEIGHT')
        print("+ Config constants accessible")
        
        # Test camera class
        cam = camera.Camera()
        print("+ Camera class instantiated")
        
        # Test face detector class
        fd = face_detector.FaceDetector()
        print("+ FaceDetector class instantiated")
        
        # Test HUD class (just importing since it's static methods)
        from hud import HUD
        print("+ HUD class imported")
        
        return True
    except Exception as e:
        print(f"- Class instantiation error: {e}")
        return False

def test_config_values():
    """Test that config values are reasonable."""
    try:
        import config
        assert config.CAMERA_WIDTH > 0
        assert config.CAMERA_HEIGHT > 0
        assert config.CAMERA_FPS > 0
        assert 0 <= config.BRIGHTNESS <= 1
        assert 0 <= config.CONTRAST <= 3
        print("+ Config values are valid")
        return True
    except Exception as e:
        print(f"- Config validation error: {e}")
        return False

if __name__ == "__main__":
    print("Testing face recognition system modules...\n")
    
    tests = [
        test_imports,
        test_classes,
        test_config_values
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Empty line between tests
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("+ All tests passed! System is ready to use.")
    else:
        print("- Some tests failed. Please check the errors above.")