#!/usr/bin/env python3
"""
Quick test script to verify all models can be loaded
"""

def test_yolo():
    print("\n🚀 Testing YOLOv8...")
    try:
        from models.yolo import YOLOModel
        model = YOLOModel('n')
        print("✅ YOLOv8 loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ YOLOv8 failed: {e}")
        return False

def test_mask_rcnn():
    print("\n🎯 Testing Mask R-CNN...")
    try:
        from models.mask_rcnn import MaskRCNNModel
        model = MaskRCNNModel()
        print("✅ Mask R-CNN loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Mask R-CNN failed: {e}")
        return False

def test_faster_rcnn():
    print("\n⚖️  Testing Faster R-CNN...")
    try:
        from models.faster_rcnn import FasterRCNNModel
        model = FasterRCNNModel()
        print("✅ Faster R-CNN loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Faster R-CNN failed: {e}")
        return False

def main():
    print("=" * 60)
    print("  Model Loading Test")
    print("=" * 60)
    
    results = []
    results.append(("YOLOv8", test_yolo()))
    results.append(("Mask R-CNN", test_mask_rcnn()))
    results.append(("Faster R-CNN", test_faster_rcnn()))
    
    print("\n" + "=" * 60)
    print("  Test Results Summary")
    print("=" * 60)
    
    for model_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name:20s}: {status}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\n{successful}/3 models loaded successfully")
    
    if successful == 3:
        print("\n🎉 All models ready! You can now run main.py")
    elif successful > 0:
        print(f"\n⚠️  {successful} model(s) working. You can proceed with those.")
    else:
        print("\n❌ No models loaded. Please check your installation.")

if __name__ == "__main__":
    main()