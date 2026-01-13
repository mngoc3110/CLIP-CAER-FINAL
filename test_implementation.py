#!/usr/bin/env python3
"""
Quick test to verify Phase 1 implementation is correct.
Run this before starting experiments to catch any errors early.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from utils.builders import get_class_weights_from_annotation
        from utils.loss import FocalLoss, MILoss, DCLoss
        from trainer import Trainer
        import torch
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_class_weights():
    """Test class weight computation."""
    print("\nTesting class weight computation...")
    try:
        from utils.builders import get_class_weights_from_annotation
        import torch

        # Use train_80.txt for testing
        annotation_file = "RAER/annotation/train_80.txt"
        if not os.path.exists(annotation_file):
            print(f"  ⚠ Annotation file not found: {annotation_file}")
            print(f"  Skipping test (this is OK if you haven't set up data yet)")
            return True

        weights = get_class_weights_from_annotation(annotation_file, num_classes=5)

        # Check weights are valid
        assert weights.shape[0] == 5, "Should have 5 class weights"
        assert torch.all(weights > 0), "All weights should be positive"
        assert weights[2] > weights[0], "Class 2 (minority) should have higher weight than Class 0 (majority)"

        print(f"  ✓ Class weights computed successfully")
        print(f"    Class 0 weight: {weights[0]:.3f}")
        print(f"    Class 2 weight: {weights[2]:.3f} (minority class)")
        return True
    except AssertionError as e:
        print(f"  ✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_focal_loss():
    """Test Focal Loss with class weights."""
    print("\nTesting Focal Loss...")
    try:
        from utils.loss import FocalLoss
        import torch

        # Test with scalar alpha
        criterion1 = FocalLoss(alpha=0.25, gamma=2.0)
        print("  ✓ Focal Loss with scalar alpha works")

        # Test with tensor alpha (class weights)
        class_weights = torch.tensor([0.92, 2.15, 5.87, 1.68, 1.05])
        criterion2 = FocalLoss(alpha=class_weights, gamma=2.0)
        print("  ✓ Focal Loss with tensor alpha works")

        # Test forward pass
        logits = torch.randn(4, 5)  # batch_size=4, num_classes=5
        targets = torch.tensor([0, 1, 2, 3])
        loss = criterion2(logits, targets)

        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"
        print(f"  ✓ Forward pass successful, loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_experiment_scripts():
    """Test that experiment scripts exist and are executable."""
    print("\nTesting experiment scripts...")
    scripts = [
        'experiments/exp1_proper_baseline.sh',
        'experiments/exp2_focal_loss.sh',
        'experiments/exp3_full_stack.sh',
        'experiments/compare_experiments.py'
    ]

    all_ok = True
    for script in scripts:
        if os.path.exists(script):
            is_executable = os.access(script, os.X_OK)
            if is_executable:
                print(f"  ✓ {script} exists and is executable")
            else:
                print(f"  ⚠ {script} exists but not executable (run: chmod +x {script})")
                all_ok = False
        else:
            print(f"  ✗ {script} not found")
            all_ok = False

    return all_ok

def test_annotation_files():
    """Test that annotation files exist."""
    print("\nTesting annotation files...")
    files = [
        ('RAER/annotation/train_80.txt', 'Training'),
        ('RAER/annotation/val_20.txt', 'Validation'),
        ('RAER/annotation/test.txt', 'Test')
    ]

    all_ok = True
    for filepath, name in files:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                num_lines = sum(1 for _ in f)
            print(f"  ✓ {name} file exists ({num_lines} samples)")
        else:
            print(f"  ⚠ {name} file not found: {filepath}")
            all_ok = False

    if not all_ok:
        print("\n  Note: If annotation files are missing, experiments won't run.")
        print("  Make sure RAER dataset is properly set up.")

    return all_ok

def main():
    print("=" * 60)
    print("PHASE 1 IMPLEMENTATION VERIFICATION")
    print("=" * 60)

    tests = [
        ("Module Imports", test_imports),
        ("Class Weights", test_class_weights),
        ("Focal Loss", test_focal_loss),
        ("Experiment Scripts", test_experiment_scripts),
        ("Annotation Files", test_annotation_files),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed! Ready to run experiments.")
        print("\nNext step: bash experiments/exp3_full_stack.sh")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix issues before running experiments.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
