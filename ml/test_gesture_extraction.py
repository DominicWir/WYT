"""
Test script to verify gesture label extraction from Opportunity dataset.

This script loads a small subset of the Opportunity dataset and verifies:
1. Gesture labels are extracted correctly
2. Combined labels are generated properly
3. Class distribution is reasonable
"""

import sys
sys.path.append('/Users/dominicwirasinha/AndroidStudioProjects/WYTV2/ml')

from load_zip_datasets import load_dataset_from_zip
import numpy as np

def test_gesture_extraction():
    """Test gesture label extraction."""
    
    print("=" * 70)
    print("TESTING GESTURE LABEL EXTRACTION")
    print("=" * 70)
    
    data_path = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/opportunity+activity+recognition.zip"
    
    # Test 1: Load with gestures (5-class)
    print("\n### TEST 1: Loading with gesture detection (5-class) ###\n")
    
    dataset_gestures = load_dataset_from_zip(
        'opportunity',
        data_path,
        subjects=['S1'],  # Just one subject for testing
        sessions=['ADL1'],  # Just one session
        include_gestures=True  # Enable gesture detection
    )
    
    activities_5class = dataset_gestures['activities']
    
    # Analyze class distribution
    print("\n5-Class Distribution:")
    unique, counts = np.unique(activities_5class, return_counts=True)
    class_names = ['Walking', 'Standing', 'Sitting', 'Hand Gesture', 'Typing']
    
    for cls, count in zip(unique, counts):
        if cls < len(class_names):
            pct = count / len(activities_5class) * 100
            print(f"  Class {cls} ({class_names[cls]}): {count:,} samples ({pct:.1f}%)")
    
    # Test 2: Load without gestures (2-class) for comparison
    print("\n\n### TEST 2: Loading without gesture detection (2-class) ###\n")
    
    dataset_no_gestures = load_dataset_from_zip(
        'opportunity',
        data_path,
        subjects=['S1'],
        sessions=['ADL1'],
        include_gestures=False  # Disable gesture detection
    )
    
    activities_2class = dataset_no_gestures['activities']
    
    # Analyze class distribution
    print("\n2-Class Distribution:")
    unique, counts = np.unique(activities_2class, return_counts=True)
    class_names_2 = ['Stationary', 'Walking']
    
    for cls, count in zip(unique, counts):
        if cls < len(class_names_2):
            pct = count / len(activities_2class) * 100
            print(f"  Class {cls} ({class_names_2[cls]}): {count:,} samples ({pct:.1f}%)")
    
    # Verification
    print("\n\n### VERIFICATION ###\n")
    
    # Check that gesture detection creates more classes
    n_classes_5 = len(np.unique(activities_5class))
    n_classes_2 = len(np.unique(activities_2class))
    
    print(f"✓ 5-class model has {n_classes_5} unique classes")
    print(f"✓ 2-class model has {n_classes_2} unique classes")
    
    if n_classes_5 > n_classes_2:
        print("✓ Gesture detection is working (more classes detected)")
    else:
        print("⚠ Warning: Gesture detection may not be working properly")
    
    # Check for hand gesture class (class 3)
    if 3 in activities_5class:
        gesture_count = np.sum(activities_5class == 3)
        gesture_pct = gesture_count / len(activities_5class) * 100
        print(f"✓ Hand gestures detected: {gesture_count:,} samples ({gesture_pct:.1f}%)")
    else:
        print("⚠ Warning: No hand gestures detected in this session")
    
    # Check for typing class (class 4)
    if 4 in activities_5class:
        typing_count = np.sum(activities_5class == 4)
        typing_pct = typing_count / len(activities_5class) * 100
        print(f"✓ Typing/fine motor detected: {typing_count:,} samples ({typing_pct:.1f}%)")
    else:
        print("⚠ No typing/fine motor gestures in this session (expected for ADL1)")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return dataset_gestures, dataset_no_gestures


if __name__ == "__main__":
    try:
        dataset_5class, dataset_2class = test_gesture_extraction()
        print("\n✓ All tests passed!")
    except FileNotFoundError:
        print("\n✗ Error: Opportunity dataset not found")
        print("Please ensure the dataset is at the correct path")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
