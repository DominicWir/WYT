# Quick Fix for Class Weights Bug
# Run this in Colab to fix the training script

# The issue: np.bincount only creates bins for existing classes
# Fix: Ensure we have weights for all 5 classes

# Replace lines 194-198 in train_gesture_detection_colab.py with:

"""
# Class weights for imbalanced data (fixed for all 5 classes)
class_counts = np.bincount(activities_aligned, minlength=5)  # Force 5 bins
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.FloatTensor(class_weights).to(device)
"""

print("Copy the code above and replace lines 194-198 in the Colab notebook")
print("\nOR run this cell in Colab to patch the file:")
print("""
# Patch the training script
with open('train_gesture_detection_colab.py', 'r') as f:
    lines = f.readlines()

# Fix line 195 (0-indexed: line 194)
lines[194] = '    class_counts = np.bincount(activities_aligned, minlength=5)  # Force 5 bins\\n'

with open('train_gesture_detection_colab.py', 'w') as f:
    f.writelines(lines)

print("✓ Fixed! Re-run the training script")
""")
