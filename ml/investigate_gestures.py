"""
Investigate actual gesture label values in Opportunity dataset.
"""

import sys
sys.path.append('/Users/dominicwirasinha/AndroidStudioProjects/WYTV2/ml')

import zipfile
import pandas as pd
import numpy as np

def investigate_gesture_labels():
    """Investigate what gesture label values actually exist in the dataset."""
    
    print("=" * 70)
    print("INVESTIGATING GESTURE LABEL VALUES")
    print("=" * 70)
    
    data_path = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/opportunity+activity+recognition.zip"
    
    # Load raw data
    with zipfile.ZipFile(data_path, 'r') as z:
        filename = 'OpportunityUCIDataset/dataset/S1-ADL1.dat'
        
        with z.open(filename) as f:
            print(f"\nLoading {filename}...")
            data = pd.read_csv(f, sep=' ', header=None)
            
            print(f"Total columns: {data.shape[1]}")
            print(f"Total rows: {data.shape[0]}")
            
            # Check gesture columns
            print("\n### GESTURE LABEL ANALYSIS ###\n")
            
            # Column indices (0-indexed)
            LOCOMOTION_COL = 243
            ML_BOTH_ARMS_COL = 244
            LL_LEFT_ARM_COL = 245
            LL_RIGHT_ARM_COL = 246
            ML_LEFT_ARM_COL = 247
            ML_RIGHT_ARM_COL = 248
            
            # Locomotion labels
            print("1. LOCOMOTION LABELS (Column 244):")
            loc_labels = data.iloc[:, LOCOMOTION_COL].values
            unique_loc = np.unique(loc_labels[~np.isnan(loc_labels)])
            print(f"   Unique values: {unique_loc}")
            for val in unique_loc:
                count = np.sum(loc_labels == val)
                pct = count / len(loc_labels) * 100
                print(f"   {int(val)}: {count:,} samples ({pct:.1f}%)")
            
            # ML_Both_Arms labels
            print("\n2. ML_BOTH_ARMS LABELS (Column 245):")
            ml_both = data.iloc[:, ML_BOTH_ARMS_COL].values
            unique_ml_both = np.unique(ml_both[~np.isnan(ml_both)])
            print(f"   Unique values: {unique_ml_both}")
            print(f"   Total unique gestures: {len(unique_ml_both)}")
            for val in unique_ml_both[:20]:  # Show first 20
                count = np.sum(ml_both == val)
                pct = count / len(ml_both) * 100
                print(f"   {int(val)}: {count:,} samples ({pct:.1f}%)")
            
            # LL_Left_Arm labels
            print("\n3. LL_LEFT_ARM LABELS (Column 246):")
            ll_left = data.iloc[:, LL_LEFT_ARM_COL].values
            unique_ll_left = np.unique(ll_left[~np.isnan(ll_left)])
            print(f"   Unique values: {unique_ll_left}")
            for val in unique_ll_left[:10]:  # Show first 10
                count = np.sum(ll_left == val)
                pct = count / len(ll_left) * 100
                print(f"   {int(val)}: {count:,} samples ({pct:.1f}%)")
            
            # LL_Right_Arm labels
            print("\n4. LL_RIGHT_ARM LABELS (Column 247):")
            ll_right = data.iloc[:, LL_RIGHT_ARM_COL].values
            unique_ll_right = np.unique(ll_right[~np.isnan(ll_right)])
            print(f"   Unique values: {unique_ll_right}")
            for val in unique_ll_right[:10]:  # Show first 10
                count = np.sum(ll_right == val)
                pct = count / len(ll_right) * 100
                print(f"   {int(val)}: {count:,} samples ({pct:.1f}%)")
            
            print("\n" + "=" * 70)
            print("INVESTIGATION COMPLETE")
            print("=" * 70)
            
            return data


if __name__ == "__main__":
    data = investigate_gesture_labels()
