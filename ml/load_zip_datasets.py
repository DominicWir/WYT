"""
Data Loader for ZIP Datasets

This script loads and preprocesses datasets from ZIP files:
1. UJIIndoorLoc - Wi-Fi indoor localization dataset
2. Opportunity Activity Recognition - IMU activity recognition dataset
"""

import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings


class UJIIndoorLocLoader:
    """
    Loader for UJIIndoorLoc dataset (Wi-Fi fingerprinting for indoor localization).
    
    Dataset contains:
    - 520 Wi-Fi access point RSSI values
    - Position labels (longitude, latitude, floor, building, space, position)
    - Training: 19937 samples
    - Validation: 1111 samples
    """
    
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.df_train = None
        self.df_val = None
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and validation data."""
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            # Load training data
            with z.open('UJIndoorLoc/trainingData.csv') as f:
                self.df_train = pd.read_csv(f)
            
            # Load validation data
            with z.open('UJIndoorLoc/validationData.csv') as f:
                self.df_val = pd.read_csv(f)
        
        print(f"UJIIndoorLoc dataset loaded:")
        print(f"  Training samples: {len(self.df_train)}")
        print(f"  Validation samples: {len(self.df_val)}")
        
        return self.df_train, self.df_val
    
    def extract_for_xlstm(
        self,
        use_validation: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract data in format suitable for xLSTM training.
        
        Note: This dataset only has Wi-Fi data, no IMU or magnetic sensors.
        We'll need to create synthetic IMU data or use only Wi-Fi features.
        
        Args:
            use_validation: If True, use validation set instead of training
        
        Returns:
            Dictionary with sensor data and labels
        """
        if self.df_train is None:
            self.load()
        
        df = self.df_val if use_validation else self.df_train
        
        # Extract Wi-Fi RSSI values (first 520 columns)
        # Values are in range [-104, 0] or 100 (not detected)
        wifi_columns = [col for col in df.columns if col.startswith('WAP')]
        wifi_data = df[wifi_columns].values
        
        # Replace 100 (not detected) with -100 (very weak signal)
        wifi_data[wifi_data == 100] = -100
        
        # Extract position labels
        longitude = df['LONGITUDE'].values
        latitude = df['LATITUDE'].values
        floor = df['FLOOR'].values
        
        # Normalize longitude/latitude to meters (approximate)
        # UJI dataset uses specific coordinate system
        positions = np.column_stack([longitude, latitude, floor])
        
        # Extract building/space as activity proxy
        # (since this dataset doesn't have activity labels)
        building = df['BUILDINGID'].values
        space = df['SPACEID'].values
        
        # Create synthetic activity labels based on space type
        # 0-4 for different space types
        activities = (space % 5).astype(int)
        
        # Create synthetic IMU data (since dataset doesn't have it)
        # This is just for compatibility - you should replace with real IMU data
        n_samples = len(df)
        imu_data = np.random.randn(n_samples, 6) * 0.5
        imu_data[:, 2] += 9.81  # Add gravity to z-axis accelerometer
        
        # Create synthetic magnetic data
        mag_data = np.random.randn(n_samples, 3) * 5 + [20, 30, -40]
        
        print(f"\nExtracted data shapes:")
        print(f"  Wi-Fi: {wifi_data.shape}")
        print(f"  Positions: {positions.shape}")
        print(f"  Activities: {activities.shape}")
        print(f"  IMU (synthetic): {imu_data.shape}")
        print(f"  Magnetic (synthetic): {mag_data.shape}")
        
        warnings.warn(
            "UJIIndoorLoc dataset only contains Wi-Fi data. "
            "IMU and magnetic data are synthetically generated. "
            "For real training, collect actual IMU/magnetic data from your Android app."
        )
        
        return {
            'imu': imu_data,
            'magnetic': mag_data,
            'wifi': wifi_data,
            'positions': positions,
            'activities': activities,
            'metadata': {
                'building': df['BUILDINGID'].values,
                'floor': floor,
                'space': space,
                'user': df['USERID'].values if 'USERID' in df.columns else None
            }
        }


class OpportunityDatasetLoader:
    """
    Loader for Opportunity Activity Recognition dataset.
    
    Dataset contains:
    - 250 sensor channels (accelerometers, gyroscopes, magnetometers)
    - Multiple body locations (RKN, HIP, LUA, RUA, LH, BACK)
    - Activity labels (locomotion, high-level, low-level)
    - 4 subjects, 6 sessions each (ADL1-5, Drill)
    - Sampling rate: 30 Hz
    
    Column structure:
    - Column 1: Timestamp (milliseconds)
    - Columns 2-46: Inertial sensors (acc, gyro, mag) from body locations
    - Columns 47-243: Object sensors
    - Columns 244-250: Activity labels
    """
    
    # Sensor column indices (0-indexed, subtract 1 from documentation)
    # Back sensor (best for general activity recognition)
    BACK_ACC_COLS = [37, 38, 39]  # Columns 38-40 in docs
    BACK_GYRO_COLS = [40, 41, 42]  # Columns 41-43 in docs
    BACK_MAG_COLS = [43, 44, 45]   # Columns 44-46 in docs
    
    # Right knee (RKN) sensor
    RKN_ACC_COLS = [1, 2, 3]      # Columns 2-4 in docs
    
    # Hip sensor
    HIP_ACC_COLS = [4, 5, 6]      # Columns 5-7 in docs
    
    # Activity label columns (0-indexed, subtract 1 from documentation)
    LOCOMOTION_LABEL_COL = 243    # Column 244 in docs - Basic locomotion
    ML_BOTH_ARMS_COL = 244        # Column 245 in docs - Mid-level both arms gestures
    LL_LEFT_ARM_COL = 245         # Column 246 in docs - Low-level left arm actions
    LL_RIGHT_ARM_COL = 246        # Column 247 in docs - Low-level right arm actions  
    ML_LEFT_ARM_COL = 247         # Column 248 in docs - Mid-level left arm gestures
    ML_RIGHT_ARM_COL = 248        # Column 249 in docs - Mid-level right arm gestures
    
    # Locomotion activity mappings (basic movement)
    LOCOMOTION_MAP = {
        0: 'other',
        1: 'stand',
        2: 'walk',
        4: 'sit',
        5: 'lie',
    }
    
    # Mid-level gesture mappings (ML_Both_Arms column)
    # These are daily living activities involving hand/arm movements
    # Based on actual dataset investigation
    GESTURE_MAP = {
        0: 'none',      # No gesture
        101: 'open',    # Opening action (door, drawer, fridge, etc.)
        102: 'close',   # Closing action
        103: 'reach',   # Reaching movement
        104: 'move',    # Moving/carrying object
        105: 'use',     # Using object (clean, drink, toggle, etc.)
    }
    
    # Low-level arm action codes (for reference)
    # LL_Left_Arm: 204-213 (reach, grasp, release, etc.)
    # LL_Right_Arm: 302-323 (reach, grasp, release, etc.)
    
    # Combined activity classes for step detection
    # 0: Walking (allow steps)
    # 1: Standing (block steps)
    # 2: Sitting/Lying (block steps)
    # 3: Hand Gestures (block steps - NEW!)
    # 4: Fine Motor/Typing (block steps - NEW!)
    COMBINED_ACTIVITY_MAP = {
        'walk': 0,
        'stand': 1,
        'sit': 2,
        'lie': 2,
        'hand_gesture': 3,
        'typing': 4,
        'other': 1,  # Treat as standing
    }
    
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.data_files = []
    
    def load(self, subjects: list = None, sessions: list = None) -> pd.DataFrame:
        """
        Load Opportunity dataset.
        
        Args:
            subjects: List of subjects to load (e.g., ['S1', 'S2']). None = all
            sessions: List of sessions to load (e.g., ['ADL1', 'ADL2']). None = all
        
        Returns:
            Combined DataFrame with all loaded data
        """
        if subjects is None:
            subjects = ['S1', 'S2', 'S3', 'S4']
        
        if sessions is None:
            sessions = ['ADL1', 'ADL2', 'ADL3', 'ADL4', 'ADL5', 'Drill']
        
        all_data = []
        
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            dat_files = [f for f in z.namelist() if f.endswith('.dat')]
            
            for subject in subjects:
                for session in sessions:
                    filename = f'OpportunityUCIDataset/dataset/{subject}-{session}.dat'
                    
                    if filename in dat_files:
                        print(f"Loading {subject}-{session}...")
                        
                        with z.open(filename) as f:
                            # Read data (space-separated)
                            data = pd.read_csv(f, sep=' ', header=None)
                            data['subject'] = subject
                            data['session'] = session
                            all_data.append(data)
        
        # Combine all data
        combined = pd.concat(all_data, ignore_index=True)
        
        print(f"\nLoaded Opportunity dataset:")
        print(f"  Total samples: {len(combined)}")
        print(f"  Subjects: {subjects}")
        print(f"  Sessions per subject: {sessions}")
        print(f"  Columns: {combined.shape[1]}")
        
        return combined
    
    def extract_for_xlstm(
        self,
        subjects: list = None,
        sessions: list = None,
        use_back_sensor: bool = True,
        include_gestures: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract data in format suitable for xLSTM training.
        
        Args:
            subjects: List of subjects to load. None = all
            sessions: List of sessions to load. None = all
            use_back_sensor: If True, use BACK sensor (has acc+gyro+mag).
                           If False, use RKN+HIP sensors (acc only)
            include_gestures: If True, create 5-class labels (walk, stand, sit, gesture, typing)
                            If False, create 2-class labels (walk, stationary)
        
        Returns:
            Dictionary with sensor data and labels
        """
        # Load data
        df = self.load(subjects, sessions)
        
        # Replace NaN with 0
        df = df.fillna(0)
        
        # Extract IMU data
        if use_back_sensor:
            # Back sensor has all: accelerometer, gyroscope, magnetometer
            acc_data = df.iloc[:, self.BACK_ACC_COLS].values
            gyro_data = df.iloc[:, self.BACK_GYRO_COLS].values
            mag_data = df.iloc[:, self.BACK_MAG_COLS].values
            
            # Convert from milli-g to m/s² for accelerometer
            acc_data = acc_data / 1000.0 * 9.81
            
            # Gyroscope is already in proper units
            # Magnetometer is in local units
            
        else:
            # Use RKN + HIP accelerometers
            rkn_acc = df.iloc[:, self.RKN_ACC_COLS].values
            hip_acc = df.iloc[:, self.HIP_ACC_COLS].values
            
            # Average the two sensors
            acc_data = (rkn_acc + hip_acc) / 2.0
            acc_data = acc_data / 1000.0 * 9.81  # Convert to m/s²
            
            # Create synthetic gyro and mag (since RKN/HIP don't have them)
            gyro_data = np.zeros((len(df), 3))
            mag_data = np.random.randn(len(df), 3) * 5 + [20, 30, -40]
            
            warnings.warn(
                "RKN/HIP sensors don't have gyroscope/magnetometer. "
                "Using synthetic data. Recommend use_back_sensor=True."
            )
        
        # Combine IMU (acc + gyro)
        imu_data = np.column_stack([acc_data, gyro_data])
        
        # Extract activity labels
        if include_gestures:
            # Extract both locomotion and gesture labels
            locomotion_labels = df.iloc[:, self.LOCOMOTION_LABEL_COL].values
            gesture_labels = df.iloc[:, self.ML_BOTH_ARMS_COL].values
            
            # Create combined 5-class labels
            activities = self._create_combined_labels(locomotion_labels, gesture_labels)
            
            print(f"\nUsing 5-class labels (with gesture detection):")
        else:
            # Extract locomotion labels only (original 2-class)
            locomotion_labels = df.iloc[:, self.LOCOMOTION_LABEL_COL].values
            
            # Map to simplified activity classes (0=stationary, 1=walking)
            activities = np.array([
                1 if int(label) == 2 else 0  # 2=walk, everything else=stationary
                for label in locomotion_labels
            ])
            
            print(f"\nUsing 2-class labels (locomotion only):")
        
        # Create synthetic position data (dataset doesn't have ground truth positions)
        # Use cumulative displacement from accelerometer as proxy
        positions = self._estimate_positions_from_imu(acc_data)
        
        # No Wi-Fi data in this dataset
        wifi_data = None
        
        print(f"\nExtracted data shapes:")
        print(f"  IMU: {imu_data.shape}")
        print(f"  Magnetic: {mag_data.shape}")
        print(f"  Positions (estimated): {positions.shape}")
        print(f"  Activities: {activities.shape}")
        print(f"\nActivity distribution:")
        unique, counts = np.unique(activities, return_counts=True)
        for act, count in zip(unique, counts):
            act_name = ['stationary', 'walking', 'running', 'stairs_up', 'stairs_down'][int(act)]
            print(f"  {act_name}: {count} samples ({count/len(activities)*100:.1f}%)")
        
        warnings.warn(
            "Opportunity dataset doesn't have ground truth positions. "
            "Positions are estimated from IMU data. "
            "For real positioning, use UJIIndoorLoc or collect your own data."
        )
        
        return {
            'imu': imu_data,
            'magnetic': mag_data,
            'wifi': wifi_data,
            'positions': positions,
            'activities': activities,
            'metadata': {
                'subject': df['subject'].values,
                'session': df['session'].values,
                'timestamp': df.iloc[:, 0].values
            }
        }
    
    def _create_combined_labels(
        self, 
        locomotion_labels: np.ndarray, 
        gesture_labels: np.ndarray
    ) -> np.ndarray:
        """
        Create combined activity labels from locomotion and gesture data.
        
        Logic:
        - If walking → Class 0 (walking) - ALLOW STEPS
        - If standing + hand gesture → Class 3 (hand_gesture) - BLOCK STEPS
        - If standing + no gesture → Class 1 (standing) - BLOCK STEPS  
        - If sitting/lying → Class 2 (sitting) - BLOCK STEPS
        - Fine motor gestures (typing-like) → Class 4 (typing) - BLOCK STEPS
        
        Args:
            locomotion_labels: Raw locomotion labels from dataset
            gesture_labels: Raw gesture labels from ML_Both_Arms column
            
        Returns:
            Combined activity labels (0-4)
        """
        combined = []
        
        for loc_raw, gest_raw in zip(locomotion_labels, gesture_labels):
            # Map locomotion to string
            loc = self.LOCOMOTION_MAP.get(int(loc_raw), 'other')
            
            # Check if gesture is present
            gest = self.GESTURE_MAP.get(int(gest_raw), 'none')
            has_gesture = (gest != 'none')
            
            # Determine combined class
            if loc == 'walk':
                # Walking - allow steps
                combined.append(0)
            elif loc in ['stand', 'other'] and has_gesture:
                # Standing with hand gesture - block steps
                # Check if it's a fine motor gesture (typing-like)
                if 'toggle' in gest or 'switch' in gest:
                    combined.append(4)  # Fine motor/typing
                else:
                    combined.append(3)  # Hand gesture
            elif loc == 'stand' or loc == 'other':
                # Standing still - block steps
                combined.append(1)
            else:
                # Sitting/lying - block steps
                combined.append(2)
        
        return np.array(combined)
    
    def _estimate_positions_from_imu(self, acc_data: np.ndarray) -> np.ndarray:
        """
        Estimate positions from accelerometer data using double integration.
        
        This is a rough approximation and will drift over time.
        """
        # Remove gravity (assume z-axis is vertical)
        acc_no_gravity = acc_data.copy()
        acc_no_gravity[:, 2] -= 9.81
        
        # Simple double integration (very rough approximation)
        dt = 1/30.0  # 30 Hz sampling rate
        
        velocity = np.cumsum(acc_no_gravity * dt, axis=0)
        position = np.cumsum(velocity * dt, axis=0)
        
        # Add floor (assume ground level)
        floor = np.zeros((len(position), 1))
        
        positions = np.column_stack([position[:, 0], position[:, 1], floor])
        
        return positions


def load_dataset_from_zip(
    dataset_name: str,
    zip_path: str,
    use_validation: bool = False,
    subjects: list = None,
    sessions: list = None,
    include_gestures: bool = False
) -> Dict[str, np.ndarray]:
    """
    Load dataset from ZIP file.
    
    Args:
        dataset_name: 'ujiindoorloc' or 'opportunity'
        zip_path: Path to ZIP file
        use_validation: For UJIIndoorLoc, use validation set
        subjects: For Opportunity, list of subjects (e.g., ['S1', 'S2'])
        sessions: For Opportunity, list of sessions (e.g., ['ADL1', 'ADL2'])
        include_gestures: For Opportunity, create 5-class labels with gesture detection
    
    Returns:
        Dictionary with sensor data and labels
    """
    if dataset_name.lower() == 'ujiindoorloc':
        loader = UJIIndoorLocLoader(zip_path)
        return loader.extract_for_xlstm(use_validation=use_validation)
    
    elif dataset_name.lower() == 'opportunity':
        loader = OpportunityDatasetLoader(zip_path)
        return loader.extract_for_xlstm(
            subjects=subjects, 
            sessions=sessions,
            include_gestures=include_gestures
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Loader Demo")
    print("=" * 60 + "\n")
    
    # Test Opportunity loader (if available)
    opp_path = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/opportunity+activity+recognition.zip"
    
    if Path(opp_path).exists():
        print("Loading Opportunity dataset...\n")
        dataset = load_dataset_from_zip(
            'opportunity', 
            opp_path,
            subjects=['S1'],  # Load just one subject for demo
            sessions=['ADL1']  # Load just one session for demo
        )
        
        print("\n✓ Opportunity dataset loaded successfully!")
        print("\nYou can now use this data for training:")
        print("  - IMU data: Real accelerometer + gyroscope")
        print("  - Magnetic data: Real magnetometer")
        print("  - Activity labels: Real locomotion labels")
        print("  - Positions: Estimated from IMU (not ground truth)")
    else:
        print(f"Opportunity dataset not found at: {opp_path}")
        print("\nTrying UJIIndoorLoc instead...\n")
        
        # Test UJIIndoorLoc loader
        uji_path = "/Users/dominicwirasinha/Documents/IIT/Year 5/FYP Dev/ujiindoorloc.zip"
        
        if Path(uji_path).exists():
            print("Loading UJIIndoorLoc dataset...\n")
            dataset = load_dataset_from_zip('ujiindoorloc', uji_path)
            
            print("\n✓ Dataset loaded successfully!")
            print("\nYou can now use this data for training:")
            print("  - Wi-Fi data: Real RSSI values from 520 APs")
            print("  - Position labels: Real (longitude, latitude, floor)")
            print("  - IMU/Magnetic: Synthetic (replace with real data)")
        else:
            print(f"UJIIndoorLoc dataset not found at: {uji_path}")
