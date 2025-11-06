import numpy as np
import pandas as pd

class FeatureEngineer:
    """
    Creates engineered features from raw performance data for ML models
    Based on EDA notebook analysis - 9 enhanced features
    """
    
    def __init__(self):
        self.difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
        
    def create_features_from_tracker(self, tracker, current_difficulty):
        """
        Create features from PerformanceTracker for real-time prediction
        Implements all 9 features from EDA notebook Cell 2
        
        Args:
            tracker: PerformanceTracker object
            current_difficulty: Current difficulty level
            
        Returns:
            dict: Feature dictionary with 9 features
        """
        if len(tracker.session_data) < 3:
            return None  # Need at least 3 problems
        
        recent_3 = tracker.session_data[-3:]
        all_data = tracker.session_data
        
        # Calculate base metrics
        accuracy_last_3 = sum(1 for a in recent_3 if a['is_correct']) / len(recent_3)
        avg_time_last_3 = np.mean([a['time_taken'] for a in recent_3])
        std_time_last_3 = np.std([a['time_taken'] for a in recent_3])
        current_diff_encoded = self.difficulty_map[current_difficulty]
        
        # Calculate overall metrics for trends
        overall_accuracy = sum(1 for a in all_data if a['is_correct']) / len(all_data)
        overall_time = np.mean([a['time_taken'] for a in all_data])
        
        # Feature 1-4: Base features
        features = {
            'accuracy_last_3': accuracy_last_3,
            'avg_time_last_3': avg_time_last_3,
            'std_time_last_3': std_time_last_3,
            'current_difficulty': current_diff_encoded,
        }
        
        # Feature 5: Accuracy trend (improvement over time)
        features['accuracy_trend'] = accuracy_last_3 - overall_accuracy
        
        # Feature 6: Speed-accuracy ratio (efficiency metric)
        features['speed_accuracy_ratio'] = avg_time_last_3 / (accuracy_last_3 + 0.01)
        
        # Feature 7: Difficulty squared (non-linear effects)
        features['current_difficulty_squared'] = current_diff_encoded ** 2
        
        # Feature 8: Time improvement (getting faster?)
        features['time_improvement'] = overall_time - avg_time_last_3
        
        # Feature 9: Consistency score (inverse of time std)
        features['consistency_score'] = 1 / (std_time_last_3 + 0.1)
        
        return features
    
    def create_features_from_dataframe(self, df, window_size=3):
        """
        Create features from pandas DataFrame for batch training
        Implements all 9 enhanced features from EDA notebook
        
        Args:
            df: DataFrame with columns [problem_number, student_type, current_difficulty, 
                                       is_correct, time_taken, next_difficulty]
            window_size: Size of rolling window (default 3)
            
        Returns:
            DataFrame: Feature matrix with 9 features + target
        """
        features_list = []
        
        # Group by student sessions (every 10 problems is one session)
        for student_type in df['student_type'].unique():
            student_data = df[df['student_type'] == student_type].copy()
            
            # Process each session (10 problems each)
            for session_start in range(0, len(student_data), 10):
                session_data = student_data.iloc[session_start:session_start+10]
                
                for idx in range(window_size, len(session_data)):
                    window = session_data.iloc[idx-window_size:idx]
                    current_row = session_data.iloc[idx]
                    
                    # All data up to current point (for overall metrics)
                    all_prev = session_data.iloc[:idx]
                    
                    # Base calculations
                    accuracy_last_3 = window['is_correct'].mean()
                    avg_time_last_3 = window['time_taken'].mean()
                    std_time_last_3 = window['time_taken'].std()
                    current_diff = self.difficulty_map[current_row['current_difficulty']]
                    
                    # Overall metrics
                    overall_accuracy = all_prev['is_correct'].mean()
                    overall_time = all_prev['time_taken'].mean()
                    
                    # Create 9 features matching EDA notebook
                    features = {
                        # Base features (1-4)
                        'accuracy_last_3': accuracy_last_3,
                        'avg_time_last_3': avg_time_last_3,
                        'std_time_last_3': std_time_last_3,
                        'current_difficulty': current_diff,
                        
                        # Enhanced features (5-9)
                        'accuracy_trend': accuracy_last_3 - overall_accuracy,
                        'speed_accuracy_ratio': avg_time_last_3 / (accuracy_last_3 + 0.01),
                        'current_difficulty_squared': current_diff ** 2,
                        'time_improvement': overall_time - avg_time_last_3,
                        'consistency_score': 1 / (std_time_last_3 + 0.1),
                        
                        # Target
                        'next_difficulty': self.difficulty_map[current_row['next_difficulty']]
                    }
                    
                    features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def get_feature_names(self):
        """Return list of feature names (9 features)"""
        return [
            'accuracy_last_3',
            'avg_time_last_3', 
            'std_time_last_3',
            'current_difficulty',
            'accuracy_trend',
            'speed_accuracy_ratio',
            'current_difficulty_squared',
            'time_improvement',
            'consistency_score'
        ]


# Test feature engineering
if __name__ == "__main__":
    import pandas as pd
    
    print("="*60)
    print("TESTING ENHANCED FEATURE ENGINEERING")
    print("="*60)
    
    # Load synthetic data
    try:
        df = pd.read_csv('synthetic_training_data.csv')
        print(f"\n✓ Loaded data: {len(df)} samples")
    except FileNotFoundError:
        print("\n✗ synthetic_training_data.csv not found")
        print("  Run 'python generate_synthetic_data.py' first")
        exit(1)
    
    fe = FeatureEngineer()
    feature_df = fe.create_features_from_dataframe(df)
    
    print(f"\n✓ Created {len(feature_df)} training samples")
    print(f"\nFeature columns ({len(feature_df.columns)-1} features + 1 target):")
    for i, col in enumerate(feature_df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nFirst 5 samples:")
    print(feature_df.head())
    
    print(f"\nFeature statistics:")
    print(feature_df.describe())
    
    print(f"\nTarget distribution:")
    print(feature_df['next_difficulty'].value_counts().sort_index())
    print("\n✓ Feature engineering test complete!")