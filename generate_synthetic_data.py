import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

def generate_student_session(student_type, num_problems=10):
    """
    Generate synthetic data for different student types
    
    Student Types:
    - quick_learner: High accuracy, fast response, difficulty increases
    - struggling: Low accuracy, slow response, difficulty decreases
    - inconsistent: Alternating performance, difficulty fluctuates
    - improver: Starts poor, gradually improves
    - plateau: Consistent moderate performance
    """
    
    sessions = []
    difficulty_levels = ["Easy", "Medium", "Hard"]
    difficulty_map = {"Easy": 0, "Medium": 1, "Hard": 2}
    
    # Starting difficulty
    current_difficulty = "Medium"
    
    for problem_num in range(1, num_problems + 1):
        
        # Generate performance based on student type
        if student_type == "quick_learner":
            is_correct = np.random.choice([True, False], p=[0.90, 0.10])
            time_taken = np.random.uniform(3, 7)
            
        elif student_type == "struggling":
            is_correct = np.random.choice([True, False], p=[0.35, 0.65])
            time_taken = np.random.uniform(10, 20)
            
        elif student_type == "inconsistent":
            is_correct = True if problem_num % 2 == 0 else False
            time_taken = np.random.uniform(5, 12)
            
        elif student_type == "improver":
            accuracy_prob = min(0.3 + (problem_num * 0.06), 0.95)
            is_correct = np.random.choice([True, False], p=[accuracy_prob, 1-accuracy_prob])
            time_taken = np.random.uniform(8, 15) - (problem_num * 0.3)
            
        else:  # plateau
            is_correct = np.random.choice([True, False], p=[0.65, 0.35])
            time_taken = np.random.uniform(6, 10)
        
        # Determine next difficulty based on simple rules
        if problem_num >= 3:  # Need at least 3 problems to decide
            recent_correct = sum([s['is_correct'] for s in sessions[-2:]])
            recent_time = np.mean([s['time_taken'] for s in sessions[-2:]])
            
            current_level_idx = difficulty_map[current_difficulty]
            
            # Rule: If 2+ correct and fast, increase
            if recent_correct >= 2 and recent_time < 8 and current_level_idx < 2:
                next_difficulty = difficulty_levels[current_level_idx + 1]
            # Rule: If mostly wrong or very slow, decrease
            elif recent_correct <= 1 and current_level_idx > 0:
                next_difficulty = difficulty_levels[current_level_idx - 1]
            else:
                next_difficulty = current_difficulty
        else:
            next_difficulty = current_difficulty
        
        # Record session data
        session_data = {
            'problem_number': problem_num,
            'student_type': student_type,
            'current_difficulty': current_difficulty,
            'is_correct': is_correct,
            'time_taken': round(time_taken, 2),
            'next_difficulty': next_difficulty
        }
        
        sessions.append(session_data)
        current_difficulty = next_difficulty
    
    return sessions

# Generate data for all student types
all_sessions = []

student_types = ['quick_learner', 'struggling', 'inconsistent', 'improver', 'plateau']

# Generate 5 sessions per student type
for student_type in student_types:
    for session_num in range(40):
        session = generate_student_session(student_type, num_problems=10)
        all_sessions.extend(session)

# Convert to DataFrame
df = pd.DataFrame(all_sessions)

# Save to CSV
df.to_csv('synthetic_training_data.csv', index=False)

print(f"Generated {len(df)} training samples")
print("\nData Summary:")
print(df.groupby('student_type').agg({
    'is_correct': 'mean',
    'time_taken': 'mean',
    'current_difficulty': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
}).round(2))

print("\nFirst 10 rows:")
print(df.head(10))

print("\nData saved to 'synthetic_training_data.csv'")