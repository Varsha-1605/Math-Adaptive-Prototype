import json
from datetime import datetime

class PerformanceTracker:
    """
    Tracks student performance throughout the session
    """
    
    def __init__(self, user_name, starting_difficulty):
        self.user_name = user_name
        self.starting_difficulty = starting_difficulty
        self.session_data = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_attempt(self, problem_num, difficulty, question, correct_answer, 
                   user_answer, time_taken, operation):
        """
        Log a single problem attempt
        
        Args:
            problem_num (int): Problem number in session
            difficulty (str): Current difficulty level
            question (str): The problem text
            correct_answer (int): Correct answer
            user_answer (int): User's answer
            time_taken (float): Time in seconds
            operation (str): Type of operation
        """
        is_correct = (user_answer == correct_answer)
        
        attempt = {
            'problem_number': problem_num,
            'difficulty': difficulty,
            'question': question,
            'correct_answer': correct_answer,
            'user_answer': user_answer,
            'is_correct': is_correct,
            'time_taken': round(time_taken, 2),
            'operation': operation,
            'timestamp': datetime.now().isoformat()
        }
        
        self.session_data.append(attempt)
        
    def get_recent_performance(self, n=3):
        """
        Get performance data for the last n problems
        
        Args:
            n (int): Number of recent problems to analyze
            
        Returns:
            list: Recent attempts
        """
        return self.session_data[-n:] if len(self.session_data) >= n else self.session_data
    
    def calculate_session_stats(self):
        """
        Calculate comprehensive session statistics
        
        Returns:
            dict: Session summary statistics
        """
        if not self.session_data:
            return {}
        
        total_problems = len(self.session_data)
        correct_count = sum(1 for a in self.session_data if a['is_correct'])
        accuracy = (correct_count / total_problems) * 100
        avg_time = sum(a['time_taken'] for a in self.session_data) / total_problems
        
        # Difficulty progression
        difficulty_progression = [a['difficulty'] for a in self.session_data]
        
        # Problems per difficulty
        problems_per_difficulty = {}
        for difficulty in ['Easy', 'Medium', 'Hard']:
            count = sum(1 for a in self.session_data if a['difficulty'] == difficulty)
            problems_per_difficulty[difficulty] = count
        
        # Final difficulty
        final_difficulty = self.session_data[-1]['difficulty']
        
        stats = {
            'user_name': self.user_name,
            'session_id': self.session_id,
            'total_problems': total_problems,
            'correct': correct_count,
            'incorrect': total_problems - correct_count,
            'accuracy_percentage': round(accuracy, 2),
            'average_time': round(avg_time, 2),
            'starting_difficulty': self.starting_difficulty,
            'final_difficulty': final_difficulty,
            'difficulty_progression': difficulty_progression,
            'problems_per_difficulty': problems_per_difficulty
        }
        
        return stats
    
    def get_performance_trend(self):
        """
        Analyze if student is improving, declining, or stable
        
        Returns:
            str: "improving", "declining", or "stable"
        """
        if len(self.session_data) < 4:
            return "insufficient_data"
        
        # Split session into first half and second half
        mid_point = len(self.session_data) // 2
        first_half = self.session_data[:mid_point]
        second_half = self.session_data[mid_point:]
        
        first_accuracy = sum(1 for a in first_half if a['is_correct']) / len(first_half)
        second_accuracy = sum(1 for a in second_half if a['is_correct']) / len(second_half)
        
        if second_accuracy > first_accuracy + 0.15:
            return "improving"
        elif second_accuracy < first_accuracy - 0.15:
            return "declining"
        else:
            return "stable"
    
    def export_session(self, filename=None):
        """
        Export session data to JSON file
        
        Args:
            filename (str): Output filename (optional)
        """
        if filename is None:
            filename = f"session_{self.session_id}.json"
        
        export_data = {
            'session_info': {
                'user_name': self.user_name,
                'session_id': self.session_id,
                'starting_difficulty': self.starting_difficulty
            },
            'problems': self.session_data,
            'summary': self.calculate_session_stats()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename


# Test the tracker
if __name__ == "__main__":
    tracker = PerformanceTracker("Test Student", "Medium")
    
    # Simulate some attempts
    tracker.log_attempt(1, "Medium", "23 + 15", 38, 38, 5.2, "addition")
    tracker.log_attempt(2, "Medium", "45 - 18", 27, 27, 6.1, "subtraction")
    tracker.log_attempt(3, "Medium", "7 × 6", 42, 40, 8.3, "multiplication")
    
    print("Recent Performance:")
    print(tracker.get_recent_performance(3))
    
    print("\nSession Stats:")
    print(tracker.calculate_session_stats())