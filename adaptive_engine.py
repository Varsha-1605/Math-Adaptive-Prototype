class AdaptiveEngine:
    """
    Rule-based adaptive learning engine that adjusts difficulty
    based on student performance
    """
    
    def __init__(self):
        self.difficulty_levels = ["Easy", "Medium", "Hard"]
        
        # Time thresholds for each difficulty (in seconds)
        self.time_thresholds = {
            "Easy": 8.0,
            "Medium": 12.0,
            "Hard": 20.0
        }
        
        self.last_adjustment_at = 0  # Track when we last adjusted (anti-bounce)
        self.decision_history = []   # Track decision reasoning
        
    def decide_next_difficulty(self, tracker, current_difficulty, problem_count):
        """
        Main decision function - determines next difficulty level
        
        Args:
            tracker (PerformanceTracker): The performance tracker object
            current_difficulty (str): Current difficulty level
            problem_count (int): Number of problems completed so far
            
        Returns:
            tuple: (next_difficulty, reasoning)
        """
        
        # Need at least 3 problems to make a decision
        if problem_count < 3:
            return current_difficulty, "Insufficient data (need 3 problems)"
        
        # Anti-bounce: Don't adjust too frequently
        if problem_count - self.last_adjustment_at < 3:
            return current_difficulty, "Too soon to adjust (waiting for stability)"
        
        # Get recent performance
        recent_attempts = tracker.get_recent_performance(n=3)
        
        # Calculate metrics
        accuracy = sum(1 for a in recent_attempts if a['is_correct']) / len(recent_attempts)
        avg_time = sum(a['time_taken'] for a in recent_attempts) / len(recent_attempts)
        
        # Calculate performance score
        score = self._calculate_performance_score(accuracy, avg_time, current_difficulty)
        
        # Get current difficulty index
        current_idx = self.difficulty_levels.index(current_difficulty)
        
        # Decision logic
        next_difficulty = current_difficulty
        reasoning = ""
        
        # Rule 1: INCREASE difficulty
        if self._should_increase(accuracy, avg_time, current_difficulty, current_idx):
            if current_idx < len(self.difficulty_levels) - 1:
                next_difficulty = self.difficulty_levels[current_idx + 1]
                reasoning = f"Strong performance (accuracy: {accuracy:.1%}, avg time: {avg_time:.1f}s)"
                self.last_adjustment_at = problem_count
            else:
                reasoning = "Already at maximum difficulty"
        
        # Rule 2: DECREASE difficulty
        elif self._should_decrease(accuracy, avg_time, current_difficulty, current_idx):
            if current_idx > 0:
                next_difficulty = self.difficulty_levels[current_idx - 1]
                reasoning = f"Struggling (accuracy: {accuracy:.1%}, avg time: {avg_time:.1f}s)"
                self.last_adjustment_at = problem_count
            else:
                reasoning = "Already at minimum difficulty"
        
        # Rule 3: MAINTAIN difficulty
        else:
            reasoning = f"Appropriate level (accuracy: {accuracy:.1%}, avg time: {avg_time:.1f}s)"
        
        # Log decision
        self.decision_history.append({
            'problem_count': problem_count,
            'current_difficulty': current_difficulty,
            'next_difficulty': next_difficulty,
            'accuracy': round(accuracy, 2),
            'avg_time': round(avg_time, 2),
            'score': round(score, 2),
            'reasoning': reasoning
        })
        
        return next_difficulty, reasoning
    
    def _calculate_performance_score(self, accuracy, avg_time, difficulty):
        """
        Calculate a performance score combining accuracy and speed
        
        Score = (Accuracy × 0.7) + (Speed Factor × 0.3)
        
        Args:
            accuracy (float): Accuracy ratio (0-1)
            avg_time (float): Average time taken
            difficulty (str): Current difficulty level
            
        Returns:
            float: Performance score (0-1)
        """
        threshold = self.time_thresholds[difficulty]
        
        # Speed factor: 1.0 if fast, 0.5 if at threshold, 0.0 if too slow
        if avg_time <= threshold * 0.7:
            speed_factor = 1.0
        elif avg_time <= threshold:
            speed_factor = 0.7
        elif avg_time <= threshold * 1.5:
            speed_factor = 0.4
        else:
            speed_factor = 0.0
        
        score = (accuracy * 0.7) + (speed_factor * 0.3)
        return score
    
    def _should_increase(self, accuracy, avg_time, difficulty, current_idx):
        """
        Check if difficulty should be increased
        
        Conditions:
        - Accuracy >= 66% (2 out of 3 correct)
        - Average time under threshold
        - Not already at Hard level
        """
        threshold = self.time_thresholds[difficulty]
        
        return (accuracy >= 0.66 and 
                avg_time < threshold and 
                current_idx < len(self.difficulty_levels) - 1)
    
    def _should_decrease(self, accuracy, avg_time, difficulty, current_idx):
        """
        Check if difficulty should be decreased
        
        Conditions:
        - Accuracy <= 33% (1 out of 3 or worse)
        - OR average time > 1.5x threshold
        - Not already at Easy level
        """
        threshold = self.time_thresholds[difficulty]
        
        return ((accuracy <= 0.33 or avg_time > threshold * 1.5) and 
                current_idx > 0)
    
    def get_decision_history(self):
        """
        Get the history of all adaptation decisions
        
        Returns:
            list: Decision history
        """
        return self.decision_history
    
    def explain_last_decision(self):
        """
        Get explanation for the most recent decision
        
        Returns:
            dict: Last decision details
        """
        if self.decision_history:
            return self.decision_history[-1]
        return None


# Test the adaptive engine
if __name__ == "__main__":
    from tracker import PerformanceTracker
    
    engine = AdaptiveEngine()
    tracker = PerformanceTracker("Test Student", "Medium")
    
    # Simulate strong performance
    print("Testing Adaptive Engine\n")
    print("Scenario: Strong Performance")
    
    tracker.log_attempt(1, "Medium", "23 + 15", 38, 38, 5.0, "addition")
    tracker.log_attempt(2, "Medium", "45 - 18", 27, 27, 4.5, "subtraction")
    tracker.log_attempt(3, "Medium", "7 × 6", 42, 42, 5.5, "multiplication")
    
    next_diff, reasoning = engine.decide_next_difficulty(tracker, "Medium", 3)
    print(f"Decision: {next_diff}")
    print(f"Reasoning: {reasoning}")
    
    print("\n" + "="*50 + "\n")
    
    # Simulate poor performance
    print("Scenario: Poor Performance")
    tracker2 = PerformanceTracker("Test Student 2", "Medium")
    engine2 = AdaptiveEngine()
    
    tracker2.log_attempt(1, "Medium", "23 + 15", 38, 40, 15.0, "addition")
    tracker2.log_attempt(2, "Medium", "45 - 18", 27, 30, 18.0, "subtraction")
    tracker2.log_attempt(3, "Medium", "7 × 6", 42, 40, 20.0, "multiplication")
    
    next_diff2, reasoning2 = engine2.decide_next_difficulty(tracker2, "Medium", 3)
    print(f"Decision: {next_diff2}")
    print(f"Reasoning: {reasoning2}")