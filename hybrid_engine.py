from adaptive_engine import AdaptiveEngine
from ml_adaptive_engine import MLAdaptiveEngine

class HybridAdaptiveEngine:
    """
    Hybrid engine combining Rule-Based and ML approaches
    
    Strategy from EDA notebook:
    - First 10 problems: Use rule-based (cold start)
    - After 10 problems: Switch to ML-based predictions
    - Use confidence thresholds for fallback
    """
    
    def __init__(self, ml_model_path='models/random_forest_model.pkl'):
        """
        Initialize hybrid engine with both components
        
        Args:
            ml_model_path: Path to trained Random Forest model
        """
        self.rule_engine = AdaptiveEngine()
        self.ml_engine = MLAdaptiveEngine(ml_model_path)
        
        # Configuration based on EDA findings
        self.ml_start_threshold = 10  # Switch to ML after 5 problems
        self.ml_confidence_threshold = 0.7  # High confidence = trust ML fully
        self.low_confidence_threshold = 0.5  # Below this = use rules
        
        self.weights = {
            'rule_based': 0.3,
            'ml_based': 0.7
        }
        
        self.difficulty_levels = ["Easy", "Medium", "Hard"]
        self.decision_history = []
        
    def decide_next_difficulty(self, tracker, current_difficulty, problem_count):
        """
        Make decision using hybrid approach with 10-problem threshold
        
        Decision Flow:
        1. Problems 1-9: Use rule-based only (insufficient data for ML)
        2. Problem 10+: Primary ML, fallback to rules if low confidence
        3. If ML and rules disagree with low ML confidence: Use weighted vote
        
        Args:
            tracker: PerformanceTracker object
            current_difficulty: Current difficulty level
            problem_count: Number of problems completed
            
        Returns:
            tuple: (next_difficulty, reasoning, confidence, details)
        """
        
        # Get predictions from both engines
        rule_pred, rule_reasoning = self.rule_engine.decide_next_difficulty(
            tracker, current_difficulty, problem_count
        )
        
        # ML prediction (will return low confidence if < 3 problems)
        ml_pred, ml_reasoning, ml_confidence = self.ml_engine.decide_next_difficulty(
            tracker, current_difficulty, problem_count
        )
        
        # Decision variables
        decision_method = ""
        final_difficulty = current_difficulty
        final_confidence = 0.5
        reasoning = ""
        
        # PHASE 1: Problems 1-4 → Rule-based only (cold start)
        if problem_count < self.ml_start_threshold:
            final_difficulty = rule_pred
            decision_method = "rule_based (cold_start)"
            final_confidence = 0.6
            reasoning = f"Cold start phase (problem {problem_count}/5): {rule_reasoning}"
        
        # PHASE 2: Problem 5+ → ML primary, rules fallback
        else:
            # Case A: ML model not available → use rules
            if self.ml_engine.model is None:
                final_difficulty = rule_pred
                decision_method = "rule_based (no_ml_model)"
                final_confidence = 0.6
                reasoning = f"ML unavailable: {rule_reasoning}"
            
            # Case B: ML very confident (>70%) → trust ML fully
            elif ml_confidence > self.ml_confidence_threshold:
                final_difficulty = ml_pred
                decision_method = "ml_based (high_confidence)"
                final_confidence = ml_confidence
                reasoning = f"ML high confidence: {ml_reasoning}"
            
            # Case C: ML and rules agree → consensus decision
            elif rule_pred == ml_pred:
                final_difficulty = rule_pred
                decision_method = "consensus (agreement)"
                final_confidence = min(0.85, (ml_confidence + 0.6) / 2)
                reasoning = f"Both agree on {final_difficulty}: {ml_reasoning}"
            
            # Case D: ML moderate confidence (50-70%) + disagreement → weighted vote
            elif ml_confidence >= self.low_confidence_threshold:
                final_difficulty = ml_pred  # Trust ML more in this range
                decision_method = "ml_based (moderate_confidence)"
                final_confidence = ml_confidence
                reasoning = f"ML moderate confidence, preferring ML: {ml_reasoning}"
            
            # Case E: ML low confidence (<50%) + disagreement → use rules
            else:
                final_difficulty = rule_pred
                decision_method = "rule_based (low_ml_confidence)"
                final_confidence = 0.55
                reasoning = f"ML uncertain ({ml_confidence*100:.0f}%), using rules: {rule_reasoning}"
        
        # Compile detailed decision record
        decision_details = {
            'problem_count': problem_count,
            'current_difficulty': current_difficulty,
            'final_difficulty': final_difficulty,
            'decision_method': decision_method,
            'confidence': float(final_confidence),
            'reasoning': reasoning,
            'phase': 'cold_start' if problem_count < self.ml_start_threshold else 'ml_active',
            'components': {
                'rule_based': {
                    'prediction': rule_pred,
                    'reasoning': rule_reasoning,
                    'active': problem_count < self.ml_start_threshold or ml_confidence < self.low_confidence_threshold
                },
                'ml_based': {
                    'prediction': ml_pred,
                    'reasoning': ml_reasoning,
                    'confidence': float(ml_confidence),
                    'active': problem_count >= self.ml_start_threshold
                }
            }
        }
        
        self.decision_history.append(decision_details)
        
        return final_difficulty, reasoning, final_confidence, decision_details
    
    def explain_last_decision(self):
        """Get detailed explanation of last decision"""
        if self.decision_history:
            return self.decision_history[-1]
        return None
    
    def get_decision_history(self):
        """Get full decision history"""
        return self.decision_history
    
    def get_agreement_statistics(self):
        """
        Calculate agreement statistics between rule-based and ML
        
        Returns:
            dict: Comprehensive agreement stats
        """
        if not self.decision_history:
            return None
        
        stats = {
            'total_decisions': 0,
            'cold_start_decisions': 0,
            'ml_active_decisions': 0,
            'agreements': 0,
            'disagreements': 0,
            'ml_high_confidence': 0,
            'ml_low_confidence': 0,
            'method_breakdown': {}
        }
        
        for decision in self.decision_history:
            stats['total_decisions'] += 1
            
            # Phase tracking
            if decision['phase'] == 'cold_start':
                stats['cold_start_decisions'] += 1
            else:
                stats['ml_active_decisions'] += 1
            
            # Agreement tracking (only when both are active)
            if decision['phase'] == 'ml_active':
                rule_pred = decision['components']['rule_based']['prediction']
                ml_pred = decision['components']['ml_based']['prediction']
                
                if rule_pred == ml_pred:
                    stats['agreements'] += 1
                else:
                    stats['disagreements'] += 1
                
                # Confidence tracking
                ml_conf = decision['components']['ml_based']['confidence']
                if ml_conf > self.ml_confidence_threshold:
                    stats['ml_high_confidence'] += 1
                elif ml_conf < self.low_confidence_threshold:
                    stats['ml_low_confidence'] += 1
            
            # Method breakdown
            method = decision['decision_method']
            stats['method_breakdown'][method] = stats['method_breakdown'].get(method, 0) + 1
        
        # Calculate rates
        if stats['ml_active_decisions'] > 0:
            stats['agreement_rate'] = stats['agreements'] / stats['ml_active_decisions']
        else:
            stats['agreement_rate'] = 0.0
        
        return stats
    
    def get_phase_info(self, problem_count):
        """Get current phase information"""
        if problem_count < self.ml_start_threshold:
            return {
                'phase': 'cold_start',
                'description': f'Rule-based learning (problems 1-{self.ml_start_threshold-1})',
                'problems_until_ml': self.ml_start_threshold - problem_count
            }
        else:
            return {
                'phase': 'ml_active',
                'description': 'ML-powered adaptive learning',
                'ml_activated_at': self.ml_start_threshold
            }


# Test the hybrid engine
if __name__ == "__main__":
    from tracker import PerformanceTracker
    
    print("="*60)
    print("TESTING HYBRID ADAPTIVE ENGINE")
    print("10-Problem Threshold Implementation")
    print("="*60)
    
    # Create tracker
    tracker = PerformanceTracker("Test Student", "Medium")
    hybrid = HybridAdaptiveEngine()
    
    # Simulate 12 problems to test both phases
    test_cases = [
        (1, "Medium", "23 + 15", 38, 38, 8.0, "addition"),
        (2, "Medium", "45 - 18", 27, 27, 7.5, "subtraction"),
        (3, "Medium", "7 × 6", 42, 42, 7.0, "multiplication"),
        (4, "Medium", "12 × 5", 60, 60, 6.5, "multiplication"),
        (5, "Medium", "48 - 23", 25, 25, 6.0, "subtraction"),  # ML activates here
        (6, "Hard", "235 + 178", 413, 413, 8.0, "addition"),
        (7, "Hard", "15 × 8", 120, 120, 7.5, "multiplication"),
        (8, "Hard", "500 - 275", 225, 225, 7.0, "subtraction"),
        (9, "Hard", "32 × 4", 128, 128, 6.5, "multiplication"),
        (10, "Hard", "123 + 456", 579, 579, 6.0, "addition"),
        (11, "Medium", "84 - 29", 55, 55, 5.5, "subtraction"),
        (12, "Medium", "9 × 7", 63, 63, 5.0, "multiplication"),
    ]
    
    print("\nSimulating 12 problems:\n")
    
    for prob_num, diff, question, correct, answer, time, op in test_cases:
        tracker.log_attempt(prob_num, diff, question, correct, answer, time, op)
        
        next_diff, reasoning, confidence, details = hybrid.decide_next_difficulty(
            tracker, diff, prob_num
        )
        
        # Show phase info
        phase_info = hybrid.get_phase_info(prob_num)
        
        print(f"Problem {prob_num}: {question} = {answer}")
        print(f"  Phase: {phase_info['description']}")
        print(f"  Decision: {diff} → {next_diff}")
        print(f"  Method: {details['decision_method']}")
        print(f"  Confidence: {confidence*100:.0f}%")
        
        if prob_num >= hybrid.ml_start_threshold:
            print(f"  Rule prediction: {details['components']['rule_based']['prediction']}")
            print(f"  ML prediction: {details['components']['ml_based']['prediction']} "
                  f"({details['components']['ml_based']['confidence']*100:.0f}% conf)")
        
        print()
    
    # Show statistics
    print("="*60)
    print("SESSION STATISTICS")
    print("="*60)
    
    stats = hybrid.get_agreement_statistics()
    if stats:
        print(f"\nTotal decisions: {stats['total_decisions']}")
        print(f"Cold start phase: {stats['cold_start_decisions']} decisions")
        print(f"ML active phase: {stats['ml_active_decisions']} decisions")
        
        if stats['ml_active_decisions'] > 0:
            print(f"\nAgreement rate (ML phase): {stats['agreement_rate']*100:.1f}%")
            print(f"Agreements: {stats['agreements']}")
            print(f"Disagreements: {stats['disagreements']}")
            print(f"ML high confidence: {stats['ml_high_confidence']}")
            print(f"ML low confidence: {stats['ml_low_confidence']}")
        
        print(f"\nDecision method breakdown:")
        for method, count in sorted(stats['method_breakdown'].items()):
            print(f"  {method}: {count}")