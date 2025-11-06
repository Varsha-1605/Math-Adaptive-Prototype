import pickle
import numpy as np
import pandas as pd
from feature_engineering import FeatureEngineer

class MLAdaptiveEngine:
    """
    Machine Learning based adaptive engine using trained Random Forest
    Uses 9 enhanced features from EDA notebook
    """

    def __init__(self, model_path='models/random_forest_model.pkl'):
        """
        Initialize ML engine with trained model
        
        Args:
            model_path: Path to pickled Random Forest model
        """
        self.feature_engineer = FeatureEngineer()
        self.model = self._load_model(model_path)
        self.feature_names = self.feature_engineer.get_feature_names()
        self.difficulty_levels = ["Easy", "Medium", "Hard"]
        self.decision_history = []
        
    def _load_model(self, model_path):
        """Load trained Random Forest model from file"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Loaded Random Forest model from {model_path}")
            return model
        except FileNotFoundError:
            print(f"⚠ Model file not found: {model_path}")
            print("  Run 'python model_trainer.py' first to train the model")
            return None
    
    def decide_next_difficulty(self, tracker, current_difficulty, problem_count):
        """
        Predict next difficulty using Random Forest with 9 features
        
        Args:
            tracker: PerformanceTracker object
            current_difficulty: Current difficulty level
            problem_count: Number of problems completed
            
        Returns:
            tuple: (next_difficulty, reasoning, confidence)
        """
        # Need minimum 3 problems for feature calculation
        if problem_count < 3:
            return current_difficulty, "Insufficient data (need 3+ problems)", 0.33
        
        if self.model is None:
            return current_difficulty, "ML model not available", 0.0
        
        # Create 9 enhanced features
        features_dict = self.feature_engineer.create_features_from_tracker(
            tracker, current_difficulty
        )
        
        if features_dict is None:
            return current_difficulty, "Feature creation failed", 0.0
        
        # Convert to DataFrame with correct feature order
        feature_values = [features_dict[fname] for fname in self.feature_names]
        X = pd.DataFrame([feature_values], columns=self.feature_names)
        
        # Predict using Random Forest
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        next_difficulty = self.difficulty_levels[int(prediction)]
        confidence = probabilities[int(prediction)]
        
        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(
            features_dict, current_difficulty, next_difficulty, confidence, probabilities
        )
        
        # Log decision
        self.decision_history.append({
            'problem_count': problem_count,
            'current_difficulty': current_difficulty,
            'next_difficulty': next_difficulty,
            'confidence': float(confidence),
            'reasoning': reasoning,
            'features': features_dict,
            'probabilities': {
                'Easy': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'Hard': float(probabilities[2])
            }
        })
        
        return next_difficulty, reasoning, confidence
    
    def _generate_reasoning(self, features, current_diff, next_diff, confidence, probs):
        """Generate human-readable explanation using enhanced features"""
        parts = []
        
        # 1. Confidence level
        if confidence > 0.7:
            parts.append(f"High confidence ({confidence*100:.0f}%)")
        elif confidence > 0.5:
            parts.append(f"Moderate confidence ({confidence*100:.0f}%)")
        else:
            parts.append(f"Low confidence ({confidence*100:.0f}%)")
        
        # 2. Accuracy assessment
        acc = features['accuracy_last_3']
        if acc >= 0.66:
            parts.append(f"strong accuracy ({acc*100:.0f}%)")
        elif acc >= 0.33:
            parts.append(f"moderate accuracy ({acc*100:.0f}%)")
        else:
            parts.append(f"low accuracy ({acc*100:.0f}%)")
        
        # 3. Speed assessment
        time = features['avg_time_last_3']
        if time < 7:
            parts.append("fast responses")
        elif time < 12:
            parts.append("moderate speed")
        else:
            parts.append("slow responses")
        
        # 4. Trend analysis (new feature)
        trend = features['accuracy_trend']
        if trend > 0.1:
            parts.append("improving trend")
        elif trend < -0.1:
            parts.append("declining trend")
        
        # 5. Efficiency (new feature)
        efficiency = features['speed_accuracy_ratio']
        if efficiency < 10:
            parts.append("efficient")
        elif efficiency > 20:
            parts.append("needs more time")
        
        # 6. Direction
        if next_diff != current_diff:
            curr_idx = self.difficulty_levels.index(current_diff)
            next_idx = self.difficulty_levels.index(next_diff)
            direction = "↑" if next_idx > curr_idx else "↓"
            parts.append(f"{direction} ML suggests {next_diff}")
        else:
            parts.append(f"maintaining {current_diff}")
        
        return ", ".join(parts)
    
    def get_feature_importance_for_decision(self):
        """
        Get key features that influenced the last decision
        
        Returns:
            dict: Feature values and importance
        """
        if not self.decision_history:
            return None
        
        last_decision = self.decision_history[-1]
        features = last_decision['features']
        
        # Get feature importances from model
        if hasattr(self.model, 'feature_importances_'):
            importances = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Return top 5 features with their values and importance
            top_features = {}
            for fname in sorted(importances, key=importances.get, reverse=True)[:5]:
                top_features[fname] = {
                    'value': features[fname],
                    'importance': importances[fname]
                }
            return top_features
        
        return features
    
    def explain_last_decision(self):
        """Get detailed explanation for most recent decision"""
        if self.decision_history:
            return self.decision_history[-1]
        return None
    
    def get_decision_history(self):
        """Get full decision history"""
        return self.decision_history
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return None
        
        return {
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': len(self.feature_names),
            'features': self.feature_names,
            'classes': self.difficulty_levels
        }


# Test the ML engine
if __name__ == "__main__":
    from tracker import PerformanceTracker
    
    print("="*60)
    print("TESTING ML ADAPTIVE ENGINE")
    print("="*60)
    
    # Create mock tracker with varied performance
    tracker = PerformanceTracker("Test Student", "Medium")
    
    # Simulate 5 attempts showing improvement
    print("\nScenario: Student showing improvement")
    tracker.log_attempt(1, "Medium", "23 + 15", 38, 35, 8.0, "addition")  # Wrong, slow
    tracker.log_attempt(2, "Medium", "45 - 18", 27, 27, 7.0, "subtraction")  # Correct
    tracker.log_attempt(3, "Medium", "7 × 6", 42, 42, 6.5, "multiplication")  # Correct, faster
    tracker.log_attempt(4, "Medium", "12 × 5", 60, 60, 6.0, "multiplication")  # Correct, faster
    tracker.log_attempt(5, "Medium", "48 - 23", 25, 25, 5.5, "subtraction")  # Correct, fast
    
    # Test ML engine
    ml_engine = MLAdaptiveEngine()
    
    if ml_engine.model is not None:
        print("\n" + "="*60)
        print("MODEL INFO")
        print("="*60)
        info = ml_engine.get_model_info()
        print(f"Model: {info['model_type']}")
        print(f"Trees: {info['n_estimators']}")
        print(f"Max Depth: {info['max_depth']}")
        print(f"Features: {info['n_features']}")
        
        print("\n" + "="*60)
        print("PREDICTION TEST")
        print("="*60)
        
        next_diff, reasoning, confidence = ml_engine.decide_next_difficulty(
            tracker, "Medium", 5
        )
        
        print(f"\nCurrent Difficulty: Medium")
        print(f"Next Difficulty: {next_diff}")
        print(f"Confidence: {confidence*100:.1f}%")
        print(f"Reasoning: {reasoning}")
        
        print(f"\n" + "="*60)
        print("TOP FEATURES FOR DECISION")
        print("="*60)
        top_features = ml_engine.get_feature_importance_for_decision()
        if top_features:
            for i, (fname, data) in enumerate(top_features.items(), 1):
                print(f"{i}. {fname:30s}: {data['value']:.3f} (importance: {data['importance']:.3f})")
        
        print(f"\n" + "="*60)
        print("PROBABILITY BREAKDOWN")
        print("="*60)
        last_decision = ml_engine.explain_last_decision()
        if last_decision:
            probs = last_decision['probabilities']
            for diff, prob in probs.items():
                bar = '█' * int(prob * 50)
                print(f"{diff:6s}: {prob*100:5.1f}% {bar}")
    else:
        print("\n⚠ Please train the model first:")
        print("  python model_trainer.py")
