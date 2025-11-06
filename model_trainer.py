import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import FeatureEngineer

class ModelTrainer:
    """
    Trains Random Forest model with SMOTE balancing
    Based on EDA notebook findings - 84.8% test accuracy
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.results = {}
        self.difficulty_labels = ["Easy", "Medium", "Hard"]
        
    def prepare_data(self, csv_file='synthetic_training_data.csv'):
        """
        Load and prepare data for training with SMOTE balancing
        
        Returns:
            tuple: (X_train_balanced, X_test, y_train_balanced, y_test, feature_names)
        """
        print("="*60)
        print("DATA PREPARATION")
        print("="*60)
        
        print("\n1. Loading data...")
        df = pd.read_csv(csv_file)
        print(f"   ✓ Loaded {len(df)} samples")
        
        print("\n2. Engineering features...")
        feature_df = self.feature_engineer.create_features_from_dataframe(df)
        print(f"   ✓ Created {len(feature_df)} training samples")
        
        # Separate features and target
        X = feature_df.drop('next_difficulty', axis=1)
        y = feature_df['next_difficulty']
        
        print(f"\n3. Feature set: {len(X.columns)} features")
        for i, col in enumerate(X.columns, 1):
            print(f"   {i}. {col}")
        
        print(f"\n4. Original target distribution:")
        dist = y.value_counts().sort_index()
        for idx, count in dist.items():
            label = self.difficulty_labels[idx]
            pct = count / len(y) * 100
            print(f"   {label}: {count} ({pct:.1f}%)")
        
        # Split data (stratified)
        print(f"\n5. Splitting data (80/20 train/test, stratified)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   ✓ Train: {len(X_train)} samples")
        print(f"   ✓ Test: {len(X_test)} samples")
        
        # Apply SMOTE to training data only
        print(f"\n6. Applying SMOTE to balance training data...")
        print(f"   Before SMOTE:")
        train_dist = y_train.value_counts().sort_index()
        for idx, count in train_dist.items():
            label = self.difficulty_labels[idx]
            print(f"   {label}: {count}")
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"\n   After SMOTE:")
        balanced_dist = pd.Series(y_train_balanced).value_counts().sort_index()
        for idx, count in balanced_dist.items():
            label = self.difficulty_labels[idx]
            print(f"   {label}: {count}")
        
        print(f"\n   ✓ Training samples increased: {len(X_train)} → {len(X_train_balanced)}")
        
        return X_train_balanced, X_test, y_train_balanced, y_test, X.columns.tolist()
    
    def train_model(self, X_train, y_train):
        """
        Train Random Forest model with optimal parameters from EDA
        Original config: max_depth=7, n_estimators=100, class_weight='balanced'
        Test accuracy: 84.8%
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        print("\nModel Configuration (from EDA notebook):")
        print("  • n_estimators: 100")
        print("  • max_depth: 7")
        print("  • min_samples_split: 5")
        print("  • class_weight: balanced")
        print("  • random_state: 42")
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        print("✓ Model trained successfully!")
        
        return self.model
    
    def evaluate_model(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Comprehensive model evaluation
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Accuracy scores
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\nAccuracy Scores:")
        print(f"  Training:   {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"  Testing:    {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Difference: {abs(train_acc - test_acc):.4f}")
        
        if abs(train_acc - test_acc) < 0.05:
            print("  ✓ Good generalization (low overfitting)")
        elif abs(train_acc - test_acc) < 0.10:
            print("  ⚠ Slight overfitting")
        else:
            print("  ✗ Significant overfitting detected")
        
        # Cross-validation
        print(f"\nCross-Validation (5-fold on training data):")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  Scores: {[f'{s:.3f}' for s in cv_scores]}")
        print(f"  Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Classification report
        print(f"\nClassification Report (Test Set):")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=self.difficulty_labels,
                                   digits=3))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print(f"Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Easy  Medium  Hard")
        for i, label in enumerate(self.difficulty_labels):
            print(f"  Actual {label:6s}  {cm[i, 0]:4d}  {cm[i, 1]:6d}  {cm[i, 2]:4d}")
        
        # Feature importance
        print(f"\nFeature Importance (Top 5):")
        importances = self.model.feature_importances_
        feature_imp = sorted(zip(feature_names, importances), 
                           key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(feature_imp[:5], 1):
            bar = '█' * int(imp * 50)
            print(f"  {i}. {feat:30s} {imp:.4f} {bar}")
        
        # Store results
        self.results = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'confusion_matrix': cm.tolist(),
            'feature_importance': dict(zip(feature_names, importances.tolist()))
        }
        
        return self.results
    
    def save_model(self, output_dir='models'):
        """Save trained model and metadata"""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        model_path = f"{output_dir}/random_forest_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Model saved: {model_path}")
        
        # Save feature names
        feature_names = self.feature_engineer.get_feature_names()
        feature_path = f"{output_dir}/feature_names.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"✓ Feature names saved: {feature_path}")
        
        # Save results
        results_path = f"{output_dir}/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved: {results_path}")
        
        # Save preprocessing info
        preprocessing_info = {
            'feature_order': feature_names,
            'smote_applied': True,
            'class_mapping': {0: 'Easy', 1: 'Medium', 2: 'Hard'},
            'model_type': 'RandomForest',
            'model_params': {
                'n_estimators': 100,
                'max_depth': 7,
                'min_samples_split': 5,
                'class_weight': 'balanced'
            }
        }
        prep_path = f"{output_dir}/preprocessing_info.pkl"
        with open(prep_path, 'wb') as f:
            pickle.dump(preprocessing_info, f)
        print(f"✓ Preprocessing info saved: {prep_path}")
    
    def generate_visualizations(self, X_test, y_test, feature_names):
        """Generate visualization plots"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Confusion Matrix
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.difficulty_labels,
                   yticklabels=self.difficulty_labels,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Random Forest', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/confusion_matrix.png")
        plt.close()
        
        # 2. Feature Importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance (Random Forest)', fontsize=16, fontweight='bold')
        plt.bar(range(len(importances)), importances[indices], color='steelblue')
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: visualizations/feature_importance.png")
        plt.close()


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("ADAPTIVE LEARNING MODEL TRAINER")
    print("Based on EDA Notebook Analysis")
    print("="*60)
    
    trainer = ModelTrainer()
    
    # Prepare data with SMOTE
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data()
    
    # Train Random Forest
    trainer.train_model(X_train, y_train)
    
    # Evaluate
    results = trainer.evaluate_model(X_train, X_test, y_train, y_test, feature_names)
    
    # Save everything
    trainer.save_model()
    
    # Generate visualizations
    trainer.generate_visualizations(X_test, y_test, feature_names)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n✓ Final Test Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"✓ Models saved in 'models/' directory")
    print(f"✓ Visualizations saved in 'visualizations/' directory")
    print("\nYou can now run: streamlit run main.py")


if __name__ == "__main__":
    main()