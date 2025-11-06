# Math Adventures - AI-Powered Adaptive Learning

**An intelligent math tutoring system for children aged 5-10 that adapts difficulty in real-time using machine learning.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

---

## 🎯 Project Overview

This adaptive learning system uses a **Random Forest classifier** trained on synthetic student data to dynamically adjust math problem difficulty. The system achieves **96.8% test accuracy** (tuned version) using 9 enhanced features extracted from student performance patterns.

### Key Features

- 🤖 **Hybrid Adaptation**: Rule-based logic for cold start (problems 1-10), ML-powered for problems 11-20
- 📊 **9 Enhanced Features**: Including speed_accuracy_ratio (21% importance), accuracy trends, and consistency scores
- 🎓 **Age-Appropriate**: Problems designed for ages 5-10 (Kindergarten through Grade 5)
- 🔄 **SMOTE Balancing**: Addresses class imbalance in training data (73.9% → 33.3% Easy)
- 📈 **Real-Time Tracking**: Live performance visualization and decision explanations
- 🎯 **20 Problems per Session**: Extended practice with 10 rule-based, 10 ML-driven

---

## 📊 Performance Metrics (From EDA)

### Model Comparison

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| Logistic Regression | 87.9% | Simple baseline |
| Decision Tree | 91.8% | Prone to overfitting |
| Random Forest | 95.4% | Strong performance |
| **Random Forest (Tuned)** | **96.8%** | ⭐ Best model (our choice) |
| Gradient Boosting | 96.4% | High train accuracy (100%) |
| Ensemble | 96.4% | Slightly worse than RF Tuned |

**Why Random Forest Tuned?**
- Best test accuracy (96.8%)
- Good generalization (low overfitting: 1.9% gap)
- Optimal hyperparameters: n_estimators=50, max_depth=10
- Clean Medium class predictions (74.2% precision, 95.8% recall)

### Class-Specific Performance

| Difficulty | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| **Easy** | 100.0% | 96.6% | 0.983 | 207 |
| **Medium** | 74.2% | 95.8% | 0.836 | 24 |
| **Hard** | 98.0% | 98.0% | 0.980 | 49 |

**Key Insight**: SMOTE eliminated all Hard→Easy misclassifications!

---

## 🔍 Repository Structure

```
math-adaptive-prototype/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── generate_synthetic_data.py     # Creates training dataset (1000 samples)
├── feature_engineering.py         # 9 enhanced features from EDA
├── model_trainer.py              # Random Forest + SMOTE training pipeline
├── adaptive_engine.py            # Rule-based adaptation logic
├── ml_adaptive_engine.py         # ML-based predictions (Random Forest)
├── hybrid_engine.py              # Hybrid approach (10-problem threshold)
├── puzzle_generator.py           # Age-appropriate math problem generator
├── tracker.py                    # Performance tracking and statistics
├── main.py                       # Streamlit web interface (20 problems)
├── models/                       # Trained model files (generated)
│   ├── random_forest_model.pkl
│   
└── visualizations/               # Performance plots (generated)
    ├── confusion_matrix.png
    └── feature_importance.png
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd math-adaptive-prototype

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python generate_synthetic_data.py
```

**Output**: `synthetic_training_data.csv` (1000 samples, 5 student types)

### 3. Train the Model

```bash
python model_trainer.py
```

**Output**:
- `models/random_forest_model.pkl` - Trained Random Forest (Tuned)
- `models/feature_names.pkl` - Feature order
- `visualizations/confusion_matrix.png`
- `visualizations/feature_importance.png`

**Expected Results**:
```
Training Accuracy: 98.6%
Testing Accuracy:  96.8%
Cross-Validation:  98.0% ± 0.01
```

### 4. Run the Application

```bash
streamlit run main.py
```

**Access**: Open browser at `http://localhost:8501`

---

## 🧠 Adaptive Logic Explained

### Hybrid Approach (Recommended)

The system uses a **two-phase strategy** based on EDA findings:

#### Phase 1: Cold Start (Problems 1-10)
- **Engine**: Rule-based logic
- **Reason**: Insufficient data for reliable ML features
- **Logic**:
  - Accuracy ≥ 66% + fast → Increase difficulty
  - Accuracy ≤ 33% or very slow → Decrease difficulty
  - Otherwise → Maintain current level
- **Anti-bounce**: Wait 3 problems between adjustments

#### Phase 2: ML Active (Problems 11-20)
- **Engine**: Random Forest classifier (96.8% accuracy)
- **Features**: 9 enhanced features (see below)
- **Confidence Thresholds**:
  - \>70%: Trust ML fully
  - 50-70%: Prefer ML (moderate confidence)
  - <50%: Fallback to rule-based

**Why 10-problem threshold?**
- EDA showed 20% agreement rate with 5-problem threshold
- Need more data for stable ML predictions
- Rule-based performs well for first 10 problems
- Hybrid approach balances cold start vs. ML benefits

### 9 Enhanced Features (From EDA Cell 2)

Based on EDA notebook feature engineering with importance from tuned Random Forest:

| # | Feature Name | Description | Importance | Notes |
|---|--------------|-------------|------------|-------|
| 1 | **speed_accuracy_ratio** | avg_time / (accuracy + 0.01) | **32.1%** ⭐ | Most important! |
| 2 | **accuracy_last_3** | Recent accuracy (0-1) | 21.6% | Strong predictor |
| 3 | **avg_time_last_3** | Average time (seconds) | 13.2% | Speed matters |
| 4 | **current_difficulty** | Encoded (0=Easy, 1=Medium, 2=Hard) | 10.0% | Baseline |
| 5 | **current_difficulty_squared** | Non-linear effects | 9.9% | Captures jumps |
| 6 | **accuracy_trend** | Improvement over time | 5.1% | Learning rate |
| 7 | **time_improvement** | Getting faster? | 4.2% | Speed trend |
| 8 | **consistency_score** | 1 / (time_std + 0.1) | 2.3% | Stability |
| 9 | **std_time_last_3** | Time consistency | 1.6% | Variance |

**Key Finding**: Speed-accuracy ratio (efficiency) is MORE important than current difficulty level!

---

## 📈 EDA Findings Implementation

This implementation directly follows the EDA notebook analysis:

### What We Implemented

1. ✅ **Random Forest Tuned** (96.8% accuracy)
   - Best hyperparameters: n_estimators=50, max_depth=10, min_samples_split=5
   - Outperforms ensemble by 0.4 percentage points

2. ✅ **9 Enhanced Features** (Cell 2)
   - All features from EDA implemented
   - speed_accuracy_ratio is top feature (32.1% importance)
   - Feature engineering added 5 new informative features

3. ✅ **SMOTE Balancing** (Cell 3)
   - Fixed class imbalance: Easy 73.9% → 33.3%
   - Improved Hard class F1-score significantly
   - Eliminated all Hard→Easy misclassifications

4. ✅ **10-Problem Threshold** (Updated from 5)
   - EDA showed 20% agreement rate with 5-problem threshold
   - Extended to 10 problems for stable rule-based learning
   - 20 total problems (10 rule-based, 10 ML-driven)

5. ✅ **Feature Importance Analysis** (Cell 7)
   - speed_accuracy_ratio dominates (32.1%)
   - Accuracy features: 58.8% total importance
   - Time features: 19.0% total importance

### Improvements from EDA

| Stage | Accuracy | Improvement |
|-------|----------|-------------|
| Baseline (Original) | 81.7% | - |
| After SMOTE | 83.0% | +1.3% |
| After Feature Engineering | 86.0% | +3.0% |
| After Hyperparameter Tuning | 88.0% | +2.0% |
| **Random Forest Tuned** | **96.8%** | **+8.8%** |
| **Total Gain** | - | **+15.1 points** |

---

## 🎓 Age-Appropriate Problem Design

### Easy Level (Ages 5-6)
- **Grade**: Kindergarten - Grade 1
- **Operations**: 
  - Single-digit addition (sum ≤ 10)
  - Single-digit subtraction (no negatives)
- **Example**: `3 + 4 = 7`, `8 - 3 = 5`

### Medium Level (Ages 7-8)
- **Grade**: Grade 2-3
- **Operations**:
  - Two-digit addition/subtraction
  - Times tables 2-5
  - Simple division (clean results)
- **Example**: `23 + 34 = 57`, `3 × 7 = 21`, `15 ÷ 3 = 5`

### Hard Level (Ages 9-10)
- **Grade**: Grade 4-5
- **Operations**:
  - Two-digit addition (can exceed 100)
  - Times tables 6-12
  - Division with larger numbers
- **Example**: `67 + 89 = 156`, `9 × 8 = 72`, `72 ÷ 8 = 9`

---

## 🧪 Testing

### Test Feature Engineering

```bash
python feature_engineering.py
```

**Output**: Creates 9-feature training samples from synthetic data.

### Test ML Engine

```bash
python ml_adaptive_engine.py
```

**Output**: Shows prediction for mock student scenario with top features.

### Test Hybrid Engine

```bash
python hybrid_engine.py
```

**Output**: Simulates 12 problems showing phase transition at problem 10.

---

## 🔧 Customization

### Adjust Total Problems

Edit `main.py`:

```python
st.session_state.total_problems = 20  # Change to desired number
```

### Adjust ML Confidence Thresholds

Edit `hybrid_engine.py`:

```python
self.ml_confidence_threshold = 0.7  # High confidence
self.low_confidence_threshold = 0.5  # Low confidence
```

### Change Hybrid Threshold

Edit `hybrid_engine.py`:

```python
self.ml_start_threshold = 10  # Switch to ML after 10 problems
```

### Modify Problem Difficulty

Edit `puzzle_generator.py` ranges for each difficulty level.

---

## 📊 Sample Outputs

### Training Output

```
DATA PREPARATION
================
✓ Loaded 1000 samples
✓ Created 700 training samples
✓ Train: 560 samples → 1365 (after SMOTE)
✓ Test: 140 samples

TRAINING RANDOM FOREST MODEL
=============================
✓ Model trained successfully!

MODEL EVALUATION
================
Test Accuracy:    96.8%
Cross-Validation: 98.0% ± 1.0%
✓ Good generalization (low overfitting: 1.9%)

Feature Importance (Top 5):
1. speed_accuracy_ratio      0.3214 ████████████████████
2. accuracy_last_3            0.2163 █████████████
3. avg_time_last_3            0.1317 ████████
4. current_difficulty         0.1004 ██████
5. current_difficulty_squared 0.0986 ██████
```

### Streamlit UI Features

- ✅ Real-time phase indicator (Cold Start / ML Active)
- ✅ Live difficulty progression chart
- ✅ Decision method breakdown with confidence
- ✅ Top 3 feature values displayed after each decision
- ✅ Agreement/disagreement indicators
- ✅ Comprehensive session summary with visualizations
- ✅ 20-problem session with 10/10 rule/ML split

---

## 🤔 Discussion Questions

### 1. How would you collect real data?

**Answer**:
- Deploy as pilot in 2-3 schools (ages 5-10)
- Log all interactions: problem, answer, time, difficulty transitions
- Track long-term retention (1 week, 1 month follow-ups)
- Collect teacher feedback on difficulty appropriateness
- A/B test: 50% ML, 50% rule-based, compare learning outcomes
- Use adaptive sampling (oversample rare transitions like Easy→Hard)

### 2. Handling noisy/inconsistent performance?

**Implementation** (Already in code):
- Use exponential moving average (EMA) for recent metrics
- Minimum 10-problem observation window before ML activation
- Confidence thresholds: Only adjust if ML >70% confident
- Smooth transitions: Max 1 difficulty level change per decision
- Fallback to rules when ML confidence <50%
- Anti-bounce mechanism: Wait 3 problems between adjustments

### 3. Trade-offs: Rule-Based vs ML?

| Aspect | Rule-Based | ML-Based (RF Tuned) |
|--------|------------|---------------------|
| **Interpretability** | ⭐⭐⭐⭐⭐ Fully transparent | ⭐⭐ Requires feature importance |
| **Performance** | ⭐⭐⭐ ~80% accuracy | ⭐⭐⭐⭐⭐ 96.8% accuracy |
| **Cold Start** | ⭐⭐⭐⭐⭐ Works immediately | ⭐⭐ Needs 10+ problems |
| **Adaptability** | ⭐⭐ Fixed logic | ⭐⭐⭐⭐⭐ Learns patterns |
| **Maintenance** | ⭐⭐⭐ Manual updates | ⭐⭐⭐⭐ Self-improving |
| **Speed** | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐⭐ Fast inference (<50ms) |
| **Data Dependency** | ⭐⭐⭐⭐⭐ None needed | ⭐⭐ Needs 1000+ samples |

**Hybrid Approach** combines strengths: Rules for cold start (1-10), ML for learned adaptation (11-20).

### 4. Scaling to other subjects?

**Strategy**:
- Keep features domain-agnostic (accuracy, time, consistency)
- Add subject-specific features:
  - Reading: `reading_level`, `vocab_complexity`, `comprehension_score`
  - Science: `concept_depth`, `lab_completion_time`, `question_type`
- Train separate models per subject initially
- Build meta-model to transfer learning across subjects
- Share student behavior patterns (e.g., "quick learner" persona transfers)

---

## 🛠 Troubleshooting

### Model not found error

```bash
python model_trainer.py  # Train model first
```

### Import errors

```bash
pip install -r requirements.txt --upgrade
```

### SMOTE import error

```bash
pip install imbalanced-learn
```

### Streamlit port already in use

```bash
streamlit run main.py --server.port 8502
```

---

## 📚 Key Insights from EDA

### 1. Feature Importance Breakdown
- **Accuracy-related**: 58.8% (dominant)
- **Time-related**: 19.0%
- **Difficulty-related**: 19.9%
- **Other**: 2.3%

### 2. Confusion Matrix Analysis
- Easy: 199/207 correct (96.1% recall)
- Medium: 23/24 correct (95.8% recall) ← SMOTE fixed this!
- Hard: 48/49 correct (98.0% recall)
- **Main errors**: 8 Easy misclassified as Medium (conservative approach)

### 3. Learning Curve
- Validation accuracy plateaus at ~98% with 100% data
- Small train-validation gap (1.9%) → good generalization
- More data won't help much (diminishing returns)

### 4. Rule-ML Agreement
- Only **20%** agreement rate in EDA (5-problem threshold)
- Reason: Rule-based waits for stability, ML is aggressive
- Solution: Extend to 10-problem threshold for better stability


---

## 👥 Author

Created by **Varsha Dewangan** for Adaptive Learning AI Internship Assignment

**Contact**: varshadewangan1605@gmail.com

---

## 🙏 Acknowledgments

- EDA notebook analysis for feature engineering insights (96.8% accuracy)
- SMOTE implementation from `imbalanced-learn`
- Streamlit for rapid prototyping
- scikit-learn Random Forest implementation
- Thanks to the EDA findings for optimal hyperparameters!

---

## 📚 References

1. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique
2. Breiman, L. (2001). Random Forests. Machine Learning
3. VanLehn, K. (2011). The Relative Effectiveness of Human Tutoring, Intelligent Tutoring Systems

---

## 📌 Quick Reference

### Configuration Summary
- **Total Problems**: 20
- **Rule-Based Phase**: Problems 1-10
- **ML Phase**: Problems 11-20
- **Model**: Random Forest (Tuned)
- **Test Accuracy**: 96.8%
- **Features**: 9 enhanced features
- **Top Feature**: speed_accuracy_ratio (32.1%)
- **ML Confidence Threshold**: 70% (high), 50% (low)
- **Anti-Bounce**: 3 problems between adjustments



---

**Last Updated**: November 2025 | **Version**: 1.0 (10-problem threshold, 20 total problems)