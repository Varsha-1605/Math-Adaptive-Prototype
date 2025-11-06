# import streamlit as st
# import time
# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd
# from puzzle_generator import PuzzleGenerator
# from tracker import PerformanceTracker
# from adaptive_engine import AdaptiveEngine
# from ml_adaptive_engine import MLAdaptiveEngine
# from hybrid_engine import HybridAdaptiveEngine

# # Page configuration
# st.set_page_config(
#     page_title="Math Adventures - AI Adaptive Learning",
#     page_icon="🎓",
#     layout="wide"
# )

# # Initialize session state
# def initialize_session():
#     if 'initialized' not in st.session_state:
#         st.session_state.initialized = True
#         st.session_state.started = False
#         st.session_state.complete = False
#         st.session_state.problem_count = 0
#         st.session_state.total_problems = 10
#         st.session_state.current_problem = None
#         st.session_state.start_time = None
#         st.session_state.user_name = ""
#         st.session_state.current_difficulty = "Medium"
#         st.session_state.adaptation_mode = "Hybrid"
#         st.session_state.generator = PuzzleGenerator()
#         st.session_state.tracker = None
#         st.session_state.rule_engine = AdaptiveEngine()
#         st.session_state.ml_engine = MLAdaptiveEngine()
#         st.session_state.hybrid_engine = HybridAdaptiveEngine()
#         st.session_state.feedback_shown = False
#         st.session_state.last_correct = None
#         st.session_state.last_decision_details = None

# def welcome_screen():
#     """Display welcome screen with configuration"""
#     st.title("🎓 Math Adventures")
#     st.subheader("AI-Powered Adaptive Learning System for Ages 5-10")
    
#     st.markdown("""
#     Welcome to Math Adventures! This intelligent system adapts to your child's learning speed using **AI**.
    
#     **Features:**
#     - 🤖 Smart adaptation using Random Forest ML model
#     - 📊 Real-time performance tracking with 9 features
#     - 🎯 Age-appropriate problems for ages 5-10
#     - 🔄 Hybrid approach: Rule-based start, ML-powered continuation
#     """)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         name = st.text_input("👤 Student's name:", placeholder="Enter name")
        
#         difficulty = st.selectbox(
#             "📊 Starting difficulty:",
#             ["Easy", "Medium", "Hard"],
#             index=1,
#             help="Easy: Ages 5-6 | Medium: Ages 7-8 | Hard: Ages 9-10"
#         )
        
#         # Show difficulty info
#         info = st.session_state.generator.get_difficulty_info(difficulty)
#         st.info(f"""
#         **{difficulty} Level** ({info['age_range']})
#         - Grade: {info['grade']}
#         - Example: {info['example']}
#         """)
    
#     with col2:
#         mode = st.selectbox(
#             "🤖 Adaptation Mode:",
#             ["Rule-Based", "ML-Based", "Hybrid (Recommended)"],
#             index=2,
#             help="Hybrid: Rule-based for first 5 problems, then ML-powered"
#         )
        
#         mode_descriptions = {
#             "Rule-Based": "Traditional logic rules. Clear and predictable.",
#             "ML-Based": "Random Forest model trained on 1000+ samples. Data-driven.",
#             "Hybrid": "**Best of both worlds!** Rules for cold start (problems 1-5), then ML takes over."
#         }
        
#         st.success(f"""
#         **{mode.split()[0]} Mode:**
        
#         {mode_descriptions[mode.split()[0]]}
#         """)
        
#         if mode.split()[0] == "Hybrid":
#             st.info("🎯 **Hybrid Strategy:**\n- Problems 1-5: Rule-based learning\n- Problem 6+: ML-powered adaptation")
    
#     st.markdown("---")
    
#     if st.button("🚀 Start Learning!", type="primary", width='stretch'):
#         if name.strip():
#             st.session_state.user_name = name
#             st.session_state.current_difficulty = difficulty
#             st.session_state.adaptation_mode = mode.split()[0]
#             st.session_state.tracker = PerformanceTracker(name, difficulty)
#             st.session_state.started = True
#             st.session_state.problem_count = 0
#             generate_new_problem()
#             st.rerun()
#         else:
#             st.error("Please enter the student's name!")

# def generate_new_problem():
#     """Generate a new problem at current difficulty"""
#     question, answer, operation = st.session_state.generator.generate(
#         st.session_state.current_difficulty
#     )
    
#     st.session_state.current_problem = {
#         'question': question,
#         'answer': answer,
#         'operation': operation
#     }
#     st.session_state.start_time = time.time()
#     st.session_state.feedback_shown = False

# def display_problem():
#     """Display current problem and handle answer submission"""
    
#     # Sidebar with progress and stats
#     with st.sidebar:
#         st.header("📊 Session Info")
#         st.metric("Student", st.session_state.user_name)
#         st.metric("Mode", st.session_state.adaptation_mode)
        
#         # Phase indicator for Hybrid mode
#         if st.session_state.adaptation_mode == "Hybrid":
#             phase_info = st.session_state.hybrid_engine.get_phase_info(st.session_state.problem_count)
#             if phase_info['phase'] == 'cold_start':
#                 st.info(f"🎯 **Phase:** Cold Start\n\nUsing rule-based logic\n\n({phase_info['problems_until_ml']} more until ML activates)")
#             else:
#                 st.success(f"🤖 **Phase:** ML Active\n\nRandom Forest is adapting!")
        
#         st.metric("Problems", 
#                  f"{st.session_state.problem_count}/{st.session_state.total_problems}")
        
#         progress = st.session_state.problem_count / st.session_state.total_problems
#         st.progress(progress)
        
#         st.markdown("---")
#         st.subheader("📈 Current Stats")
        
#         if st.session_state.problem_count > 0:
#             stats = st.session_state.tracker.calculate_session_stats()
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Accuracy", f"{stats['accuracy_percentage']:.0f}%")
#                 st.metric("Problems", stats['total_problems'])
#             with col2:
#                 st.metric("Avg Time", f"{stats['average_time']:.1f}s")
#                 st.metric("Level", st.session_state.current_difficulty)
            
#             # Mini difficulty progression chart
#             if len(stats['difficulty_progression']) > 1:
#                 st.markdown("**Difficulty Trend:**")
#                 difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
#                 values = [difficulty_map[d] for d in stats['difficulty_progression']]
                
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatter(
#                     x=list(range(1, len(values) + 1)),
#                     y=values,
#                     mode='lines+markers',
#                     line=dict(color='#FF4B4B', width=2),
#                     marker=dict(size=6)
#                 ))
#                 fig.update_layout(
#                     height=150,
#                     margin=dict(l=0, r=0, t=0, b=0),
#                     yaxis=dict(tickmode='array', tickvals=[1, 2, 3], 
#                               ticktext=['E', 'M', 'H'], range=[0.5, 3.5]),
#                     xaxis=dict(showticklabels=False),
#                     showlegend=False
#                 )
#                 st.plotly_chart(fig, width='stretch')
    
#     # Main problem area
#     st.title("🎓 Math Adventures")
    
#     difficulty_colors = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
    
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.markdown(f"### Problem {st.session_state.problem_count + 1} of {st.session_state.total_problems}")
#     with col2:
#         st.markdown(f"**Difficulty:** {difficulty_colors[st.session_state.current_difficulty]} {st.session_state.current_difficulty}")
    
#     st.markdown("---")
    
#     problem = st.session_state.current_problem
    
#     if not st.session_state.feedback_shown:
#         # Show problem
#         st.markdown(f"## {problem['question']} = ?")
        
#         col1, col2, col3 = st.columns([2, 1, 2])
        
#         with col2:
#             user_answer = st.number_input(
#                 "Your Answer:",
#                 value=None,
#                 step=1,
#                 key=f"answer_{st.session_state.problem_count}"
#             )
            
#             if st.button("Submit Answer", type="primary", width='stretch'):
#                 if user_answer is not None:
#                     check_answer(user_answer)
#                     st.rerun()
#                 else:
#                     st.error("Please enter an answer!")
    
#     else:
#         # Show feedback
#         display_feedback()

# def display_feedback():
#     """Display feedback and decision details with enhanced features"""
#     time_taken = time.time() - st.session_state.start_time
#     is_correct = st.session_state.last_correct
#     problem = st.session_state.current_problem
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         if is_correct:
#             st.success("✅ Correct! Excellent work!")
#             st.balloons()
#         else:
#             st.error(f"❌ Not quite. The correct answer is **{problem['answer']}**")
        
#         st.info(f"⏱️ Time taken: **{time_taken:.1f} seconds**")
    
#     with col2:
#         # Show decision details if available
#         if st.session_state.last_decision_details:
#             details = st.session_state.last_decision_details
            
#             st.metric("AI Decision", details['final_difficulty'])
#             st.metric("Confidence", f"{details['confidence']*100:.0f}%")
            
#             if details['final_difficulty'] != details['current_difficulty']:
#                 direction = "📈" if details['final_difficulty'] == "Hard" or \
#                            (details['final_difficulty'] == "Medium" and details['current_difficulty'] == "Easy") else "📉"
#                 st.markdown(f"{direction} **Difficulty Adjusted!**")
    
#     # Detailed decision breakdown with 9 enhanced features
#     if st.session_state.last_decision_details and st.session_state.problem_count >= 3:
#         st.markdown("---")
        
#         with st.expander("🤖 AI Decision Details (9 Enhanced Features)", expanded=True):
#             details = st.session_state.last_decision_details
            
#             # Show phase for hybrid mode
#             if st.session_state.adaptation_mode == "Hybrid":
#                 phase = details.get('phase', 'unknown')
#                 if phase == 'cold_start':
#                     st.info("📍 **Cold Start Phase** - Using rule-based logic")
#                 else:
#                     st.success("🤖 **ML Active Phase** - Random Forest predictions")
            
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.markdown("**Rule-Based**")
#                 st.write(f"Prediction: {details['components']['rule_based']['prediction']}")
#                 st.caption(details['components']['rule_based']['reasoning'][:80] + "...")
            
#             with col2:
#                 st.markdown("**ML-Based (Random Forest)**")
#                 ml_comp = details['components']['ml_based']
#                 st.write(f"Prediction: {ml_comp['prediction']}")
#                 st.write(f"Confidence: {ml_comp['confidence']*100:.1f}%")
#                 st.caption(ml_comp['reasoning'][:80] + "...")
            
#             with col3:
#                 st.markdown("**Final Decision**")
#                 st.write(f"Method: {details['decision_method'].replace('_', ' ').title()}")
#                 st.write(f"Result: {details['final_difficulty']}")
                
#                 # Agreement indicator
#                 if details['components']['rule_based']['prediction'] == \
#                    details['components']['ml_based']['prediction']:
#                     st.success("✅ Consensus")
#                 else:
#                     st.warning("⚠️ Disagreement")
            
#             # Show ML features if available (for ML-Based or Hybrid in ML phase)
#             if st.session_state.adaptation_mode in ["ML-Based", "Hybrid"] and \
#                st.session_state.problem_count >= 5:
#                 st.markdown("---")
#                 st.markdown("**🔍 Enhanced Features Used:**")
                
#                 # Get feature importance from ML engine
#                 top_features = st.session_state.ml_engine.get_feature_importance_for_decision()
                
#                 if top_features:
#                     feature_cols = st.columns(3)
#                     for i, (fname, data) in enumerate(list(top_features.items())[:3]):
#                         with feature_cols[i]:
#                             st.metric(
#                                 fname.replace('_', ' ').title(),
#                                 f"{data['value']:.2f}",
#                                 delta=f"Importance: {data['importance']*100:.1f}%"
#                             )
    
#     st.markdown("---")
    
#     if st.session_state.problem_count < st.session_state.total_problems:
#         if st.button("Next Problem →", type="primary", width='stretch'):
#             generate_new_problem()
#             st.rerun()
#     else:
#         st.success("🎉 Session Complete!")
#         if st.button("View Results", type="primary", width='stretch'):
#             st.session_state.complete = True
#             st.rerun()

# def check_answer(user_answer):
#     """Check answer and apply adaptive logic"""
#     time_taken = time.time() - st.session_state.start_time
#     problem = st.session_state.current_problem
    
#     st.session_state.problem_count += 1
    
#     # Log attempt
#     st.session_state.tracker.log_attempt(
#         st.session_state.problem_count,
#         st.session_state.current_difficulty,
#         problem['question'],
#         problem['answer'],
#         user_answer,
#         time_taken,
#         problem['operation']
#     )
    
#     st.session_state.last_correct = (user_answer == problem['answer'])
#     st.session_state.feedback_shown = True
    
#     # Apply adaptive logic based on mode (minimum 3 problems for features)
#     if st.session_state.problem_count >= 3:
#         mode = st.session_state.adaptation_mode
        
#         if mode == "Rule-Based":
#             next_diff, reasoning = st.session_state.rule_engine.decide_next_difficulty(
#                 st.session_state.tracker,
#                 st.session_state.current_difficulty,
#                 st.session_state.problem_count
#             )
#             st.session_state.last_decision_details = {
#                 'current_difficulty': st.session_state.current_difficulty,
#                 'final_difficulty': next_diff,
#                 'confidence': 0.7,
#                 'decision_method': 'rule_based',
#                 'components': {
#                     'rule_based': {
#                         'prediction': next_diff,
#                         'reasoning': reasoning
#                     },
#                     'ml_based': {
#                         'prediction': 'N/A',
#                         'confidence': 0.0,
#                         'reasoning': 'Not used in Rule-Based mode'
#                     }
#                 }
#             }
            
#         elif mode == "ML-Based":
#             next_diff, reasoning, confidence = st.session_state.ml_engine.decide_next_difficulty(
#                 st.session_state.tracker,
#                 st.session_state.current_difficulty,
#                 st.session_state.problem_count
#             )
#             st.session_state.last_decision_details = {
#                 'current_difficulty': st.session_state.current_difficulty,
#                 'final_difficulty': next_diff,
#                 'confidence': confidence,
#                 'decision_method': 'ml_based',
#                 'components': {
#                     'rule_based': {
#                         'prediction': 'N/A',
#                         'reasoning': 'Not used in ML-Based mode'
#                     },
#                     'ml_based': {
#                         'prediction': next_diff,
#                         'confidence': confidence,
#                         'reasoning': reasoning
#                     }
#                 }
#             }
            
#         else:  # Hybrid - uses 5-problem threshold
#             next_diff, reasoning, confidence, details = st.session_state.hybrid_engine.decide_next_difficulty(
#                 st.session_state.tracker,
#                 st.session_state.current_difficulty,
#                 st.session_state.problem_count
#             )
#             st.session_state.last_decision_details = details
        
#         st.session_state.current_difficulty = next_diff

# def show_summary():
#     """Display comprehensive session summary"""
#     st.title("📊 Session Summary")
#     st.subheader(f"Excellent work, {st.session_state.user_name}! 🎉")
    
#     stats = st.session_state.tracker.calculate_session_stats()
    
#     # Key metrics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.metric("Total Problems", stats['total_problems'])
#     with col2:
#         st.metric("Accuracy", f"{stats['accuracy_percentage']:.1f}%",
#                  delta=f"{stats['correct']} correct")
#     with col3:
#         st.metric("Avg Time", f"{stats['average_time']:.1f}s")
#     with col4:
#         st.metric("Final Level", stats['final_difficulty'], 
#                  f"{stats['starting_difficulty']} → {stats['final_difficulty']}")
    
#     st.markdown("---")
    
#     # Hybrid-specific statistics
#     if st.session_state.adaptation_mode == "Hybrid":
#         st.subheader("🤖 Hybrid Engine Performance")
#         hybrid_stats = st.session_state.hybrid_engine.get_agreement_statistics()
        
#         if hybrid_stats:
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Cold Start", f"{hybrid_stats['cold_start_decisions']} problems")
#             with col2:
#                 st.metric("ML Active", f"{hybrid_stats['ml_active_decisions']} problems")
#             with col3:
#                 if hybrid_stats['ml_active_decisions'] > 0:
#                     st.metric("Agreement Rate", f"{hybrid_stats['agreement_rate']*100:.0f}%")
#             with col4:
#                 st.metric("ML High Confidence", hybrid_stats['ml_high_confidence'])
        
#         st.markdown("---")
    
#     # Visualizations
#     tab1, tab2, tab3 = st.tabs(["📈 Performance", "🤖 AI Decisions", "📝 Problem Log"])
    
#     with tab1:
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Difficulty Progression")
#             difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
#             difficulty_values = [difficulty_map[d] for d in stats['difficulty_progression']]
            
#             fig1 = go.Figure()
#             fig1.add_trace(go.Scatter(
#                 x=list(range(1, len(difficulty_values) + 1)),
#                 y=difficulty_values,
#                 mode='lines+markers',
#                 line=dict(color='#FF4B4B', width=3),
#                 marker=dict(size=10),
#                 fill='tozeroy',
#                 fillcolor='rgba(255, 75, 75, 0.2)'
#             ))
#             fig1.update_layout(
#                 xaxis_title="Problem Number",
#                 yaxis_title="Difficulty Level",
#                 yaxis=dict(tickmode='array', tickvals=[1, 2, 3], 
#                           ticktext=['Easy', 'Medium', 'Hard']),
#                 height=350
#             )
#             st.plotly_chart(fig1, width='stretch')
        
#         with col2:
#             st.subheader("Problems per Difficulty")
            
#             fig2 = go.Figure(data=[
#                 go.Bar(
#                     x=list(stats['problems_per_difficulty'].keys()),
#                     y=list(stats['problems_per_difficulty'].values()),
#                     marker_color=['#90EE90', '#FFD700', '#FF6B6B'],
#                     text=list(stats['problems_per_difficulty'].values()),
#                     textposition='auto'
#                 )
#             ])
#             fig2.update_layout(
#                 xaxis_title="Difficulty Level",
#                 yaxis_title="Number of Problems",
#                 height=350
#             )
#             st.plotly_chart(fig2, width='stretch')
    
#     with tab2:
#         st.subheader("AI Decision Analysis")
        
#         # Get decision history
#         if st.session_state.adaptation_mode == "Hybrid":
#             history = st.session_state.hybrid_engine.get_decision_history()
#         elif st.session_state.adaptation_mode == "ML-Based":
#             history = st.session_state.ml_engine.get_decision_history()
#         else:
#             history = st.session_state.rule_engine.get_decision_history()
        
#         if history:
#             st.markdown("### Decision History")
#             df_history = pd.DataFrame([
#                 {
#                     'Problem': h['problem_count'],
#                     'Current': h['current_difficulty'],
#                     'Decision': h.get('final_difficulty', h.get('next_difficulty')),
#                     'Confidence': f"{h.get('confidence', 0)*100:.0f}%",
#                     'Method': h.get('decision_method', 'rule_based').replace('_', ' ').title(),
#                     'Reasoning': h['reasoning'][:60] + "..." if len(h['reasoning']) > 60 else h['reasoning']
#                 }
#                 for h in history
#             ])
#             st.dataframe(df_history, width='stretch')
    
#     with tab3:
#         st.subheader("Detailed Problem Log")
        
#         problem_data = []
#         for attempt in st.session_state.tracker.session_data:
#             problem_data.append({
#                 "Problem": attempt['problem_number'],
#                 "Difficulty": attempt['difficulty'],
#                 "Question": attempt['question'],
#                 "Your Answer": attempt['user_answer'],
#                 "Correct": attempt['correct_answer'],
#                 "Result": "✅" if attempt['is_correct'] else "❌",
#                 "Time (s)": f"{attempt['time_taken']:.1f}"
#             })
        
#         df_problems = pd.DataFrame(problem_data)
#         st.dataframe(df_problems, width='stretch')
    
#     # Learning trend
#     st.markdown("---")
#     trend = st.session_state.tracker.get_performance_trend()
#     if trend == "improving":
#         st.success("📈 Excellent! The student showed clear improvement throughout the session!")
#     elif trend == "declining":
#         st.info("💪 Keep practicing! Everyone has challenging sessions.")
#     else:
#         st.info("➡️ Consistent performance maintained throughout the session.")
    
#     # Actions
#     st.markdown("---")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("💾 Export Session Data", width='stretch'):
#             filename = st.session_state.tracker.export_session()
#             st.success(f"✅ Data exported to {filename}")
    
#     with col2:
#         if st.button("🔄 Start New Session", type="primary", width='stretch'):
#             for key in list(st.session_state.keys()):
#                 del st.session_state[key]
#             st.rerun()

# def main():
#     """Main application flow"""
#     initialize_session()
    
#     if not st.session_state.started:
#         welcome_screen()
#     elif not st.session_state.complete:
#         display_problem()
#     else:
#         show_summary()

# if __name__ == "__main__":
#     main()














import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from puzzle_generator import PuzzleGenerator
from tracker import PerformanceTracker
from adaptive_engine import AdaptiveEngine
from ml_adaptive_engine import MLAdaptiveEngine
from hybrid_engine import HybridAdaptiveEngine

# Page configuration
st.set_page_config(
    page_title="Math Adventures - AI Adaptive Learning",
    page_icon="🎓",
    layout="wide"
)

# Initialize session state
def initialize_session():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.started = False
        st.session_state.complete = False
        st.session_state.problem_count = 0
        st.session_state.total_problems = 20  # CHANGED: 10 → 20
        st.session_state.current_problem = None
        st.session_state.start_time = None
        st.session_state.user_name = ""
        st.session_state.current_difficulty = "Medium"
        st.session_state.adaptation_mode = "Hybrid"
        st.session_state.generator = PuzzleGenerator()
        st.session_state.tracker = None
        st.session_state.rule_engine = AdaptiveEngine()
        st.session_state.ml_engine = MLAdaptiveEngine()
        st.session_state.hybrid_engine = HybridAdaptiveEngine()
        st.session_state.feedback_shown = False
        st.session_state.last_correct = None
        st.session_state.last_decision_details = None

def welcome_screen():
    """Display welcome screen with configuration"""
    st.title("🎓 Math Adventures")
    st.subheader("AI-Powered Adaptive Learning System for Ages 5-10")
    
    st.markdown("""
    Welcome to Math Adventures! This intelligent system adapts to your child's learning speed using **AI**.
    
    **Features:**
    - 🤖 Smart adaptation using Random Forest ML model (96.8% accuracy)
    - 📊 Real-time performance tracking with 9 enhanced features
    - 🎯 Age-appropriate problems for ages 5-10
    - 🔄 Hybrid approach: Rule-based start (10 problems), ML-powered continuation
    - 📈 20 total problems per session
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("👤 Student's name:", placeholder="Enter name")
        
        difficulty = st.selectbox(
            "📊 Starting difficulty:",
            ["Easy", "Medium", "Hard"],
            index=1,
            help="Easy: Ages 5-6 | Medium: Ages 7-8 | Hard: Ages 9-10"
        )
        
        # Show difficulty info
        info = st.session_state.generator.get_difficulty_info(difficulty)
        st.info(f"""
        **{difficulty} Level** ({info['age_range']})
        - Grade: {info['grade']}
        - Example: {info['example']}
        """)
    
    with col2:
        mode = st.selectbox(
            "🤖 Adaptation Mode:",
            ["Rule-Based", "ML-Based", "Hybrid (Recommended)"],
            index=2,
            help="Hybrid: Rule-based for first 10 problems, then ML-powered"
        )
        
        mode_descriptions = {
            "Rule-Based": "Traditional logic rules. Clear and predictable.",
            "ML-Based": "Random Forest model (96.8% accuracy). Data-driven from problem 3.",
            "Hybrid": "**Best of both worlds!** Rules for cold start (problems 1-10), then ML takes over."
        }
        
        st.success(f"""
        **{mode.split()[0]} Mode:**
        
        {mode_descriptions[mode.split()[0]]}
        """)
        
        if mode.split()[0] == "Hybrid":
            st.info("🎯 **Hybrid Strategy:**\n- Problems 1-10: Rule-based learning\n- Problems 11-20: ML-powered adaptation")
    
    st.markdown("---")
    
    if st.button("🚀 Start Learning!", type="primary", use_container_width=True):
        if name.strip():
            st.session_state.user_name = name
            st.session_state.current_difficulty = difficulty
            st.session_state.adaptation_mode = mode.split()[0]
            st.session_state.tracker = PerformanceTracker(name, difficulty)
            st.session_state.started = True
            st.session_state.problem_count = 0
            generate_new_problem()
            st.rerun()
        else:
            st.error("Please enter the student's name!")

def generate_new_problem():
    """Generate a new problem at current difficulty"""
    question, answer, operation = st.session_state.generator.generate(
        st.session_state.current_difficulty
    )
    
    st.session_state.current_problem = {
        'question': question,
        'answer': answer,
        'operation': operation
    }
    st.session_state.start_time = time.time()
    st.session_state.feedback_shown = False

def display_problem():
    """Display current problem and handle answer submission"""
    
    # Sidebar with progress and stats
    with st.sidebar:
        st.header("📊 Session Info")
        st.metric("Student", st.session_state.user_name)
        st.metric("Mode", st.session_state.adaptation_mode)
        
        # Phase indicator for Hybrid mode
        if st.session_state.adaptation_mode == "Hybrid":
            phase_info = st.session_state.hybrid_engine.get_phase_info(st.session_state.problem_count)
            if phase_info['phase'] == 'cold_start':
                st.info(f"🎯 **Phase:** Cold Start\n\nUsing rule-based logic\n\n({phase_info['problems_until_ml']} more until ML activates)")
            else:
                st.success(f"🤖 **Phase:** ML Active\n\nRandom Forest is adapting!")
        
        st.metric("Problems", 
                 f"{st.session_state.problem_count}/{st.session_state.total_problems}")
        
        progress = st.session_state.problem_count / st.session_state.total_problems
        st.progress(progress)
        
        st.markdown("---")
        st.subheader("📈 Current Stats")
        
        if st.session_state.problem_count > 0:
            stats = st.session_state.tracker.calculate_session_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{stats['accuracy_percentage']:.0f}%")
                st.metric("Problems", stats['total_problems'])
            with col2:
                st.metric("Avg Time", f"{stats['average_time']:.1f}s")
                st.metric("Level", st.session_state.current_difficulty)
            
            # Mini difficulty progression chart
            if len(stats['difficulty_progression']) > 1:
                st.markdown("**Difficulty Trend:**")
                difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
                values = [difficulty_map[d] for d in stats['difficulty_progression']]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(values) + 1)),
                    y=values,
                    mode='lines+markers',
                    line=dict(color='#FF4B4B', width=2),
                    marker=dict(size=6)
                ))
                fig.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    yaxis=dict(tickmode='array', tickvals=[1, 2, 3], 
                              ticktext=['E', 'M', 'H'], range=[0.5, 3.5]),
                    xaxis=dict(showticklabels=False),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Main problem area
    st.title("🎓 Math Adventures")
    
    difficulty_colors = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### Problem {st.session_state.problem_count + 1} of {st.session_state.total_problems}")
    with col2:
        st.markdown(f"**Difficulty:** {difficulty_colors[st.session_state.current_difficulty]} {st.session_state.current_difficulty}")
    
    st.markdown("---")
    
    problem = st.session_state.current_problem
    
    if not st.session_state.feedback_shown:
        # Show problem
        st.markdown(f"## {problem['question']} = ?")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            user_answer = st.number_input(
                "Your Answer:",
                value=None,
                step=1,
                key=f"answer_{st.session_state.problem_count}"
            )
            
            if st.button("Submit Answer", type="primary", use_container_width=True):
                if user_answer is not None:
                    check_answer(user_answer)
                    st.rerun()
                else:
                    st.error("Please enter an answer!")
    
    else:
        # Show feedback
        display_feedback()

def display_feedback():
    """Display feedback and decision details with enhanced features"""
    time_taken = time.time() - st.session_state.start_time
    is_correct = st.session_state.last_correct
    problem = st.session_state.current_problem
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if is_correct:
            st.success("✅ Correct! Excellent work!")
            st.balloons()
        else:
            st.error(f"❌ Not quite. The correct answer is **{problem['answer']}**")
        
        st.info(f"⏱️ Time taken: **{time_taken:.1f} seconds**")
    
    with col2:
        # Show decision details if available
        if st.session_state.last_decision_details:
            details = st.session_state.last_decision_details
            
            st.metric("AI Decision", details['final_difficulty'])
            st.metric("Confidence", f"{details['confidence']*100:.0f}%")
            
            if details['final_difficulty'] != details['current_difficulty']:
                direction = "📈" if details['final_difficulty'] == "Hard" or \
                           (details['final_difficulty'] == "Medium" and details['current_difficulty'] == "Easy") else "📉"
                st.markdown(f"{direction} **Difficulty Adjusted!**")
    
    # Detailed decision breakdown with 9 enhanced features
    if st.session_state.last_decision_details and st.session_state.problem_count >= 3:
        st.markdown("---")
        
        with st.expander("🤖 AI Decision Details (9 Enhanced Features)", expanded=True):
            details = st.session_state.last_decision_details
            
            # Show phase for hybrid mode
            if st.session_state.adaptation_mode == "Hybrid":
                phase = details.get('phase', 'unknown')
                if phase == 'cold_start':
                    st.info("🔍 **Cold Start Phase** - Using rule-based logic")
                else:
                    st.success("🤖 **ML Active Phase** - Random Forest predictions (96.8% accuracy)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Rule-Based**")
                st.write(f"Prediction: {details['components']['rule_based']['prediction']}")
                st.caption(details['components']['rule_based']['reasoning'][:80] + "...")
            
            with col2:
                st.markdown("**ML-Based (Random Forest)**")
                ml_comp = details['components']['ml_based']
                st.write(f"Prediction: {ml_comp['prediction']}")
                st.write(f"Confidence: {ml_comp['confidence']*100:.1f}%")
                st.caption(ml_comp['reasoning'][:80] + "...")
            
            with col3:
                st.markdown("**Final Decision**")
                st.write(f"Method: {details['decision_method'].replace('_', ' ').title()}")
                st.write(f"Result: {details['final_difficulty']}")
                
                # Agreement indicator
                if details['components']['rule_based']['prediction'] == \
                   details['components']['ml_based']['prediction']:
                    st.success("✅ Consensus")
                else:
                    st.warning("⚠️ Disagreement")
            
            # Show ML features if available (for ML-Based or Hybrid in ML phase)
            if st.session_state.adaptation_mode in ["ML-Based", "Hybrid"] and \
               st.session_state.problem_count >= 10:  # CHANGED: 5 → 10
                st.markdown("---")
                st.markdown("**🔍 Enhanced Features Used:**")
                
                # Get feature importance from ML engine
                top_features = st.session_state.ml_engine.get_feature_importance_for_decision()
                
                if top_features:
                    feature_cols = st.columns(3)
                    for i, (fname, data) in enumerate(list(top_features.items())[:3]):
                        with feature_cols[i]:
                            st.metric(
                                fname.replace('_', ' ').title(),
                                f"{data['value']:.2f}",
                                delta=f"Importance: {data['importance']*100:.1f}%"
                            )
    
    st.markdown("---")
    
    if st.session_state.problem_count < st.session_state.total_problems:
        if st.button("Next Problem →", type="primary", use_container_width=True):
            generate_new_problem()
            st.rerun()
    else:
        st.success("🎉 Session Complete!")
        if st.button("View Results", type="primary", use_container_width=True):
            st.session_state.complete = True
            st.rerun()

def check_answer(user_answer):
    """Check answer and apply adaptive logic"""
    time_taken = time.time() - st.session_state.start_time
    problem = st.session_state.current_problem
    
    st.session_state.problem_count += 1
    
    # Log attempt
    st.session_state.tracker.log_attempt(
        st.session_state.problem_count,
        st.session_state.current_difficulty,
        problem['question'],
        problem['answer'],
        user_answer,
        time_taken,
        problem['operation']
    )
    
    st.session_state.last_correct = (user_answer == problem['answer'])
    st.session_state.feedback_shown = True
    
    # Apply adaptive logic based on mode (minimum 3 problems for features)
    if st.session_state.problem_count >= 3:
        mode = st.session_state.adaptation_mode
        
        if mode == "Rule-Based":
            next_diff, reasoning = st.session_state.rule_engine.decide_next_difficulty(
                st.session_state.tracker,
                st.session_state.current_difficulty,
                st.session_state.problem_count
            )
            st.session_state.last_decision_details = {
                'current_difficulty': st.session_state.current_difficulty,
                'final_difficulty': next_diff,
                'confidence': 0.7,
                'decision_method': 'rule_based',
                'components': {
                    'rule_based': {
                        'prediction': next_diff,
                        'reasoning': reasoning
                    },
                    'ml_based': {
                        'prediction': 'N/A',
                        'confidence': 0.0,
                        'reasoning': 'Not used in Rule-Based mode'
                    }
                }
            }
            
        elif mode == "ML-Based":
            next_diff, reasoning, confidence = st.session_state.ml_engine.decide_next_difficulty(
                st.session_state.tracker,
                st.session_state.current_difficulty,
                st.session_state.problem_count
            )
            st.session_state.last_decision_details = {
                'current_difficulty': st.session_state.current_difficulty,
                'final_difficulty': next_diff,
                'confidence': confidence,
                'decision_method': 'ml_based',
                'components': {
                    'rule_based': {
                        'prediction': 'N/A',
                        'reasoning': 'Not used in ML-Based mode'
                    },
                    'ml_based': {
                        'prediction': next_diff,
                        'confidence': confidence,
                        'reasoning': reasoning
                    }
                }
            }
            
        else:  # Hybrid - uses 10-problem threshold
            next_diff, reasoning, confidence, details = st.session_state.hybrid_engine.decide_next_difficulty(
                st.session_state.tracker,
                st.session_state.current_difficulty,
                st.session_state.problem_count
            )
            st.session_state.last_decision_details = details
        
        st.session_state.current_difficulty = next_diff

def show_summary():
    """Display comprehensive session summary"""
    st.title("📊 Session Summary")
    st.subheader(f"Excellent work, {st.session_state.user_name}! 🎉")
    
    stats = st.session_state.tracker.calculate_session_stats()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Problems", stats['total_problems'])
    with col2:
        st.metric("Accuracy", f"{stats['accuracy_percentage']:.1f}%",
                 delta=f"{stats['correct']} correct")
    with col3:
        st.metric("Avg Time", f"{stats['average_time']:.1f}s")
    with col4:
        st.metric("Final Level", stats['final_difficulty'], 
                 f"{stats['starting_difficulty']} → {stats['final_difficulty']}")
    
    st.markdown("---")
    
    # Hybrid-specific statistics
    if st.session_state.adaptation_mode == "Hybrid":
        st.subheader("🤖 Hybrid Engine Performance")
        hybrid_stats = st.session_state.hybrid_engine.get_agreement_statistics()
        
        if hybrid_stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cold Start", f"{hybrid_stats['cold_start_decisions']} problems")
            with col2:
                st.metric("ML Active", f"{hybrid_stats['ml_active_decisions']} problems")
            with col3:
                if hybrid_stats['ml_active_decisions'] > 0:
                    st.metric("Agreement Rate", f"{hybrid_stats['agreement_rate']*100:.0f}%")
            with col4:
                st.metric("ML High Confidence", hybrid_stats['ml_high_confidence'])
        
        st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "🤖 AI Decisions", "📝 Problem Log"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Difficulty Progression")
            difficulty_map = {"Easy": 1, "Medium": 2, "Hard": 3}
            difficulty_values = [difficulty_map[d] for d in stats['difficulty_progression']]
            
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=list(range(1, len(difficulty_values) + 1)),
                y=difficulty_values,
                mode='lines+markers',
                line=dict(color='#FF4B4B', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(255, 75, 75, 0.2)'
            ))
            fig1.update_layout(
                xaxis_title="Problem Number",
                yaxis_title="Difficulty Level",
                yaxis=dict(tickmode='array', tickvals=[1, 2, 3], 
                          ticktext=['Easy', 'Medium', 'Hard']),
                height=350
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Problems per Difficulty")
            
            fig2 = go.Figure(data=[
                go.Bar(
                    x=list(stats['problems_per_difficulty'].keys()),
                    y=list(stats['problems_per_difficulty'].values()),
                    marker_color=['#90EE90', '#FFD700', '#FF6B6B'],
                    text=list(stats['problems_per_difficulty'].values()),
                    textposition='auto'
                )
            ])
            fig2.update_layout(
                xaxis_title="Difficulty Level",
                yaxis_title="Number of Problems",
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("AI Decision Analysis")
        
        # Get decision history
        if st.session_state.adaptation_mode == "Hybrid":
            history = st.session_state.hybrid_engine.get_decision_history()
        elif st.session_state.adaptation_mode == "ML-Based":
            history = st.session_state.ml_engine.get_decision_history()
        else:
            history = st.session_state.rule_engine.get_decision_history()
        
        if history:
            st.markdown("### Decision History")
            df_history = pd.DataFrame([
                {
                    'Problem': h['problem_count'],
                    'Current': h['current_difficulty'],
                    'Decision': h.get('final_difficulty', h.get('next_difficulty')),
                    'Confidence': f"{h.get('confidence', 0)*100:.0f}%",
                    'Method': h.get('decision_method', 'rule_based').replace('_', ' ').title(),
                    'Reasoning': h['reasoning'][:60] + "..." if len(h['reasoning']) > 60 else h['reasoning']
                }
                for h in history
            ])
            st.dataframe(df_history, use_container_width=True)
    
    with tab3:
        st.subheader("Detailed Problem Log")
        
        problem_data = []
        for attempt in st.session_state.tracker.session_data:
            problem_data.append({
                "Problem": attempt['problem_number'],
                "Difficulty": attempt['difficulty'],
                "Question": attempt['question'],
                "Your Answer": attempt['user_answer'],
                "Correct": attempt['correct_answer'],
                "Result": "✅" if attempt['is_correct'] else "❌",
                "Time (s)": f"{attempt['time_taken']:.1f}"
            })
        
        df_problems = pd.DataFrame(problem_data)
        st.dataframe(df_problems, use_container_width=True)
    
    # Learning trend
    st.markdown("---")
    trend = st.session_state.tracker.get_performance_trend()
    if trend == "improving":
        st.success("📈 Excellent! The student showed clear improvement throughout the session!")
    elif trend == "declining":
        st.info("💪 Keep practicing! Everyone has challenging sessions.")
    else:
        st.info("➡️ Consistent performance maintained throughout the session.")
    
    # Actions
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Export Session Data", use_container_width=True):
            filename = st.session_state.tracker.export_session()
            st.success(f"✅ Data exported to {filename}")
    
    with col2:
        if st.button("🔄 Start New Session", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def main():
    """Main application flow"""
    initialize_session()
    
    if not st.session_state.started:
        welcome_screen()
    elif not st.session_state.complete:
        display_problem()
    else:
        show_summary()

if __name__ == "__main__":
    main()