import streamlit as st
import json
import pandas as pd
from collections import defaultdict
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="LLaMAT Downstream Evaluation Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .feedback-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
    .model-output {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_evaluation_data(file_path):
    """Load evaluation data from JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_available_tasks(data):
    """Extract available tasks from the data"""
    if not data:
        return []
    
    tasks = set()
    for model_name, model_data in data.items():
        if isinstance(model_data, dict):
            tasks.update(model_data.keys())
    return sorted(list(tasks))

def get_available_datasets(data, task):
    """Extract available datasets for a given task"""
    if not data:
        return []
    
    datasets = set()
    for model_name, model_data in data.items():
        if isinstance(model_data, dict) and task in model_data:
            datasets.update(model_data[task].keys())
    return sorted(list(datasets))

def get_available_models(data):
    """Extract available model names"""
    if not data:
        return []
    return sorted(list(data.keys()))

def display_model_output(entry, model_name, task, dataset):
    """Display individual model output entry"""
    with st.expander(f"Sample {entry['sample_idx']} - {model_name}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System Prompt:**")
            st.text_area("", value=entry.get('system_prompt', 'N/A'), height=100, key=f"system_{entry['sample_idx']}_{model_name}", disabled=True)
            
            st.markdown("**Question:**")
            st.text_area("", value=entry.get('question', 'N/A'), height=150, key=f"question_{entry['sample_idx']}_{model_name}", disabled=True)
        
        with col2:
            st.markdown("**Gold Answer:**")
            st.text_area("", value=entry.get('gold_answer', 'N/A'), height=100, key=f"gold_{entry['sample_idx']}_{model_name}", disabled=True)
            
            st.markdown("**Model Prediction:**")
            st.text_area("", value=entry.get('prediction', 'N/A'), height=100, key=f"pred_{entry['sample_idx']}_{model_name}", disabled=True)
            
            # Check if prediction is correct
            is_correct = entry.get('processed_gold') == entry.get('processed_prediction')
            if is_correct:
                st.success("✅ Correct")
            else:
                st.error("❌ Incorrect")
        
        # Feedback section
        st.markdown("---")
        st.markdown("**Your Feedback:**")
        feedback_key = f"feedback_{entry['sample_idx']}_{model_name}_{task}_{dataset}"
        feedback = st.text_area(
            "Enter your feedback about this model output:",
            height=100,
            key=feedback_key,
            placeholder="Enter your observations, corrections, or comments about this model's performance..."
        )
        
        return feedback

def calculate_task_statistics(data, task, dataset):
    """Calculate statistics for a specific task and dataset"""
    if not data:
        return {}
    
    stats = {}
    for model_name, model_data in data.items():
        if isinstance(model_data, dict) and task in model_data and dataset in model_data[task]:
            entries = model_data[task][dataset]
            total = len(entries)
            correct = sum(1 for entry in entries if entry.get('processed_gold') == entry.get('processed_prediction'))
            accuracy = correct / total if total > 0 else 0
            
            stats[model_name] = {
                'total': total,
                'correct': correct,
                'accuracy': accuracy
            }
    
    return stats

def main():
    # Header
    st.markdown('<h1 class="main-header">🧪 LLaMAT Downstream Evaluation Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload and configuration
    st.sidebar.header("📁 Data Configuration")
    
    # File upload section
    uploaded_file = st.sidebar.file_uploader(
        "Upload Updated Evaluation File",
        type=['json'],
        help="Upload an updated version of _downstream_eval.json with user feedback"
    )
    
    # Default file path
    default_file_path = "./visualizations/downstream_compare_outputs/_downstream_eval.json"
    
    # Load data
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        data = load_evaluation_data(temp_path)
        # Clean up temp file
        os.remove(temp_path)
    else:
        data = load_evaluation_data(default_file_path)
    
    if data is None:
        st.error("Failed to load evaluation data. Please check the file path or upload a valid file.")
        return
    
    # Display data info
    st.sidebar.success(f"✅ Loaded data for {len(data)} models")
    
    # Get available options
    available_tasks = get_available_tasks(data)
    available_models = get_available_models(data)
    
    # Task selection
    st.sidebar.header("🎯 Task Selection")
    selected_task = st.sidebar.selectbox(
        "Select Task Type:",
        available_tasks,
        help="Choose the downstream task to analyze"
    )
    
    # Dataset selection
    available_datasets = get_available_datasets(data, selected_task)
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset:",
        available_datasets,
        help="Choose the specific dataset for the selected task"
    )
    
    # Model selection
    st.sidebar.header("🤖 Model Selection")
    selected_models = st.sidebar.multiselect(
        "Select Models to Compare:",
        available_models,
        default=available_models[:3],  # Default to first 3 models
        help="Select one or more models to visualize and compare"
    )
    
    if not selected_models:
        st.warning("Please select at least one model to visualize.")
        return
    
    # Main content area
    st.header(f"📊 Analysis: {selected_task} - {selected_dataset}")
    
    # Calculate and display statistics
    stats = calculate_task_statistics(data, selected_task, selected_dataset)
    
    # Statistics overview
    st.subheader("📈 Performance Overview")
    cols = st.columns(len(selected_models))
    
    for i, model_name in enumerate(selected_models):
        if model_name in stats:
            with cols[i]:
                st.metric(
                    label=f"{model_name}",
                    value=f"{stats[model_name]['accuracy']:.2%}",
                    delta=f"{stats[model_name]['correct']}/{stats[model_name]['total']}"
                )
    
    # Model comparison section
    st.subheader("🔍 Model Output Comparison")
    
    # Initialize feedback storage
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = {}
    
    # Get all sample indices for the selected task and dataset
    sample_indices = set()
    for model_name in selected_models:
        if model_name in data and selected_task in data[model_name] and selected_dataset in data[model_name][selected_task]:
            for entry in data[model_name][selected_task][selected_dataset]:
                sample_indices.add(entry['sample_idx'])
    
    sample_indices = sorted(list(sample_indices))
    
    # Sample selection
    selected_sample = st.selectbox(
        "Select Sample to Analyze:",
        sample_indices,
        help="Choose a specific sample to compare across selected models"
    )
    
    # Display outputs for selected sample
    st.markdown(f"### Sample {selected_sample} Analysis")
    
    for model_name in selected_models:
        if model_name in data and selected_task in data[model_name] and selected_dataset in data[model_name][selected_task]:
            # Find the entry for this sample
            entry = None
            for e in data[model_name][selected_task][selected_dataset]:
                if e['sample_idx'] == selected_sample:
                    entry = e
                    break
            
            if entry:
                st.markdown(f"#### {model_name}")
                feedback = display_model_output(entry, model_name, selected_task, selected_dataset)
                
                # Store feedback
                feedback_key = f"{model_name}_{selected_task}_{selected_dataset}_{selected_sample}"
                st.session_state.feedback_data[feedback_key] = {
                    'model': model_name,
                    'task': selected_task,
                    'dataset': selected_dataset,
                    'sample_idx': selected_sample,
                    'feedback': feedback,
                    'timestamp': datetime.now().isoformat()
                }
    
    # Feedback management section
    st.markdown("---")
    st.subheader("💾 Feedback Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        feedback_filename = st.text_input(
            "Enter filename for saving feedback:",
            value=f"feedback_{selected_task}_{selected_dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            help="Specify the filename for saving collected feedback"
        )
    
    with col2:
        if st.button("💾 Save Feedback", type="primary"):
            if feedback_filename:
                # Filter feedback for current session
                current_feedback = {}
                for key, value in st.session_state.feedback_data.items():
                    if value['task'] == selected_task and value['dataset'] == selected_dataset:
                        current_feedback[key] = value
                
                if current_feedback:
                    # Save feedback
                    feedback_path = f"./visualizations/downstream_compare_outputs/{feedback_filename}"
                    with open(feedback_path, 'w') as f:
                        json.dump(current_feedback, f, indent=2)
                    
                    st.success(f"✅ Feedback saved to {feedback_path}")
                    st.info(f"Saved {len(current_feedback)} feedback entries")
                else:
                    st.warning("No feedback to save for the current selection.")
            else:
                st.error("Please enter a filename.")
    
    # Display current feedback summary
    if st.session_state.feedback_data:
        st.subheader("📝 Current Feedback Summary")
        feedback_df = pd.DataFrame([
            {
                'Model': value['model'],
                'Task': value['task'],
                'Dataset': value['dataset'],
                'Sample': value['sample_idx'],
                'Feedback Length': len(value['feedback']) if value['feedback'] else 0,
                'Timestamp': value['timestamp']
            }
            for value in st.session_state.feedback_data.values()
        ])
        
        st.dataframe(feedback_df, use_container_width=True)
        
        # Clear feedback button
        if st.button("🗑️ Clear All Feedback"):
            st.session_state.feedback_data = {}
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            LLaMAT Downstream Evaluation Dashboard | Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
