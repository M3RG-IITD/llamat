#!/usr/bin/env python3
"""
Test script for the Streamlit dashboard
This script validates the data structure and provides a quick test of the dashboard functionality
"""

import json
import os
from streamlit_dashboard import (
    load_evaluation_data, 
    get_available_tasks, 
    get_available_datasets, 
    get_available_models,
    calculate_task_statistics
)

def test_dashboard_functionality():
    """Test the core functionality of the dashboard"""
    
    print("🧪 Testing LLaMAT Dashboard Functionality")
    print("=" * 50)
    
    # Test data loading
    data_file = "./visualizations/downstream_compare_outputs/_downstream_eval.json"
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the evaluation data file exists.")
        return False
    
    print(f"✅ Data file found: {data_file}")
    
    # Load data
    data = load_evaluation_data(data_file)
    if data is None:
        print("❌ Failed to load evaluation data")
        return False
    
    print(f"✅ Successfully loaded data for {len(data)} models")
    
    # Test available options extraction
    tasks = get_available_tasks(data)
    print(f"✅ Found {len(tasks)} tasks: {tasks}")
    
    models = get_available_models(data)
    print(f"✅ Found {len(models)} models: {models[:5]}...")  # Show first 5
    
    # Test task-dataset combinations
    if tasks:
        test_task = tasks[0]
        datasets = get_available_datasets(data, test_task)
        print(f"✅ For task '{test_task}': {len(datasets)} datasets: {datasets}")
        
        # Test statistics calculation
        if datasets:
            test_dataset = datasets[0]
            stats = calculate_task_statistics(data, test_task, test_dataset)
            print(f"✅ Statistics for {test_task}-{test_dataset}: {len(stats)} models")
            
            # Show sample statistics
            for model_name, stat in list(stats.items())[:3]:  # Show first 3
                print(f"   {model_name}: {stat['accuracy']:.2%} ({stat['correct']}/{stat['total']})")
    
    print("\n🎉 All tests passed! Dashboard should work correctly.")
    print("\nTo run the dashboard:")
    print("   streamlit run streamlit_dashboard.py")
    
    return True

if __name__ == "__main__":
    test_dashboard_functionality()
