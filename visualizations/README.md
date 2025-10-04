# LLaMAT Downstream Evaluation Dashboard

A Streamlit-based dashboard for visualizing and analyzing downstream evaluation results from 16 different models.

## Features

- **Task Selection**: Choose from available downstream tasks (ner, pc, sf, ee, re, sar, sc, qna, mcq)
- **Model Comparison**: Select and compare outputs from multiple models
- **Interactive Visualization**: View model outputs side-by-side with correctness indicators
- **Feedback Collection**: Add your own feedback for each model output
- **Data Management**: Save feedback and load updated evaluation files

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_streamlit.txt
```

## Usage

1. Run the dashboard:
```bash
streamlit run streamlit_dashboard.py
```

2. The dashboard will open in your browser (typically at http://localhost:8501)

3. Use the sidebar to:
   - Upload updated evaluation files
   - Select task type and dataset
   - Choose models to compare

4. In the main area:
   - View performance statistics
   - Select samples to analyze
   - Add feedback for model outputs
   - Save feedback with custom filenames

## Data Structure

The dashboard expects the `_downstream_eval.json` file with the following structure:
```json
{
  "model_name": {
    "task_name": {
      "dataset_name": [
        {
          "sample_idx": 0,
          "gold_answer": "...",
          "prediction": "...",
          "processed_gold": [...],
          "processed_prediction": [...],
          "system_prompt": "...",
          "question": "..."
        }
      ]
    }
  }
}
```

## Available Models

The dashboard supports all 16 models from the evaluation:
- LLaMA-2, LLaMA-2-chat, LLaMA-3, LLaMA-3-chat
- LLaMat-2, LLaMat-2-chat, LLaMat-3, LLaMat-3-chat
- Claude-3-Opus, Claude-3-Haiku, Claude-3.5-Sonnet
- GPT-4, GPT-4o
- Gemini-1.5-flash, Gemini-1.5-Flash-8b, Gemini-1.5-pro

## Feedback System

- Each model output has a dedicated feedback text area
- Feedback is automatically stored in session state
- Use the "Save Feedback" button to export feedback to JSON files
- Upload updated evaluation files to load previous feedback

## File Locations

- Default data file: `./visualizations/downstream_compare_outputs/_downstream_eval.json`
- Saved feedback: `./visualizations/downstream_compare_outputs/feedback_*.json`
