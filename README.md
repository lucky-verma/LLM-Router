# Multi-Model LLM System with Router

## Overview

This project develops a system that uses two specialized LLMs trained on different topics and implements a router to direct queries to the appropriate model based on the input question.

## Folder Structure

    multi_model_llm_system/
    ├── data/
    │   ├── raw/
    │   │   ├── sleep_science_qa.csv
    │   │   ├── car_history_qa.csv
    │   ├── processed/
    │       ├── sleep_data_cleaned.csv
    │       ├── car_data_cleaned.csv
    ├── models/
    │   ├── base_model/
    │   ├── model_sleep/
    │   ├── model_car/
    ├── scripts/
    │   ├── data_preparation.py
    │   ├── fine_tune_model.py
    │   ├── train_router.py
    │   ├── inference.py
    │   ├── evaluate.py
    ├── notebooks/
    │   ├── data_exploration.ipynb
    │   ├── model_training.ipynb
    │   ├── router_training.ipynb
    │   ├── inference_tests.ipynb
    ├── config/
    │   ├── config.yaml
    ├── logs/
    │   ├── training_logs/
    │   ├── inference_logs/
    ├── requirements.txt
    ├── README.md
    ├── .env


## Setup Instructions

1. Clone the repository

    ```bash
    git clone https://github.com/yourusername/multi_model_llm_system.git
    cd multi_model_llm_system
    ```

2. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

3. Prepare data
    Place your raw datasets in the data/raw/ directory.
    Run the data preparation script:

    ```bash
    python scripts/data_preparation.py
    ```

4. Fine-tune models
    Run the fine-tuning script:

    ```bash
    python scripts/fine_tune_model.py
    ```

5. Train the router
    Run the router training script:

    ```bash
    python scripts/train_router.py
    ```

6. Run inference
    Test the inference pipeline:

    ```bash
    python scripts/inference.py
    ```

### Design Choices

Base Model: GPT-3 was chosen for its versatility and performance.
Router: A logistic regression classifier was used for simplicity and efficiency.
Fine-Tuning: Models were fine-tuned on domain-specific datasets to specialize them.

### Potential Improvements

Use more sophisticated models for the router.
Implement real-time inference capabilities.
Add more comprehensive performance evaluation metrics.
