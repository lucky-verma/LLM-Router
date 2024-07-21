# Multi-Model LLM System with Router

## Overview

This project develops a system that uses two specialized LLMs trained on different topics (sleep science and car history) and implements a router to direct queries to the appropriate model based on the input question. The system utilizes FastAPI for the backend, Gradio for the user interface, and implements asynchronous processing for improved performance.

## Folder Structure

    multi_model_llm_system/
    ├── data/
    │   ├── raw/
    │   │   ├── sleep_science_qa.csv
    │   │   ├── car_history_qa.csv
    │   ├── processed/
    │       ├── train_sleep.csv
    │       ├── train_car.csv
    │       ├── test_sleep.csv
    │       ├── test_car.csv
    ├── scripts/
    │   ├── data_preparation.py
    │   ├── fine_tune_model.py
    │   ├── query_router.py
    │   ├── inference.py
    │   ├── evaluate.py
    ├── notebooks/
    │   ├── data_exploration.ipynb
    │   ├── model_training.ipynb
    │   ├── router_training.ipynb
    │   ├── inference_tests.ipynb
    ├── requirements.txt
    ├── environment.yml
    ├── README.md
    ├── .env # Add your HF_TOKEN here

## Setup Instructions

1. Clone the repository

    ```bash
    git clone https://github.com/lucky-verma/LLM-Router.git
    cd Multi-Model-LLM-Router
    ```

2. Install dependencies
    Make sure you have conda installed
    My CUDA version is 12.2 on Ubuntu 22.04

    ```bash
    conda env create -f environment.yml
    conda activate webai
    ```

3. Prepare data
    Place your raw datasets in the data/raw/ directory.
    Run the data preparation script:

    ```bash
    python -m scripts.data_preparation
    ```

4. Fine-tune models
    Run the fine-tuning script:

    ```bash
    python -m scripts.fine_tune_model
    ```

5. Run inference
    Test the inference pipeline:

    ```bash
    python -m scripts.inference
    ```

6. Run evaluation
    Test the evaluation pipeline:

    ```bash
    python -m scripts.evaluate
    ```

## Usage

1. Start the FastAPI backend:

   ```bash
   python main.py
   ```

2. Start the Gradio user interface:

    ```bash
    python gradio_app.py
    ```

    Open the provided URL in your web browser to interact with the chat interface.

## Design Choices

* Base Model: We selected the Mistral 7B model, implemented via Unsloth, as our foundation. This choice offers an optimal balance between performance and efficiency, providing robust natural language understanding while maintaining reasonable computational requirements.
* Query Router: For query classification, we employed a zero-shot classification model (BART-large-mnli). This approach allows for flexible and accurate routing of queries to the appropriate domain-specific model without requiring extensive labeled training data for each new domain.
* Domain Specialization: We fine-tuned separate models on domain-specific datasets:
  * Sleep Science Model: Trained on a comprehensive dataset of sleep-related research, studies, and expert knowledge.
  * Car History Model: Fine-tuned using a rich dataset encompassing automotive history, technological advancements, and industry developments.
This specialization ensures high-quality, domain-specific responses.
* Backend Framework: We chose FastAPI for our backend due to its:
  * Asynchronous request handling capabilities, enabling efficient processing of multiple queries.
  * Built-in support for API documentation and validation.
  * Ease of integration with machine learning models and other Python libraries.
* Frontend Interface: Gradio was selected to create our user interface because it offers:
  * A simple yet powerful framework for building interactive AI applications.
* Model Optimization: We utilized quantization techniques to reduce model size and inference time, allowing for more efficient deployment and faster response times.
* Scalability Considerations: The architecture is designed to easily accommodate additional domain-specific models, allowing for future expansion of the system's knowledge base.

## Implemented Improvements

* synchronous Processing: Implemented async functions in FastAPI to handle concurrent requests more efficiently.
* Model Caching: Implemented lazy loading and caching of models to reduce startup time and memory usage.
* Gradio Interface: Created a user-friendly chat interface that displays which model (Sleep or Car) is responding to each query.

## Potential Future Improvements

1. Model Optimization: Further optimize the models using techniques like qLoRA and pruning to reduce inference time.
2. Distributed Computing: Implement a distributed system to handle model inference across multiple GPUs or machines.
3. Caching Mechanism: Implement a response cache for frequent queries to reduce unnecessary model inference.
4. Advanced Router: Develop a more sophisticated routing mechanism that can handle multi-topic queries or ambiguous cases. Train a router on sleep and car datasets or create a Synthetic Dataset to train the router.
5. Performance Profiling: Use detailed profiling tools to identify and address specific bottlenecks in the system.
6. Load Balancing: Introduce a load balancer to distribute requests across multiple worker processes or servers.
7. Streaming Responses: Implement streaming responses to improve perceived responsiveness for users.
8. Monitoring and Logging: Add comprehensive logging and monitoring to track system performance and identify issues in real-time.
Known Issues
9. High latency: The current system has a high average response time(~2000 ms), which needs to be addressed for real-time applications.
10. Limited scalability: The system doesn't show significant performance improvements with increased concurrency.
11. Containerization: The system should be containerized to provide scalability and robustness.
