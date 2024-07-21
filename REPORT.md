# Detailed Report: Multi-Model LLM System with Router

## 1. Executive Summary

This report outlines the development of a Multi-Model Large Language Model (LLM) System with a Router, designed to handle queries in two specialized domains: sleep science and car history. The system employs two fine-tuned LLMs and a routing mechanism to direct queries to the appropriate model, ensuring accurate and domain-specific responses.

## 2. System Architecture

### 2.1 Model Training

Two separate LLMs were fine-tuned using the Mistral 7B model as the base:

- **a. Sleep Science Model**: Specialized in answering queries related to sleep research, disorders, and hygiene.
- **b. Car History Model**: Focused on automotive history, technological advancements, and industry developments.

The models were fine-tuned using domain-specific datasets provided, split into training and validation sets. We employed the Unsloth library for efficient fine-tuning, utilizing quantization techniques to optimize performance and reduce model size.

### 2.2 Router

The routing mechanism uses a zero-shot classification model (BART-large-mnli) to analyze input queries and determine the appropriate specialized model to handle the request. This approach allows for flexible and accurate routing without requiring extensive labeled training data for each domain.

### 2.3 Inference Pipeline

The inference pipeline is built using FastAPI, providing an asynchronous backend capable of handling concurrent requests efficiently. The pipeline includes:

- Query preprocessing
- Topic classification using the router
- Model inference using the appropriate specialized LLM
- Response formatting and delivery

### 2.4 User Interface

A Gradio-based chat interface provides a user-friendly front-end for interacting with the system. The interface displays which model (Sleep or Car) is responding to each query, enhancing transparency and user understanding.

## 3. Implementation Details

### 3.1 Model Training

- **Base Model**: Mistral 7B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) for efficient adaptation
- **Training Data**: Domain-specific Q&A datasets for sleep science and car history
- **Optimization**: 4-bit quantization for reduced model size and faster inference

### 3.2 Router Implementation

- **Model**: BART-large-mnli for zero-shot classification
- **Input**: User query
- **Output**: Classification label ("sleep" or "car")
- **Implementation**: Integrated into the FastAPI backend for seamless query processing

### 3.3 Inference Pipeline

- **Backend Framework**: FastAPI
- **Asynchronous Processing**: Implemented for handling concurrent requests
- **Model Caching**: Lazy loading and caching of models to reduce startup time and memory usage
- **Error Handling**: Robust error handling and logging for system reliability

### 3.4 Performance Optimization

- **Quantization**: 4-bit quantization applied to reduce model size and inference time
- **Asynchronous Processing**: Utilized FastAPI's asynchronous capabilities for improved concurrency
- **Caching**: Implemented model and classification result caching to reduce redundant computations

## 4. Challenges Faced

- **Model Size vs. Performance Trade-off**: Balancing the need for high-quality responses with the computational constraints of larger models.
  - **Solution**: Utilized 4-bit quantization and the Unsloth library to optimize model size and performance.
- **Routing Accuracy**: Ensuring accurate classification of queries, especially for ambiguous or multi-topic questions.
  - **Solution**: Employed a zero-shot classification model, allowing for flexible and accurate routing without extensive labeled data.
- **Latency Issues**: Initial high response times, particularly under concurrent load.
  - **Solution**: Implemented asynchronous processing, model caching, and optimized the inference pipeline to reduce latency.
- **Scalability**: Designing the system to handle increased load and potential addition of new domains.
  - **Solution**: Adopted a modular architecture that allows for easy integration of additional specialized models.

## 5. Performance Metrics

- **Average Inference Time**: 2259.29 ms
- **Standard Deviation of Inference Time**: 10.02 ms
- **95th Percentile Latency**: 2274.62 ms
- **Queries Per Second (QPS)**: 2.80
- **GPU Utilization**: 98.75%
- **Sleep Model ROUGE Scores**:
  - *ROUGE-1*: 0.3012691012691013
  - *ROUGE-2*: 0.1251693017946506
  - *ROUGE-L*: 0.18873348873348875
- **Car Model ROUGE Scores**:
  - *ROUGE-1*: 0.33112472515337554
  - *ROUGE-2*: 0.10982220912707506
  - *ROUGE-L*: 0.21737507194154593

## 6. Potential Improvements

- **Advanced Router**: Develop a more sophisticated routing mechanism capable of handling multi-topic queries or ambiguous cases.
- **Distributed Computing**: Implement a distributed system to handle model inference across multiple GPUs or machines for improved scalability.
- **Response Caching**: Implement a response cache for frequent queries to reduce unnecessary model inference.
- **Streaming Responses**: Implement streaming responses to improve perceived responsiveness for users.
- **Comprehensive Monitoring**: Add detailed logging and monitoring to track system performance and identify issues in real-time.
- **Containerization**: Containerize the system using Docker for improved deployment and scalability.
- **Continuous Learning**: Implement a feedback loop to continuously improve model performance based on user interactions.

## 7. Timelines and Effort Levels

### 7.1 Small Effort (Current Deliverable) - 1-2 weeks

- Basic implementation of two fine-tuned models (Sleep and Car)
- Simple routing mechanism using zero-shot classification
- FastAPI backend with basic error handling
- Gradio frontend for user interaction
- Basic documentation and setup instructions

### 7.2 Medium Effort - 3-4 weeks

- Improved model fine-tuning with hyperparameter optimization
- Enhanced routing mechanism with better handling of edge cases
- Implementation of response caching
- Basic performance metrics and monitoring
- Docker containerization for easier deployment
- Comprehensive documentation and user guide

### 7.3 Large Effort - 1-2 months

- Advanced multi-model system with the ability to easily add new domains
- Sophisticated routing mechanism capable of handling multi-topic queries
- Distributed computing setup for improved scalability
- Streaming responses for better user experience
- Comprehensive monitoring and alerting system
- Continuous learning pipeline for model improvement
- Extensive testing suite including stress tests and security audits
- Detailed API documentation and developer resources

## 8. Conclusion

The Multi-Model LLM System with Router demonstrates the potential of specialized language models combined with intelligent query routing. While the current implementation provides a solid foundation, there is significant room for improvement in areas such as scalability, response time, and routing sophistication. Future iterations of the system could greatly enhance its capabilities and applicability across various domains.
