import os
import torch
from transformers import pipeline

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Load the zero-shot classification model
classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=device
)

# Define the labels
candidate_labels = ["sleep", "car"]


# Function to classify the topic of the query
def classify_query(text: str) -> str:
    result = classifier(text, candidate_labels)
    return result  # Return the most likely topic


if __name__ == "__main__":
    # Test the classifier
    queries = [
        "What's the impact of sleep deprivation on cognitive function?",
        "Can you explain the history of the internal combustion engine?",
        "How does REM sleep affect memory consolidation?",
        "What are the main components of an electric vehicle's drivetrain?",
    ]

    for query in queries:
        result = classify_query(query)
        print("-" * 50)
        print(f"Query: {query}")
        print(f"Classified as: {result['labels'][0]}, score: {result['scores'][0]}\n")
