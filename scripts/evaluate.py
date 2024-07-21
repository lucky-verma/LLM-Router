import json
import torch

import numpy as np
import time as python_time

from unsloth import FastLanguageModel
from datasets import load_dataset
from rouge_score import rouge_scorer


def load_model(model_name):
    """
    Function to load and prepare the model for inference.

    Args:
        model_name (str): The name of the model to load and prepare.

    Returns:
        tuple: A tuple containing the loaded model and the corresponding tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=None,  # Auto-detect dtype
        load_in_4bit=True,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    return model, tokenizer


def generate_response(model, tokenizer, prompt):
    """
    Performs inference using the provided model and tokenizer on a given prompt.

    Args:
        model (object): The model to use for generating the response.
        tokenizer (object): The tokenizer used to process the prompt.
        prompt (str): The input prompt for generating the response.

    Returns:
        str: The generated response based on the input prompt.
            Returns None if an error occurs during the response generation.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(
            "\n", ""
        )
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


def evaluate_model(model, tokenizer, dataset):
    """
    Function to evaluate the model.

    Args:
        model: The model for evaluation.
        tokenizer: The tokenizer for processing the input.
        dataset: The dataset containing examples for evaluation.

    Returns:
        A tuple containing the predictions and references.
    """
    predictions = []
    references = []

    for example in dataset:
        messages = json.loads(example["messages"])
        prompt = messages["messages"][0]["content"]
        reference = messages["messages"][1]["content"]
        response = generate_response(model, tokenizer, prompt).replace("\n", "")
        predictions.append(response)
        references.append(reference)

    return predictions, references


def calculate_rouge(predictions, references):
    """
    Calculate ROUGE scores for the given predictions and references.

    Parameters:
    - predictions: List of predicted responses.
    - references: List of reference responses.

    Returns:
    - avg_scores: Dictionary containing average ROUGE scores for rouge1, rouge2, and rougeL.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = []

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score)

    avg_scores = {
        key: sum(score[key].fmeasure for score in scores) / len(scores)
        for key in scores[0].keys()
    }
    return avg_scores


def measure_inference_time(model, input_ids, num_repeats=3):
    """
    Measure the inference time of the model for a given input.

    Parameters:
    - model: The model used for inference.
    - input_ids: The input IDs for the model.
    - num_repeats: The number of times to repeat the inference.

    Returns:
    - timings: An array containing the time taken for each repeat of inference.
    """
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    timings = np.zeros((num_repeats, 1))

    for _ in range(3):
        _ = model.generate(input_ids, max_new_tokens=64)

    with torch.no_grad():
        for rep in range(num_repeats):
            torch.cuda.synchronize()
            starter.record()
            _ = model.generate(input_ids, max_new_tokens=64)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    return timings.flatten()


def calculate_throughput(model, input_ids, batch_size, duration=10):
    """
    Calculates the throughput of the model based on the given input parameters.

    Parameters:
    - model: The model used for throughput calculation.
    - input_ids: The input IDs for the model.
    - batch_size: The batch size used for the calculation.
    - duration: The duration over which the throughput is calculated.

    Returns:
    - throughput: The calculated throughput value.
    """
    start_time = python_time.time()
    count = 0
    while python_time.time() - start_time < duration:
        _ = model.generate(input_ids.repeat(batch_size, 1), max_new_tokens=20)
        count += batch_size

    throughput = count / duration
    return throughput


def measure_gpu_utilization():
    """
    Measures GPU utilization by calculating the ratio of allocated memory to maximum memory if GPU is available. Returns None if GPU is not available.
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
    else:
        return None


if __name__ == "__main__":
    # Load the sleep model
    sleep_model_name = "thinkersloop/sleep-llm-model"
    sleep_model, sleep_tokenizer = load_model(sleep_model_name)

    # Load the car model
    car_model_name = "thinkersloop/car-llm-model"
    car_model, car_tokenizer = load_model(car_model_name)

    # Load test datasets
    sleep_test_dataset = load_dataset("thinkersloop/sleep-dataset-llm", split="test")
    car_test_dataset = load_dataset("thinkersloop/car-dataset-llm", split="test")

    # Evaluate sleep model
    print("Evaluating sleep model...")
    sleep_predictions, sleep_references = evaluate_model(
        sleep_model, sleep_tokenizer, sleep_test_dataset
    )

    # Evaluate car model
    print("Evaluating car model...")
    car_predictions, car_references = evaluate_model(
        car_model, car_tokenizer, car_test_dataset
    )

    # Calculate ROUGE scores
    sleep_rouge_scores = calculate_rouge(sleep_predictions, sleep_references)
    print("Sleep Model ROUGE Scores:", sleep_rouge_scores)

    car_rouge_scores = calculate_rouge(car_predictions, car_references)
    print("Car Model ROUGE Scores:", car_rouge_scores)

    # Example prompts for inference
    sample_inputs = [
        "What are the benefits of sleep for mental health?",
        "What were the key innovations that led to the development of the first gasoline-powered automobiles?",
    ]

    # Tokenize inputs
    encoded_inputs = [
        sleep_tokenizer.encode(text, return_tensors="pt").to("cuda")
        for text in sample_inputs
    ]

    # Measure inference time
    inference_times = []
    for input_ids in encoded_inputs:
        times = measure_inference_time(sleep_model, input_ids)
        inference_times.extend(times)

    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    percentile_95 = np.percentile(inference_times, 95)

    # Measure throughput (QPS)
    batch_size = 4
    qps = calculate_throughput(sleep_model, encoded_inputs[0], batch_size)

    # Measure GPU utilization
    gpu_utilization = measure_gpu_utilization()

    # Print results
    print(f"Average Inference Time: {mean_time:.2f} ms")
    print(f"Standard Deviation of Inference Time: {std_time:.2f} ms")
    print(f"95th Percentile Latency: {percentile_95:.2f} ms")
    print(f"Queries Per Second (QPS): {qps:.2f}")
    print(
        f"GPU Utilization: {gpu_utilization:.2%}"
        if gpu_utilization
        else "GPU Utilization: N/A"
    )

    rpm = qps * 60
    print(f"Requests Per Minute (RPM): {rpm:.2f}")
