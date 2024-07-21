import os
import json
import csv
import random
import spacy
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# Constants
SLEEP_DATA_FILE = "data/raw/training_qna_sleep.json"
CAR_DATA_FILE = "data/raw/training_qna_car.json"
OUTPUT_FOLDER = "data/processed"
TRAIN_SLEEP_FILE = "train_sleep.csv"
TRAIN_CAR_FILE = "train_car.csv"
TEST_SLEEP_FILE = "test_sleep.csv"
TEST_CAR_FILE = "test_car.csv"
TRAIN_RATIO = 0.9

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """
    Clean the input text by joining non-space tokens after tokenizing the text.
    """
    return " ".join(token.text for token in nlp(text) if not token.is_space)


def process_json_file(json_file, output_folder, train_file, test_file, train_ratio):
    """
    Process a JSON file and split it into training and testing data.

    Args:
        json_file (str): The path to the JSON file to be processed.
        output_folder (str): The folder where the processed data will be saved.
        train_file (str): The name of the file to save the training data.
        test_file (str): The name of the file to save the testing data.
        train_ratio (float): The ratio of the data to be used for training.

    Returns:
        None
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(json_file, "r") as f:
        data = json.load(f)

    random.shuffle(data["qna"])

    split_index = int(len(data["qna"]) * train_ratio)
    train_data = data["qna"][:split_index]
    test_data = data["qna"][split_index:]

    def format_data(data):
        formatted = []
        for item in data:
            question = clean_text(item["question"])
            answer = clean_text(item["answer"])
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            formatted.append({"messages": messages})
        return formatted

    def save_to_csv(data, csv_file):
        with open(
            os.path.join(output_folder, csv_file),
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
            writer.writerow(["messages"])
            for row in data:
                writer.writerow([json.dumps(row, ensure_ascii=False)])

    save_to_csv(format_data(train_data), train_file)
    save_to_csv(format_data(test_data), test_file)

    print(f"Training data saved to {os.path.join(output_folder, train_file)}.")
    print(f"Testing data saved to {os.path.join(output_folder, test_file)}.")


def analyze_data():
    """
    Analyzes the data by reading the training data from CSV files, 
    extracting the text from the messages, and calculating the token length.
    Then, it generates histograms, probability plots, and box plots of the token lengths. 
    Finally, it calculates the mean, median, and standard deviation of the token lengths 
    and saves them in a summary CSV file.

    Returns:
        None
    """
    train_sleep = pd.read_csv(os.path.join(OUTPUT_FOLDER, TRAIN_SLEEP_FILE))
    train_car = pd.read_csv(os.path.join(OUTPUT_FOLDER, TRAIN_CAR_FILE))
    all_data = pd.concat([train_sleep, train_car])

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def extract_text(row):
        try:
            messages = json.loads(row)["messages"]
            return " ".join([clean_text(msg["content"]) for msg in messages])
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Problematic JSON: {row}")
            return ""

    all_data["text"] = all_data["messages"].apply(extract_text)
    all_data["token_length"] = all_data["text"].apply(
        lambda x: len(tokenizer.encode(x))
    )
    all_data = all_data[all_data["text"] != ""]

    plt.figure(figsize=(12, 6))
    sns.histplot(all_data["token_length"], kde=True)
    plt.title("Distribution of Token Lengths")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "token_length_distribution.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    stats.probplot(all_data["token_length"], dist="norm", plot=plt)
    plt.title("Normal Probability Plot of Token Lengths")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "token_length_qq_plot.png"))
    plt.close()

    mean_length = np.mean(all_data["token_length"])
    median_length = np.median(all_data["token_length"])
    std_dev = np.std(all_data["token_length"])

    print(f"Mean token length: {mean_length:.2f}")
    print(f"Median token length: {median_length:.2f}")
    print(f"Standard deviation of token length: {std_dev:.2f}")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x=all_data["token_length"])
    plt.title("Box Plot of Token Lengths")
    plt.xlabel("Token Length")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "token_length_boxplot.png"))
    plt.close()

    summary = pd.DataFrame(
        {
            "Statistic": ["Mean", "Median", "Standard Deviation"],
            "Value": [mean_length, median_length, std_dev],
        }
    )
    summary.to_csv(os.path.join(OUTPUT_FOLDER, "token_length_summary.csv"), index=False)

    print("Plots and summary have been saved.")


def upload_to_huggingface():
    """
    A function that uploads datasets to the Hugging Face Hub after loading data from files.
    """
    if not hf_token:
        raise ValueError(
            "Please set the HF_TOKEN environment variable in the .env file"
        )
    HfFolder.save_token(hf_token)

    def load_dataset(train_file, test_file):
        train_df = pd.read_csv(os.path.join(OUTPUT_FOLDER, train_file))
        test_df = pd.read_csv(os.path.join(OUTPUT_FOLDER, test_file))
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        return DatasetDict({"train": train_dataset, "test": test_dataset})

    sleep_dataset = load_dataset(TRAIN_SLEEP_FILE, TEST_SLEEP_FILE)
    car_dataset = load_dataset(TRAIN_CAR_FILE, TEST_CAR_FILE)

    sleep_dataset.push_to_hub("thinkersloop/sleep-dataset-llm", private=False)
    print("Sleep dataset successfully pushed to the Hugging Face Hub!")

    car_dataset.push_to_hub("thinkersloop/car-dataset-llm", private=False)
    print("Car dataset successfully pushed to the Hugging Face Hub!")


if __name__ == "__main__":
    process_json_file(
        SLEEP_DATA_FILE, OUTPUT_FOLDER, TRAIN_SLEEP_FILE, TEST_SLEEP_FILE, TRAIN_RATIO
    )
    process_json_file(
        CAR_DATA_FILE, OUTPUT_FOLDER, TRAIN_CAR_FILE, TEST_CAR_FILE, TRAIN_RATIO
    )
    analyze_data()
    upload_to_huggingface()
