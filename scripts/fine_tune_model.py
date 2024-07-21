import os
import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import HfApi
from datasets import load_dataset
from dotenv import load_dotenv
from unsloth import FastLanguageModel, is_bfloat16_supported

# Constants
MAX_SEQ_LENGTH = 512
MODEL_ID = "unsloth/mistral-7b-v0.3-bnb-4bit"
MODELS_DIR = "models"
NUM_EPOCHS = 3

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")


def prepare_dataset(dataset_name):
    """
    Loads and prepares a dataset based on the provided dataset name.
    """
    dataset = load_dataset(dataset_name)
    return dataset


def initialize_model():
    """
    Initializes a FastLanguageModel from a pretrained model.

    Returns:
        model (FastLanguageModel): The initialized FastLanguageModel.
        tokenizer (Tokenizer): The tokenizer associated with the model.

    Raises:
        None
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer


def train_model(model, tokenizer, dataset, output_dir):
    """
    Trains a model using the provided parameters.

    Args:
        model (Any): The model to be trained.
        tokenizer (Any): The tokenizer associated with the model.
        dataset (Dict[str, Any]): The dataset containing the training and test data.
        output_dir (str): The directory where the trained model and logs will be saved.

    Returns:
        SFTTrainer: The trained model.
    """
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="messages",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=1,  # more granularity
            gradient_accumulation_steps=1,  # more granularity
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
        ),
    )

    trainer.train()
    return trainer


def push_to_hub(model_dir, repo_name):
    """
    Pushes a model to the Hugging Face Hub.

    Args:
        model_dir (str): The directory containing the model files.
        repo_name (str): The name of the repository on the Hugging Face Hub.
        
    Returns:
        None
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Please set the HF_TOKEN environment variable in the .env file"
        )

    api = HfApi()
    # Create the repository if it doesn't exist
    api.create_repo(
        repo_id=repo_name,
        token=hf_token,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_name,
        token=hf_token,
        repo_type="model",
    )


if __name__ == "__main__":
    # Train sleep model
    print("Training sleep model...")
    sleep_dataset = prepare_dataset("thinkersloop/sleep-dataset-llm")
    sleep_model, sleep_tokenizer = initialize_model()
    sleep_trainer = train_model(
        sleep_model,
        sleep_tokenizer,
        sleep_dataset,
        os.path.join(MODELS_DIR, "sleep_model"),
    )
    sleep_trainer.save_model(os.path.join(MODELS_DIR, "sleep_model_final"))
    sleep_tokenizer.save_pretrained(os.path.join(MODELS_DIR, "sleep_model_final"))
    print("Sleep model training completed.")
    push_to_hub(
        os.path.join(MODELS_DIR, "sleep_model_final"), "thinkersloop/sleep-llm-model"
    )

    # Train car model
    print("Training car model...")
    car_dataset = prepare_dataset("thinkersloop/car-dataset-llm")
    car_model, car_tokenizer = initialize_model()
    car_trainer = train_model(
        car_model, car_tokenizer, car_dataset, os.path.join(MODELS_DIR, "car_model")
    )
    car_trainer.save_model(os.path.join(MODELS_DIR, "car_model_final"))
    car_tokenizer.save_pretrained(os.path.join(MODELS_DIR, "car_model_final"))
    print("Car model training completed.")
    push_to_hub(
        os.path.join(MODELS_DIR, "car_model_final"), "thinkersloop/car-llm-model"
    )

    # Print GPU memory usage
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
