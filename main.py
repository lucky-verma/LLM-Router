# ------------------------------------- Imports ---------------------------------#
import asyncio
import torch
import uvicorn

from transformers import pipeline
from pydantic import BaseModel
from unsloth import FastLanguageModel
from fastapi import FastAPI, HTTPException

# ------------------------------------- Setup -----------------------------------#
app = FastAPI()

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the labels
candidate_labels = ["sleep", "car"]

# Lazy loading for models
sleep_model = None
sleep_tokenizer = None
car_model = None
car_tokenizer = None

MAX_NEW_TOKENS = 64  # Maximum number of new tokens to generate
PROMPT_TEMPLATE = """You are an AI assistant specialized in answering questions about sleep and cars. 
Please don't make up your own answers, just say you don't know. Stritcly fit your answers in {words} words.
Please provide a concise and accurate answer to the following question:

Question: {user_input}

Answer:"""

# Load the zero-shot classification model (Router model)
classifier = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli", device=device
)


# ------------------------------------ Models -----------------------------------#
class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    classification: str
    response: str


# ----------------------------------- Helpers --------------------------------#
def generate_prompt(user_input):
    """Generate a prompt based on the given user input."""
    return PROMPT_TEMPLATE.format(user_input=user_input, words=MAX_NEW_TOKENS)


def load_classifier():
    """
    Loads the classifier model if it is not already loaded and returns it.
    """
    global classifier
    if classifier is None:
        classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli", device=device
        )
    return classifier


def load_sleep_model():
    """
    Load the sleep model and tokenizer if they are not already loaded.

    Returns:
        Tuple[FastLanguageModel, PreTrainedTokenizer]: The loaded sleep model and tokenizer.
    """
    global sleep_model, sleep_tokenizer
    if sleep_model is None:
        sleep_model, sleep_tokenizer = FastLanguageModel.from_pretrained(
            model_name="thinkersloop/sleep-llm-model",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
        )
        FastLanguageModel.for_inference(sleep_model)
    return sleep_model, sleep_tokenizer


def load_car_model():
    """
    Load the car model and tokenizer if they are not already loaded.

    Returns:
        Tuple[FastLanguageModel, PreTrainedTokenizer]: The loaded car model and tokenizer.
    """
    global car_model, car_tokenizer
    if car_model is None:
        car_model, car_tokenizer = FastLanguageModel.from_pretrained(
            model_name="thinkersloop/car-llm-model",
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
        )
        FastLanguageModel.for_inference(car_model)
    return car_model, car_tokenizer


async def classify_query(text: str) -> str:
    """
    Asynchronously classifies a query using a zero-shot classification model.
    """
    result = await asyncio.to_thread(classifier, text, candidate_labels)
    return result["labels"][0]  # Return the most likely topic


async def generate_response(model, tokenizer, prompt):
    """
    Asynchronously generates a response using the given model and tokenizer for the given prompt.

    Args:
        model (torch.nn.Module): The model used for generating the response.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for tokenizing the prompt.
        prompt (str): The prompt used for generating the response.

    Returns:
        str: The generated response. Returns None if an error occurs during the response generation.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = await asyncio.to_thread(
            model.generate,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        ).replace("\n", "")
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


# ------------------------------------ Routes ----------------------------------#
@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Handle the incoming query request by generating a full prompt, classifying the query,
    loading the appropriate model based on the classification, generating a response
    using the model and tokenizer, and returning a QueryResponse object containing the
    query, classification, and response.

    Parameters:
    - request: QueryRequest object containing the query information

    Returns:
    - QueryResponse object containing the query, classification, and response
    """
    query = request.query
    full_prompt = generate_prompt(query)
    classification = await classify_query(query)

    if classification == "sleep":
        model, tokenizer = load_sleep_model()
    elif classification == "car":
        model, tokenizer = load_car_model()
    else:
        raise HTTPException(status_code=400, detail="Unknown classification")

    response = await generate_response(model, tokenizer, full_prompt)
    return QueryResponse(query=query, classification=classification, response=response)


# ----------------------------------- Server -----------------------------------#
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
