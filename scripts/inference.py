from unsloth import FastLanguageModel

# Function to load and prepare the model for inference
def load_model(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=None,  # Auto-detect dtype
        load_in_4bit=True,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# Function to perform inference
def generate_response(model, tokenizer, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            outputs[0][len(inputs["input_ids"]) :], skip_special_tokens=True
        ).replace("\n", "")
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


if __name__ == "__main__":
    # Load the sleep model
    sleep_model_name = "thinkersloop/sleep-llm-model"
    sleep_model, sleep_tokenizer = load_model(sleep_model_name)

    # Load the car model
    car_model_name = "thinkersloop/car-llm-model"
    car_model, car_tokenizer = load_model(car_model_name)

    # Example prompts for inference
    sleep_prompt = "What are the benefits of sleep for mental health?"
    car_prompt = "What were the key innovations that led to the development of the first gasoline-powered automobiles?"

    # Generate responses
    sleep_response = generate_response(sleep_model, sleep_tokenizer, sleep_prompt)
    car_response = generate_response(car_model, car_tokenizer, car_prompt)

    # Print the responses
    print("Sleep Model Response:")
    print(sleep_response)
    print("-" * 50)
    print("Car Model Response:")
    print(car_response)
