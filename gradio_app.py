import gradio as gr
import aiohttp
import asyncio

# FastAPI endpoint URL
API_URL = "http://localhost:8000/query"

async def chat_with_model(message, history):
    """
    Asynchronously chats with a model based on the message input and history.

    Parameters:
    - message: The input message for the model.
    - history: The history of previous messages.

    Returns:
    - A string indicating the model used and the response generated.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json={"query": message}) as response:
            if response.status == 200:
                result = await response.json()
                model_used = (
                    "Sleep Model"
                    if result["classification"] == "sleep"
                    else "Car Model"
                )
                return f"{model_used}: {result['response']}"
            else:
                return f"Error: Unable to get response (Status code: {response.status})"


def gradio_qa(message, history):
    """
    A function that uses the `chat_with_model` coroutine to asynchronously chat with a model based on the message input and history.
    """
    response = asyncio.run(chat_with_model(message, history))
    return response

# Create the gradio interface
iface = gr.ChatInterface(
    gradio_qa,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="Ask a question about sleep or cars", container=False, scale=7
    ),
    title="Multi-Model LLM Router",
    description="This chatbot routes your questions to specialized models for sleep or car-related topics.",
    theme="soft",
    examples=[
        "What are the stages of sleep?",
        "How does engine displacement affect a car's performance?",
        "What were the key innovations that led to the development of the first gasoline-powered automobiles?",
        "Can you explain how electric cars work?",
    ],
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
)

if __name__ == "__main__":
    iface.launch(debug=True, share=False)
