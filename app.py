from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pydantic import BaseModel
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")

# Initialize FastAPI app
app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat request and response models
class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str
    chat_history: list

# Initialize chat history in memory (for simplicity; consider a database for persistence)
chat_history = []

# Function to preprocess user input (custom prompt)
def preprocess_input(user_input: str):
    # Add custom instructions, context, or formatting
    return f"Respond to the following input in a friendly, helpful tone you are an ai assistant who can generate reciepe from left over food: {user_input}"

# Function to postprocess output (custom formatting)
def postprocess_output(response: str):
    # Modify the model's response if needed (e.g., formatting, tone adjustments)
    return f"{response}"

# Custom memory class to add additional context or handling
class CustomMemory(ConversationBufferMemory):
    def save_context(self, inputs: dict, outputs: dict):
        # You could add your custom rules or context-saving logic here
        super().save_context(inputs, outputs)

# Create a custom conversation chain class
class CustomConversationChain(ConversationChain):
    def run(self, input_text: str):
        custom_prompt = f"Given the following chat history, respond appropriately to the user input:\n{input_text}\n"
        return super().run(custom_prompt)

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint to handle user input and return chatbot responses.
    """
    # Preprocess user input
    user_input = preprocess_input(request.user_input)
    
    # Initialize ChatGroq with additional parameters like temperature
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.7, max_tokens=150)
    
    # Initialize custom memory and conversation chain
    memory = CustomMemory()
    conversation = CustomConversationChain(llm=groq_chat, memory=memory)
    
    # Save existing chat history in memory
    for message in chat_history:
        memory.save_context({'input': message['human']}, {'output': message['AI']})

    # Generate response
    response = conversation.run(user_input)
    
    # Postprocess the response
    processed_response = postprocess_output(response)

    # Save conversation to chat history
    message = {'human': request.user_input, 'AI': processed_response}
    chat_history.append(message)

    return ChatResponse(response=processed_response, chat_history=chat_history)

@app.get("/history")
def get_chat_history():
    """
    Endpoint to retrieve the chat history.
    """
    return JSONResponse(content={"chat_history": chat_history})

@app.delete("/history")
def clear_chat_history():
    """
    Endpoint to clear the chat history.
    """
    chat_history.clear()
    return JSONResponse(content={"message": "Chat history cleared."})

@app.get("/", response_class=HTMLResponse)
def get_frontend(request: Request):
    """
    Serve the HTML frontend using Jinja2Templates.
    """
    return templates.TemplateResponse("index.html", {"request": request})
