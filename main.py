from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# ... your imports ...

load_dotenv()

# PASTE THESE TWO LINES HERE:
print(f"MONGO_URI Loaded: {os.getenv('MONGO_URI') is not None}")
print(f"GROQ_KEY Loaded: {os.getenv('GROQ_API_KEY') is not None}")

app = FastAPI()
# ... the rest of your code ...
app = FastAPI()

# MongoDB setup
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client["studybot"]
chats = db["chats"]

# LLM setup
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"), # This pulls from your .env file
    model_name="llama-3.3-70b-versatile"  # Remember to update the model name here!
)

SYSTEM_PROMPT = """You are StudyBot, an expert academic tutor.
Explain concepts clearly, use examples, and be encouraging."""

class ChatRequest(BaseModel):
    session_id: str
    user_input: str

@app.get("/")
def home():
    return {"message": "Study Bot API Running 🚀"}

@app.post("/chat")
async def chat(request: ChatRequest):
    # Get previous messages from MongoDB
    history = []
    cursor = chats.find({"session_id": request.session_id}).sort("timestamp", 1).limit(10)
    async for doc in cursor:
        history.append(HumanMessage(content=doc["user_input"]))
        history.append(AIMessage(content=doc["bot_response"]))

    # Build messages with system prompt + history + new message
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history + [HumanMessage(content=request.user_input)]

    # Get LLM response
    response = llm.invoke(messages)

    # Save to MongoDB
    await chats.insert_one({
        "session_id": request.session_id,
        "user_input": request.user_input,
        "bot_response": response.content,
        "timestamp": datetime.utcnow()
    })

    return {"response": response.content}

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history = []
    cursor = chats.find({"session_id": session_id}, {"_id": 0}).sort("timestamp", 1)
    async for doc in cursor:
        history.append(doc)
    return {"session_id": session_id, "history": history}



