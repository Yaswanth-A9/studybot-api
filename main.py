from fastapi import FastAPI
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# Load environment variables
load_dotenv()

print(f"MONGO_URI Loaded: {os.getenv('MONGO_URI') is not None}")
print(f"GROQ_KEY Loaded: {os.getenv('GROQ_API_KEY') is not None}")

app = FastAPI()

# ---------------------------
# MongoDB Setup
# ---------------------------
client = AsyncIOMotorClient(os.getenv("MONGO_URI"))
db = client["studybot"]
chats = db["chats"]

# ---------------------------
# LLM Setup
# ---------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

# ---------------------------
# System Prompt
# ---------------------------
SYSTEM_PROMPT = """
You are StudyBot, an expert academic tutor.
Explain concepts clearly, use examples, and be encouraging.
"""

# ---------------------------
# Request Model
# ---------------------------
class ChatRequest(BaseModel):
    session_id: str
    user_input: str


# ---------------------------
# Home Route
# ---------------------------
@app.get("/")
def home():
    return {"message": "Study Bot API Running 🚀"}


# ---------------------------
# Chat Route
# ---------------------------
@app.post("/chat")
async def chat(request: ChatRequest):

    # 1️⃣ Fetch previous messages from MongoDB
    history = []
    cursor = chats.find(
        {"session_id": request.session_id}
    ).sort("timestamp", 1)

    async for doc in cursor:
        if doc["role"] == "user":
            history.append(HumanMessage(content=doc["message"]))
        elif doc["role"] == "assistant":
            history.append(AIMessage(content=doc["message"]))

    # 2️⃣ Build full conversation
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *history,
        HumanMessage(content=request.user_input)
    ]

    # 3️⃣ Get response from LLM
    response = llm.invoke(messages)

    # 4️⃣ Store user + assistant messages separately
    await chats.insert_many([
        {
            "session_id": request.session_id,
            "role": "user",
            "message": request.user_input,
            "timestamp": datetime.utcnow()
        },
        {
            "session_id": request.session_id,
            "role": "assistant",
            "message": response.content,
            "timestamp": datetime.utcnow()
        }
    ])

    return {"response": response.content}


# ---------------------------
# Get Conversation History
# ---------------------------
@app.get("/history/{session_id}")
async def get_history(session_id: str):

    history = []
    cursor = chats.find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("timestamp", 1)

    async for doc in cursor:
        history.append(doc)

    return {
        "session_id": session_id,
        "history": history
    }


