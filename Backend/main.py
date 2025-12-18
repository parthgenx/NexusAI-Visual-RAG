from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone
import nest_asyncio
from llama_parse import LlamaParse

# 1. Setup Environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

nest_asyncio.apply()

# 2. Initialize Clients
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# --- SUPER LAZY LOADING ---
model_cache = {}

def get_embedding_model():
    """Loads the library AND model only when specifically asked for."""
    if "model" not in model_cache:
        print("Importing AI Library... (This is the heavy part)")
        from sentence_transformers import SentenceTransformer 
        
        print("Loading Embedding Model...")
        model_cache["model"] = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model Loaded!")
    return model_cache["model"]

app = FastAPI()

# --- CORS: Explicitly Allow Vercel 🔒 ---
origins = [
    "http://localhost:5173",                      # Allow Local Development
    "https://nexus-ai-visual-rag.vercel.app",     # Allow Vercel (EXACT URL)
    "https://nexus-ai-visual-rag.vercel.app/"     # With slash just in case
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,      
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "NexusAI Backend is Active 🟢"}

# --- BACKGROUND TASK: The Heavy Lifter 🏋️‍♂️ ---
def process_upload_background(temp_filename: str):
    print(f"Started background processing for {temp_filename}...")
    try:
        # 1. Clear Memory (Safe Mode)
        try:
            index.delete(delete_all=True)
            print("Memory cleared.")
        except Exception:
            print("Memory was already empty, skipping delete.")

        # 2. Parse (LlamaParse)
        print("Sending to LlamaParse...")
        parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown", 
            num_workers=4,
            verbose=True,
            language="en"
        )
        # Use sync load_data inside background thread
        documents = parser.load_data(temp_filename)
        
        text = ""
        for doc in documents:
            text += doc.text
            
        if not text:
            print("Error: No text extracted.")
            return
       
        # 3. Chunk
        print("Chunking text...")
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
       
        # 4. Embed & Upsert
        print("Embedding chunks...")
        model = get_embedding_model() 
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_embedding = model.encode(chunk).tolist()
            vectors.append({
                "id": f"chunk_{i}",
                "values": vector_embedding,
                "metadata": {"text": chunk}
            })

        # Batch upsert to prevent network timeouts
        print(f"Upserting {len(vectors)} vectors to Pinecone...")
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)

        print(f"✅ Success! Document processed and stored.")

    except Exception as e:
        print(f"❌ Background Task Failed: {e}")
    
    finally:
        # Cleanup file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print("Temp file cleaned up.")

# --- ROUTE 1: Upload (Fast & Async) ⚡ ---
# FIX: Swapped arguments so background_tasks (no default) comes first!
@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Save the file locally first
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Hand off the heavy work to the background
    background_tasks.add_task(process_upload_background, temp_filename)

    # Reply to Frontend IMMEDIATELY
    return {"status": "success", "message": "Processing started in background"}

# --- ROUTE 2: Chat with Memory 🧠 ---
@app.post("/chat")
def generate_chat(request: ChatRequest):
    try:
        user_query = request.prompt.lower()
        general_keywords = ["summarize", "summary", "explain", "what is this", "page"]
        
        is_general_query = any(keyword in user_query for keyword in general_keywords)
        
        # Load Model (Lazy)
        model = get_embedding_model()
        question_embedding = model.encode(request.prompt).tolist()
        
        # Search Vector DB
        if is_general_query:
            search_results = index.query(
                vector=question_embedding,
                top_k=200, 
                include_metadata=True
            )
        else:
            search_results = index.query(
                vector=question_embedding,
                top_k=10, 
                include_metadata=True
            )

        context_text = ""
        for match in search_results['matches']:
            if 'metadata' in match:
                context_text += match['metadata']['text'] + "\n---\n"

        system_prompt = f"""
        You are a helpful AI assistant. Use the Context below to answer the user's question.
        If the answer isn't in the context, say "I couldn't find that information in the document."
        
        Context:
        {context_text}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt}
            ]
        )
        
        return {"response": response.choices[0].message.content}

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))