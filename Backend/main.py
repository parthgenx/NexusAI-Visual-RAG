from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone
# REMOVED: from sentence_transformers import SentenceTransformer  <-- REMOVE THIS
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
        # WE IMPORT IT HERE NOW. 
        # The server starts instantly because this line hasn't run yet!
        from sentence_transformers import SentenceTransformer 
        
        print("Loading Embedding Model...")
        model_cache["model"] = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model Loaded!")
    return model_cache["model"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Pinecone RAG Backend (Super-Lazy Edition) is Active 🌲🦙"}

# --- ROUTE 1: Upload & Memorize 📂 ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # Wipe Memory on new upload
        index.delete(delete_all=True) 

        # Step A: Save File
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Step B: Parse (LlamaParse)
        parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown", 
            num_workers=4,
            verbose=True,
            language="en"
        )
        documents = await parser.aload_data(temp_filename)
        
        text = ""
        for doc in documents:
            text += doc.text
            
        if not text:
            return {"error": "LlamaParse could not extract text."}
       
        # Step C: Chunk
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
       
        # Step D: Embed (Using Super-Lazy Loader)
        model = get_embedding_model() 
        
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_embedding = model.encode(chunk).tolist()
            vectors.append({
                "id": f"{file.filename}_{i}",
                "values": vector_embedding,
                "metadata": {"text": chunk}
            })

        index.upsert(vectors=vectors)

        return {"status": "success", "chunks_stored": len(chunks)}

    except Exception as e:
        print(f"Upload Error: {e}")
        return {"error": str(e)}
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- ROUTE 2: Chat with Memory 🧠 ---
@app.post("/chat")
def generate_chat(request: ChatRequest):
    try:
        user_query = request.prompt.lower()
        general_keywords = ["summarize", "summary", "explain the pdf", "explain the document", "what is this", "page"]
        
        is_general_query = any(keyword in user_query for keyword in general_keywords)
        
        # Load Model (Super-Lazy)
        model = get_embedding_model()
        question_embedding = model.encode(request.prompt).tolist()
        
        if is_general_query:
            # High Context Mode (200 Chunks)
            search_results = index.query(
                vector=question_embedding,
                top_k=200, 
                include_metadata=True
            )
        else:
            # Specific Mode
            search_results = index.query(
                vector=question_embedding,
                top_k=5, 
                include_metadata=True
            )

        context_text = ""
        for match in search_results['matches']:
            if 'metadata' in match:
                context_text += match['metadata']['text'] + "\n---\n"

        system_prompt = f"""
        You are a helpful assistant. Use the provided Context to answer the user's question.
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