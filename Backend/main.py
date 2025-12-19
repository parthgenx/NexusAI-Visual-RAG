from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# 1. Setup Environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# 👇 FIX: Force FastEmbed to use the writable /tmp folder (CRITICAL FOR RENDER)
os.environ["FASTEMBED_CACHE_PATH"] = "/tmp"

from openai import OpenAI
from pinecone import Pinecone
import nest_asyncio
from llama_parse import LlamaParse
from fastembed import TextEmbedding 

nest_asyncio.apply()

# 2. Initialize Clients
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# --- LIGHTWEIGHT LOADING ---
model_cache = {}

def get_embedding_model():
    if "model" not in model_cache:
        print("Loading FastEmbed Model...")
        # uses the /tmp cache we defined above
        model_cache["model"] = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir="/tmp"
        )
        print("Model Loaded!")
    return model_cache["model"]

app = FastAPI()

# --- CORS ---
origins = [
    "http://localhost:5173",
    "https://nexus-ai-visual-rag.vercel.app",
    "https://nexus-ai-visual-rag.vercel.app/"
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
    return {"message": "NexusAI (Lightweight Edition) is Active 🟢"}

# --- BACKGROUND TASK ---
def process_upload_background(temp_filename: str):
    print(f"Processing {temp_filename}...")
    try:
        try:
            index.delete(delete_all=True)
        except Exception:
            pass

        print("Parsing...")
        parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown", 
            num_workers=4,
            language="en"
        )
        documents = parser.load_data(temp_filename)
        
        text = ""
        for doc in documents:
            text += doc.text
            
        if not text:
            print("No text extracted.")
            return
       
        print("Chunking...")
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
       
        print("Embedding (FastEmbed)...")
        model = get_embedding_model() 
        
        embeddings_generator = model.embed(chunks)
        embeddings_list = list(embeddings_generator)
        
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
            vectors.append({
                "id": f"chunk_{i}",
                "values": embedding.tolist(),
                "metadata": {"text": chunk}
            })

        print(f"Upserting {len(vectors)} vectors...")
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)

        print(f"✅ Success!")

    except Exception as e:
        print(f"❌ Task Failed: {e}")
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- ROUTES ---
@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    background_tasks.add_task(process_upload_background, temp_filename)
    return {"status": "success", "message": "Processing started"}

@app.post("/chat")
def generate_chat(request: ChatRequest):
    try:
        model = get_embedding_model()
        
        query_embedding_gen = model.embed([request.prompt])
        query_embedding = list(query_embedding_gen)[0].tolist()
        
        search_results = index.query(
            vector=query_embedding,
            top_k=10, 
            include_metadata=True
        )

        context_text = ""
        for match in search_results['matches']:
            if 'metadata' in match:
                context_text += match['metadata']['text'] + "\n---\n"

        system_prompt = f"""
        Answer based on the Context below. If unknown, say so.
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