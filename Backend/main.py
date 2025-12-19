from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uuid
import tempfile

# 1. Setup Environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")

# FIX #2: Validate required API keys at startup
required_keys = {
    "GROQ_API_KEY": groq_api_key,
    "PINECONE_API_KEY": pinecone_api_key,
    "PINECONE_INDEX_NAME": pinecone_index_name,
    "LLAMA_CLOUD_API_KEY": llama_cloud_api_key
}
for key_name, key_value in required_keys.items():
    if not key_value:
        raise ValueError(f"Missing required environment variable: {key_name}")

# FIX #6: Use cross-platform temp directory
temp_dir = tempfile.gettempdir()
os.environ["FASTEMBED_CACHE_PATH"] = temp_dir

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
        # FIX #6: Use cross-platform temp directory
        model_cache["model"] = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=temp_dir
        )
        print("Model Loaded!")
    return model_cache["model"]


# FIX #1: Smart chunking that preserves sentence boundaries
def smart_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Chunk text by sentences with overlap for better context preservation.
    This prevents words/sentences from being cut in half.
    """
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Split by sentence-ending punctuation
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds chunk size, save current and start new
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep last part for overlap
            words = current_chunk.split()
            overlap_words = words[-min(len(words), overlap // 5):]  # ~10 words for overlap
            current_chunk = ' '.join(overlap_words) + ' ' + sentence
        else:
            current_chunk += ' ' + sentence if current_chunk else sentence
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


# FIX #5: Allowed file types
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.pptx', '.xlsx'}
MAX_FILE_SIZE_MB = 50


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


# FIX #3, #4, #5, #7: Synchronous processing with validation
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document synchronously.
    - Validates file type and size
    - Uses UUID for unique temp filenames
    - Returns only after processing is complete
    """
    
    # FIX #5: Validate file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type '{file_ext}' not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content
    content = await file.read()
    
    # FIX #5: Validate file size
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum allowed: {MAX_FILE_SIZE_MB}MB"
        )
    
    # FIX #3: Use UUID for unique temp filename to prevent race conditions
    temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
    
    try:
        # Write to temp file
        with open(temp_filename, "wb") as buffer:
            buffer.write(content)
        
        # Clear existing vectors (single document mode)
        print("Clearing existing vectors...")
        try:
            index.delete(delete_all=True)
        except Exception:
            pass

        # Parse document
        print(f"Parsing {file.filename}...")
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
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")
       
        # FIX #1: Use smart chunking instead of naive character split
        print("Chunking with sentence awareness...")
        chunks = smart_chunk_text(text, chunk_size=500, overlap=50)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document produced no valid text chunks.")
       
        # Generate embeddings
        print("Embedding (FastEmbed)...")
        model = get_embedding_model() 
        
        embeddings_generator = model.embed(chunks)
        embeddings_list = list(embeddings_generator)
        
        # Prepare vectors
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
            vectors.append({
                "id": f"chunk_{i}",
                "values": embedding.tolist(),
                "metadata": {"text": chunk}
            })

        # Upsert to Pinecone in batches
        print(f"Upserting {len(vectors)} vectors...")
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)

        print(f"✅ Successfully processed {file.filename}")
        return {
            "status": "success", 
            "message": f"Document processed successfully. {len(chunks)} chunks indexed.",
            "chunks_count": len(chunks)
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Processing Failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.post("/chat")
def generate_chat(request: ChatRequest):
    try:
        model = get_embedding_model()
        
        # 1. Embed Question
        query_embedding_gen = model.embed([request.prompt])
        query_embedding = list(query_embedding_gen)[0].tolist()
        
        # 2. Search Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=10,
            include_metadata=True
        )

        context_text = ""
        for match in search_results['matches']:
            if 'metadata' in match:
                context_text += match['metadata']['text'] + "\n---\n"

        # Debug logging
        print("--- DEBUG: RETRIEVED CONTEXT ---")
        print(context_text[:500] if context_text else "No context found")
        print("--------------------------------")

        if not context_text.strip():
            return {"response": "I cannot find relevant information in the document to answer that. Please make sure you have uploaded a document first."}

        # System prompt for RAG
        system_prompt = f"""
        You are a helpful AI assistant. Answer the user's question using ONLY the Context provided below.
        
        Guidelines:
        - If the answer can be inferred from the Context, answer it.
        - If the Context is completely irrelevant to the question, say "I cannot find the answer in the document."
        - Keep your answer concise and based on facts from the text.

        Context:
        {context_text}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.3
        )
        
        return {"response": response.choices[0].message.content}

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))