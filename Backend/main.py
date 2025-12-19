from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import uuid
import tempfile
import gc


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")


required_keys = {
    "GROQ_API_KEY": groq_api_key,
    "PINECONE_API_KEY": pinecone_api_key,
    "PINECONE_INDEX_NAME": pinecone_index_name,
    "LLAMA_CLOUD_API_KEY": llama_cloud_api_key
}
for key_name, key_value in required_keys.items():
    if not key_value:
        raise ValueError(f"Missing required environment variable: {key_name}")

temp_dir = tempfile.gettempdir()
os.environ["FASTEMBED_CACHE_PATH"] = temp_dir

from openai import OpenAI
from pinecone import Pinecone
import nest_asyncio
from llama_parse import LlamaParse
from fastembed import TextEmbedding 

nest_asyncio.apply()


client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

model_cache = {}

def get_embedding_model():
    if "model" not in model_cache:
        print("Loading FastEmbed Model...")
        model_cache["model"] = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            cache_dir=temp_dir
        )
        print("Model Loaded!")
    return model_cache["model"]


def smart_chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    import re
    text = ' '.join(text.split())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            words = current_chunk.split()
            overlap_words = words[-min(len(words), overlap // 5):]
            current_chunk = ' '.join(overlap_words) + ' ' + sentence
        else:
            current_chunk += ' ' + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks



ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.pptx', '.xlsx'}
MAX_FILE_SIZE_MB = 50


app = FastAPI()


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


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type '{file_ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    content = await file.read()
    

    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB"
        )
    
    temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}{file_ext}")
    
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(content)
        

        del content
        gc.collect()
        

        print("Clearing existing vectors...")
        try:
            index.delete(delete_all=True)
        except Exception:
            pass


        print(f"Parsing {file.filename}...")
        parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown", 
            num_workers=2,
            language="en"
        )
        documents = parser.load_data(temp_filename)
        

        text = ""
        for doc in documents:
            text += doc.text
        del documents
        gc.collect()
            
        if not text:
            raise HTTPException(status_code=400, detail="No text extracted from document.")
       

        print("Chunking...")
        chunks = smart_chunk_text(text, chunk_size=800, overlap=100)
        

        del text
        gc.collect()
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid text chunks produced.")
        
        print(f"Processing {len(chunks)} chunks in batches...")
        

        model = get_embedding_model()
        batch_size = 10
        total_upserted = 0
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            

            embeddings = list(model.embed(batch_chunks))
            

            vectors = []
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                vectors.append({
                    "id": f"chunk_{batch_start + i}",
                    "values": embedding.tolist(),
                    "metadata": {"text": chunk}
                })
            

            index.upsert(vectors=vectors)
            total_upserted += len(vectors)
            

            del embeddings, vectors, batch_chunks
            gc.collect()
            
            print(f"  Upserted batch {batch_start//batch_size + 1}: {total_upserted}/{len(chunks)} chunks")


        del chunks
        gc.collect()

        print(f"✅ Successfully processed {file.filename}")
        return {
            "status": "success", 
            "message": f"Document processed! {total_upserted} chunks indexed.",
            "chunks_count": total_upserted
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Processing Failed: {e}")
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        gc.collect()


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


        del search_results
        gc.collect()

        if not context_text.strip():
            return {"response": "I cannot find relevant information. Please upload a document first."}

        system_prompt = f"""
        You are a helpful AI assistant. Answer using ONLY the Context below.
        
        Guidelines:
        - If the answer can be inferred from the Context, answer it.
        - If the Context is irrelevant, say "I cannot find the answer in the document."
        - Keep answers concise and factual.

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
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))