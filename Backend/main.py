from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from openai import OpenAI
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
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

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    return {"message": "Pinecone RAG Backend (LlamaParse Edition) is Active 🌲🦙"}

# --- ROUTE 1: Upload & Memorize 📂 ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # 1. Save File to Disk

        index.delete(delete_all=True)
        
        with open(temp_filename, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 2. Parse File from Disk
        parser = LlamaParse(
            api_key=llama_cloud_api_key,
            result_type="markdown", 
            num_workers=4,
            verbose=True,
            language="en"
        )
        documents = await parser.aload_data(temp_filename)
        
        # 3. Process Text
        text = ""
        for doc in documents:
            text += doc.text
            
        if not text:
            return {"error": "LlamaParse could not extract text from this file."}
       
        # 4. Chunk & Embed
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
       
        vectors = []
        for i, chunk in enumerate(chunks):
            vector_embedding = embedding_model.encode(chunk).tolist()
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
        # --- LOGIC UPGRADE: Detect General/Navigation Questions ---
        user_query = request.prompt.lower()
        general_keywords = ["summarize", "summary", "explain the pdf", "explain the document", "what is this", "page"]
        
        is_general_query = any(keyword in user_query for keyword in general_keywords)
        
        question_embedding = embedding_model.encode(request.prompt).tolist()
        
        if is_general_query:
            # BROADCAST MODE: Fetch MUCH more context (Top 20 chunks = approx 10k chars)
            # This gives the AI a "Bird's Eye View" of the document
            search_results = index.query(
                vector=question_embedding,
                top_k=20, # <--- INCREASED FROM 2 TO 20
                include_metadata=True
            )
        else:
            # SNIPER MODE: Fetch specific context (Top 3 chunks)
            # This keeps answers precise for specific questions
            search_results = index.query(
                vector=question_embedding,
                top_k=3,
                include_metadata=True
            )

        # Build Context
        context_text = ""
        for match in search_results['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                context_text += match['metadata']['text'] + "\n---\n"

        system_prompt = f"""
        You are a helpful assistant. Use the provided Context to answer the user's question.
        
        Important:
        - If the user asks for a summary or about a specific page, use the Context provided to infer the answer.
        - The Context may contain the page numbers or sequential text.
        - If you really cannot find the answer, say "I don't have enough context to answer that."
        
        Context:
        {context_text}
        """

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt}
            ]
        )
       
        return {"response": response.choices[0].message.content}

    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))