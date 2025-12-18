# NexusAI

A production-grade Visual Retrieval Augmented Generation (RAG) system specialized for STEM and academic research workflows.

## Overview

NexusAI extends traditional RAG capabilities by incorporating computer vision to extract and understand complex visual elements from academic documents—including mathematical formulas, scientific diagrams, tables, and charts. Built for researchers, students, and professionals working with technical documentation.

## Architecture

### Core Components

- **Frontend Application**: React-based interface with Tailwind CSS styling
- **Backend API**: FastAPI server handling document processing and query orchestration
- **AI Processing Pipeline**: 
  - Llama 4 Scout (via Groq) for natural language understanding
  - LlamaParse for visual element extraction and OCR
  - OpenAI SDK integration for enhanced language processing
- **Vector Storage**: Pinecone serverless database for semantic search
- **Rendering Engine**: KaTeX for LaTeX mathematics and Markdown parsing

## Key Capabilities

### Visual Intelligence
Processes scanned PDFs and digital documents to extract:
- Mathematical equations and formulas
- Scientific diagrams and figures
- Data tables and charts
- Complex multi-modal content

### Semantic Retrieval
Leverages vector embeddings and Pinecone's similarity search to provide contextually relevant responses from large document collections.

### Academic Formatting
Renders mathematical notation and scientific content with proper formatting using LaTeX and Markdown support.

### Security & Privacy
Implements ephemeral file processing—documents are automatically deleted after analysis to ensure data privacy.

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Python 3.9+
- API keys for Groq, LlamaParse, OpenAI, and Pinecone

### Installation

#### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
```

#### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Environment Configuration

Create `.env` files in both `backend/` and `frontend/` directories with required API credentials:

**Backend `.env`:**
```
GROQ_API_KEY=your_groq_key
LLAMAPARSE_API_KEY=your_llamaparse_key
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

**Frontend `.env`:**
```
VITE_API_URL=http://localhost:8000
```

## Use Cases

- Academic paper analysis and literature review
- Technical documentation search
- Mathematical problem-solving with reference materials
- Research note organization and retrieval
- STEM education and tutoring support

## Project Structure

```
nexusai/
├── backend/
│   ├── main.py              # FastAPI application entry
│   ├── requirements.txt     # Python dependencies
│   └── ...
├── frontend/
│   ├── src/                 # React components
│   ├── package.json         # npm dependencies
│   └── ...
└── README.md
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Frontend | React, Tailwind CSS |
| Backend | Python, FastAPI |
| AI/ML | Llama 4 Scout, LlamaParse, OpenAI |
| Database | Pinecone (Vector DB) |
| Rendering | KaTeX, Markdown-it |

## Development Status

This project is actively maintained and suitable for research and educational purposes. Contributions and feedback are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, feedback, or collaboration opportunities:

**Email**: parthgenx@gmail.com

---

**Note**: This system requires active API subscriptions to Groq, LlamaParse, OpenAI, and Pinecone services. Ensure you have appropriate usage limits configured for production deployment.
