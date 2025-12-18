import { useState } from "react";
import axios from "axios";
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

function App() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("idle");

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadStatus("idle");
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploadStatus("uploading");
    const formData = new FormData();
    formData.append("file", file);

    try {
      await axios.post("http://127.0.0.1:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadStatus("success");
    } catch (error) {
      console.error(error);
      setUploadStatus("error");
    }
  };

  const handleChat = async () => {
    if (!prompt) return;
    setLoading(true);
    setResponse("");

    try {
      const res = await axios.post("http://127.0.0.1:8000/chat", {
        prompt: prompt,
      });
      setResponse(res.data.response);
    } catch (error) {
      console.error(error);
      setResponse("Error: Could not connect to AI.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center p-6 font-sans">
      <div className="w-full max-w-3xl bg-slate-800/50 backdrop-blur-xl rounded-3xl shadow-2xl border border-slate-700 overflow-hidden">
        
        {/* Header */}
        <div className="bg-gradient-to-r from-indigo-900 to-slate-900 p-8 border-b border-slate-700 flex items-center justify-between">
            <div>
                <h1 className="text-3xl font-extrabold text-white tracking-tight">
                    Nexus<span className="text-indigo-400">AI</span>
                </h1>
                <p className="text-slate-400 text-sm mt-2">Visual RAG System • LlamaParse Enabled</p>
            </div>
        </div>

        <div className="p-8 space-y-8">
            
            {/* Upload Area */}
            <div className={`relative group transition-all duration-300 rounded-2xl border-2 border-dashed p-8 text-center ${
                uploadStatus === "success" ? "border-green-500/50 bg-green-500/10" : "border-slate-600 hover:border-indigo-400 hover:bg-slate-700/50"
            }`}>
                <input type="file" onChange={handleFileChange} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10" />
                <div className="space-y-3 pointer-events-none">
                    <div className="text-4xl mb-2">{uploadStatus === "success" ? "✅" : "📄"}</div>
                    <p className="text-slate-200 font-medium text-lg">
                        {file ? file.name : "Drop your Document here"}
                    </p>
                </div>
                {file && uploadStatus !== "success" && (
                    <button 
                        onClick={(e) => { e.stopPropagation(); handleUpload(); }}
                        disabled={uploadStatus === "uploading"}
                        className="relative z-20 mt-4 bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-2 rounded-lg font-medium transition-all shadow-lg disabled:opacity-50"
                    >
                        {uploadStatus === "uploading" ? "Analyzing..." : "Start Analysis"}
                    </button>
                )}
                {uploadStatus === "success" && <p className="mt-4 text-green-400 text-sm">Ready to chat.</p>}
                {uploadStatus === "error" && <p className="mt-4 text-red-400 text-sm">Upload failed.</p>}
            </div>

            {/* Chat Area */}
            <div className="bg-slate-900/50 rounded-2xl border border-slate-700/50 p-6 shadow-inner">
                {response ? (
                    <div className="mb-6 animate-fade-in-up">
                        <div className="flex gap-4">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 flex-shrink-0 flex items-center justify-center font-bold text-xs">AI</div>
                            
                            {/* FIX: Move className to this wrapper div */}
                            <div className="bg-slate-800/50 p-6 rounded-2xl rounded-tl-none border border-slate-700 text-slate-300 w-full overflow-x-auto prose prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-slate-900 prose-headings:text-indigo-300 prose-a:text-blue-400">
                                <ReactMarkdown
                                    remarkPlugins={[remarkMath]}
                                    rehypePlugins={[rehypeKatex]}
                                >
                                    {response}
                                </ReactMarkdown>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="h-32 flex items-center justify-center text-slate-600 text-sm italic border border-dashed border-slate-800 rounded-xl mb-6">
                        Ask about the document content...
                    </div>
                )}

                <div className="relative">
                    <textarea
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="e.g. Summarize the second page..."
                        className="w-full bg-slate-800 text-slate-200 placeholder-slate-500 rounded-xl border border-slate-600 focus:border-indigo-500 p-4 pr-16 outline-none resize-none shadow-lg"
                        rows="3"
                        onKeyDown={(e) => {
                            if(e.key === "Enter" && !e.shiftKey) {
                                e.preventDefault();
                                handleChat();
                            }
                        }}
                    />
                    <button 
                        onClick={handleChat}
                        disabled={loading || !prompt}
                        className="absolute bottom-3 right-3 p-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-all disabled:opacity-50"
                    >
                        {loading ? "..." : "➤"}
                    </button>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}

export default App;