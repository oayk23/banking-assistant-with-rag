# main.py

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from banking_assistant.full_pipeline import FullPipeline
from pathlib import Path
import uvicorn
import os

# 💡 Pipeline başlatılıyor (ilk istekten önce yüklenmiş olur)
ARTIFACTS_PATH = Path("artifacts")
INDEXES_PATH = Path("indexes")
DEVICE = "cuda"  # veya "cpu"

pipeline = FullPipeline(
    artifacts_path=ARTIFACTS_PATH,
    indexes_path=INDEXES_PATH,
    device=DEVICE
)

app = FastAPI(
    title="Banking Assistant",
    description="Streaming RAG-based chatbot",
    version="1.0.0"
)


@app.post("/chat")
async def chat(request: Request):
    """
    Stream-based LLM yanıtı döner
    """
    data = await request.json()
    query = data.get("query")

    if not query:
        return {"error": "Query alanı boş"}

    def response_generator():
        for token in pipeline.pipe(query):
            yield token

    return StreamingResponse(response_generator(), media_type="text/plain")
