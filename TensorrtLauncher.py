import argparse
import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from engine import (EmbeddingWorker, NLIWorker, RerankerWorker,
                                  TensorrtBuilder)
from engine.schema import EmbeddingRequest

logging.basicConfig(
    level=logging.INFO,   
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

CONFIG_PATH = "./configs/config.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT Inference Server")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    
    return parser.parse_args()
    
args = parse_args()
    
builder = TensorrtBuilder(args.config)
available_model_dict = builder.model

WORKERS: Dict[str, Any] = {}

for key, lst in available_model_dict.items():
    if key == "embedding_models":
        for model_name in lst:
            worker = EmbeddingWorker(
                    name=model_name,
                    model=getattr(builder, model_name)
               )
            WORKERS[model_name] = worker
    elif key == "reranking_models":
        for model_name in lst:
            worker = RerankerWorker(
                    name=model_name,
                    model=getattr(builder, model_name)
               )
            WORKERS[model_name] = worker
    elif key == "nli_models":   
        for model_name in lst:
            worker = NLIWorker(
                    name=model_name,
                    model=getattr(builder, model_name)
               )
            WORKERS[model_name] = worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    for w in WORKERS.values():
        await w.start()
    yield
    for w in WORKERS.values():
        await w.stop()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models")
async def list_models():
    return {"models": available_model_dict}

@app.post("/infer/{model_name}")
async def infer(model_name: str, request: Request):
    if model_name not in WORKERS:
        raise HTTPException(status_code=404, detail=f"model {model_name} not found")
    payload = await request.json()
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await WORKERS[model_name].queue.put((payload, fut))
    return await fut

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    model_name = request.model

    if request.query is not None:
        # for reranker models
        if model_name not in available_model_dict.get("reranking_models", []):
            raise HTTPException(status_code=400, detail=f"Reranking model {model_name} not found")
    else:
        if model_name not in available_model_dict.get("embedding_models", []):
            raise HTTPException(status_code=400, detail=f"Embedding model {model_name} not found")
        
    if isinstance(request.input, str):
        documents = [request.input]
    elif isinstance(request.input, list):
        documents = request.input

    if len(documents) == 0:
        raise HTTPException(status_code=400, detail="Input text is empty")

    response_data = []
    try:
        if request.query is not None:
            payload = {"query": request.query, "documents": documents}
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            await WORKERS[model_name].queue.put((payload, fut))
            results = await fut
            scores = results.scores
            for idx, score in enumerate(scores):
                response_data.append(
                    {
                        "object": "reranking",
                        "embedding": float(score),
                        "index": idx
                    }
                )
            return {
                "object": "list",
                "data": response_data,
                "model": model_name,
                "usage": {
                    "prompt_tokens": len(request.query.split()),
                    "total_tokens": sum(len(t.split()) for t in documents)
                } 
            }
        else:
            payload = {"documents": documents}
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            await WORKERS[model_name].queue.put((payload, fut))
            results = await fut
            embeddings = results.embeddings
            for idx, embedding in enumerate(embeddings):
                response_data.append(
                    {
                        "object": "embedding",
                        "embedding": embedding,
                        "index": idx,
                    }
                )
            return {
                "object": "list",
                "data": response_data,
                "model": model_name,
                "usage": {
                    "prompt_tokens": sum(len(t.split()) for t in documents),
                    "total_tokens": sum(len(t.split()) for t in documents)
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
            
if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)