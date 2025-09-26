from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from tensorrt_inferencers import TensorrtBuilder, NLIWorker, RerankerWorker, EmbeddingWorker

CONFIG_PATH = "./configs/config.yaml"

builder = TensorrtBuilder(CONFIG_PATH)
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
print(f"Available models: {list(WORKERS.keys())}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # start all workers
    for w in WORKERS.values():
        await w.start()
    yield
    # stop all workers
    for w in WORKERS.values():
        await w.stop()

app = FastAPI(lifespan=lifespan)

@app.get("/models")
async def list_models():
    return {"models": list(WORKERS.keys())}

@app.post("/infer/{model_name}")
async def infer(model_name: str, request: Request):
    if model_name not in WORKERS:
        raise HTTPException(status_code=404, detail=f"model {model_name} not found")
    payload = await request.json()
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    await WORKERS[model_name].queue.put((payload, fut))
    return await fut