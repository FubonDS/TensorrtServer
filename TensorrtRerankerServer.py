import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List

from inferencers.tensorrtreranker import TensorRTReranker

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(batch_worker())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

model = TensorRTReranker(
    engine_path="./reranker_models/trt_models/bge_reranker_large_dynamic_bs.trt",
    tokenizer_path="bge-reranker-large-tokenizer"
)
executor = ThreadPoolExecutor(max_workers=2)

request_queue = asyncio.Queue()
MAX_BATCH = 8  
MAX_WAIT_MS = 10


class InferenceRequest(BaseModel):
    query: str
    documents: List[str]


class InferenceResponse(BaseModel):
    scores: List[float]
    elapsed_ms: float


@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await request_queue.put((req.query, req.documents, fut))
    return await fut


async def batch_worker():
    while True:
        try:
            query, documents, fut = await request_queue.get()
            batch = [(query, documents, fut)]
            start = asyncio.get_event_loop().time()

            while len(batch) < MAX_BATCH:
                try:
                    timeout = MAX_WAIT_MS / 1000 - (asyncio.get_event_loop().time() - start)
                    if timeout <= 0:
                        break
                    item = await asyncio.wait_for(request_queue.get(), timeout=timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            
            flat_pairs = []
            futs = []
            for q, docs, f in batch:
                flat_pairs.extend((q, d) for d in docs)
                futs.append((len(docs), f))
            
            print(f"[DEBUG] Processing {len(flat_pairs)} query-doc pairs")
            scores, elapsed = model.infer(flat_pairs)
            offset = 0
            for length, f in futs:
                resp_scores = scores[offset:offset + length]
                resp = InferenceResponse(
                    scores=resp_scores,
                    elapsed_ms=elapsed 
                )
                f.set_result(resp)
                offset += length
            

        except Exception as e:
            print(f"[ERROR] in batch_worker: {e}")
