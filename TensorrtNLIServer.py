import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List

from inferencers.tensorrtnli import TensorRTNLI

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

model = TensorRTNLI("nli_model_dynamic_bs.trt")
executor = ThreadPoolExecutor(max_workers=8)

request_queue = asyncio.Queue()
MAX_BATCH = 32  
MAX_WAIT_MS = 10


class InferenceRequest(BaseModel):
    premises: List[str]
    hypotheses: List[str]


class InferenceResponse(BaseModel):
    predictions: List[str]
    logits: List[List[float]]
    elapsed_ms: float


@app.post("/infer", response_model=InferenceResponse)
async def infer(req: InferenceRequest):
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    await request_queue.put((req.premises, req.hypotheses, fut))
    return await fut


async def batch_worker():
    while True:
        try:
            premises, hypotheses, fut = await request_queue.get()
            batch = [(premises, hypotheses, fut)]
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

            all_premises, all_hypotheses = [], []
            futs = []
            for p, h, f in batch:
                all_premises.extend(p)
                all_hypotheses.extend(h)
                futs.append((len(p), f))

            preds, logits, elapsed = model.infer(all_premises, all_hypotheses)

            offset = 0
            for length, f in futs:
                resp = InferenceResponse(
                    predictions=preds[offset:offset+length],
                    logits=logits[offset:offset+length].tolist(),
                    elapsed_ms=elapsed
                )
                f.set_result(resp)
                offset += length

        except Exception as e:
            print(f"[ERROR] in batch_worker: {e}")