import asyncio
from typing import List
from pydantic import BaseModel


from .baseworker import WorkerBase

class InferenceResponse(BaseModel):
    scores: List[float]
    elapsed_ms: float

class RerankerWorker(WorkerBase):
    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                payload, fut = await self.queue.get()
                batch = [(payload, fut)]
                start = loop.time()
                while len(batch) < self.max_batch:
                    timeout = self.max_wait_ms / 1000 - (loop.time() - start)
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                flat_pairs = []
                futs = []
                for payload, f in batch:
                    q = payload.get("query", "")
                    docs = payload.get("documents", [])
                    flat_pairs.extend((q, d) for d in docs)
                    futs.append((len(docs), f))
                result = self.model.infer(flat_pairs)
                if isinstance(result, tuple):
                    scores, elapsed = result
                else:
                    scores, elapsed = result, 0.0
                offset = 0
                for length, f in futs:
                    resp_scores = scores[offset:offset+length]
                    if hasattr(resp_scores, "tolist"):
                        resp_scores = resp_scores.tolist()
                    resp = InferenceResponse(scores=resp_scores, elapsed_ms=elapsed)
                    f.set_result(resp)
                    offset += length
            except Exception as e:
                try:
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(e)
                except Exception:
                    pass
                print(f"[ERROR] reranking worker {self.name}: {e}")
