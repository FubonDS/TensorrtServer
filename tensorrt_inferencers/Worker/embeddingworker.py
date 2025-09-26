import asyncio
from typing import List
from pydantic import BaseModel


from .baseworker import WorkerBase

class InferenceResponse(BaseModel):
    embeddings: List[List[float]]
    elapsed_ms: float

class EmbeddingWorker(WorkerBase):
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
                all_docs = []
                lengths = []
                for payload, _ in batch:
                    docs = payload.get("documents", [])
                    all_docs.extend(docs)
                    lengths.append(len(docs))
                result = self.model.infer(all_docs)
                if isinstance(result, tuple):
                    embeddings, elapsed = result
                else:
                    embeddings, elapsed = result, 0.0
                offset = 0
                for length, (_, f) in zip(lengths, batch):
                    seg = embeddings[offset:offset+length]
                    if hasattr(seg, "tolist"):
                        seg = seg.tolist()
                    resp = InferenceResponse(embeddings=seg, elapsed_ms=elapsed)
                    f.set_result(resp)
                    offset += length
            except Exception as e:
                try:
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(e)
                except Exception:
                    pass
                print(f"[ERROR] embedding worker {self.name}: {e}")
