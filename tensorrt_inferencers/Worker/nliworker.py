import asyncio
from typing import List
from pydantic import BaseModel


from .baseworker import WorkerBase

class InferenceResponse(BaseModel):
    predictions: List[str]
    logits: List[List[float]]
    elapsed_ms: float

class NLIWorker(WorkerBase):
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
                all_premises, all_hypotheses = [], []
                futs = []
                for payload, f in batch:
                    p = payload.get("premises", [])
                    h = payload.get("hypotheses", [])
                    all_premises.extend(p)
                    all_hypotheses.extend(h)
                    futs.append((len(p), f))

                preds, logits, elapsed = self.model.infer(all_premises, all_hypotheses)

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
                try:
                    for _, fut in batch:
                        if not fut.done():
                            fut.set_exception(e)
                except Exception:
                    pass
                print(f"[ERROR] embedding worker {self.name}: {e}")
