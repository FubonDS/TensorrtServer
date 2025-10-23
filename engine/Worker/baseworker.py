import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

from abc import ABC, abstractmethod

class WorkerBase(ABC):
    def __init__(self, name: str, model, max_batch: int = 32, max_wait_ms: int = 10):
        self.name = name
        self.model = model
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        if self._task is None:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @abstractmethod
    async def _run(self):
        pass
