from __future__ import annotations

import argparse
import asyncio
from typing import Any

import httpx


async def run(url: str, count: int, concurrency: int) -> None:
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)
    sem = asyncio.Semaphore(concurrency)
    stats = {"ok": 0, "errors": 0}

    async with httpx.AsyncClient(limits=limits, timeout=20.0) as client:
        async def hit(index: int) -> None:
            async with sem:
                try:
                    response = await client.get(url)
                    if response.status_code < 400:
                        stats["ok"] += 1
                    else:
                        stats["errors"] += 1
                except Exception:
                    stats["errors"] += 1

        await asyncio.gather(*(hit(index) for index in range(count)))

    print({"target": url, "requests": count, "concurrency": concurrency, **stats})


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate high API traffic.")
    parser.add_argument("--url", default="http://localhost:8000/health")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(run(args.url, max(1, args.count), max(1, args.concurrency)))


if __name__ == "__main__":
    main()
