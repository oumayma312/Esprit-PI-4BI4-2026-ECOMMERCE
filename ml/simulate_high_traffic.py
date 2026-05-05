from __future__ import annotations

import argparse
import asyncio
from typing import Any

import httpx


async def run(url: str, count: int, concurrency: int) -> None:
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency)
    sem = asyncio.Semaphore(concurrency)
    stats: dict[str, Any] = {"ok": 0, "errors": 0}

    good_payload = {
        "recency": 30.0,
        "frequency": 5.0,
        "monetary_total": 1500.0,
        "monetary_trend": 0.1,
        "product_diversity": 3.0,
        "channel_diversity": 2.0,
        "lifetime_days": 365.0,
        "purchase_velocity": 0.5,
        "avg_price": 300.0,
        "channel": "online",
    }

    async with httpx.AsyncClient(limits=limits, timeout=20.0) as client:
        async def hit(index: int) -> None:
            async with sem:
                try:
                    if index % 3 == 0:
                        response = await client.get(url)
                    else:
                        response = await client.post(f"{url.rstrip('/').rstrip('/health')}/predict", json=good_payload)
                    if response.status_code < 400:
                        stats["ok"] += 1
                    else:
                        stats["errors"] += 1
                except Exception:
                    stats["errors"] += 1

        await asyncio.gather(*(hit(index) for index in range(count)))

    print({"target": url, "requests": count, "concurrency": concurrency, **stats})


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate high churn API traffic.")
    parser.add_argument("--url", default="http://localhost:8001/health")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(run(args.url, max(1, args.count), max(1, args.concurrency)))


if __name__ == "__main__":
    main()
