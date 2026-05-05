from __future__ import annotations

import argparse
import asyncio

import httpx


async def run(url: str, count: int) -> None:
    base = url.rstrip("/")
    endpoints = [
        f"{base}/predict",
        f"{base}/predict/batch",
    ]
    stats = {"ok": 0, "errors": 0}

    async with httpx.AsyncClient(timeout=20.0) as client:
        for index in range(count):
            target = endpoints[index % len(endpoints)]
            try:
                if target.endswith("/predict"):
                    response = await client.post(
                        target,
                        json={"recency": "bad", "frequency": "bad", "monetary_total": "bad", "monetary_trend": "bad", "product_diversity": "bad", "channel_diversity": "bad", "lifetime_days": "bad", "purchase_velocity": "bad", "avg_price": "bad"},
                    )
                elif target.endswith("/predict/batch"):
                    response = await client.post(
                        target,
                        json=[{"recency": "bad", "frequency": "bad", "monetary_total": "bad", "monetary_trend": "bad", "product_diversity": "bad", "channel_diversity": "bad", "lifetime_days": "bad", "purchase_velocity": "bad", "avg_price": "bad"}],
                    )
                else:
                    response = await client.get(target)
                if response.status_code < 400:
                    stats["ok"] += 1
                else:
                    stats["errors"] += 1
            except Exception:
                stats["errors"] += 1

    print({"target": url, "requests": count, **stats})


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate churn API errors.")
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(run(args.url, max(1, args.count)))


if __name__ == "__main__":
    main()
