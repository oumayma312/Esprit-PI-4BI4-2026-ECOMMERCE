from __future__ import annotations

import argparse
import asyncio

import httpx


async def run(url: str, count: int) -> None:
    base = url.rstrip("/")
    drift_endpoint = f"{base}/drift/churn"
    predict_endpoint = f"{base}/predict"
    stats = {"ok": 0, "errors": 0}

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

    async with httpx.AsyncClient(timeout=30.0) as client:
        for index in range(count):
            try:
                if index < 10:
                    await client.post(predict_endpoint, json=good_payload)
                elif index % 2 == 0:
                    response = await client.get(drift_endpoint)
                else:
                    drifted = good_payload.copy()
                    drifted["recency"] = 300.0 + (index * 10)
                    drifted["frequency"] = 0.1
                    drifted["monetary_total"] = 100.0
                    drifted["monetary_trend"] = -0.5
                    response = await client.post(predict_endpoint, json=drifted)
                if response.status_code < 400:
                    stats["ok"] += 1
                else:
                    stats["errors"] += 1
            except Exception:
                stats["errors"] += 1

    print({"target": url, "requests": count, **stats})


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate churn drift detection checks.")
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run(args.url, max(1, args.count)))


if __name__ == "__main__":
    main()
