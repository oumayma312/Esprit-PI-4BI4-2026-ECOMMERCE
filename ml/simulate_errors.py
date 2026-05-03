from __future__ import annotations

import argparse
import asyncio

import httpx


async def run(url: str, count: int) -> None:
    endpoints = [
        url,
        f"{url.rstrip('/')}/predict/supplier",
        f"{url.rstrip('/')}/predict/sell",
        f"{url.rstrip('/')}/predict/promote",
    ]
    stats = {"ok": 0, "errors": 0}

    async with httpx.AsyncClient(timeout=20.0) as client:
        for index in range(count):
            target = endpoints[index % len(endpoints)]
            try:
                if target.endswith("/predict/supplier"):
                    response = await client.post(target, json={"quantity": "bad", "unit_price": "bad"})
                elif target.endswith("/predict/sell"):
                    response = await client.get(target, params={"limit": -1})
                elif target.endswith("/predict/promote"):
                    response = await client.post(target, json={"date": "invalid-date"})
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
    parser = argparse.ArgumentParser(description="Simulate API errors.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    asyncio.run(run(args.url, max(1, args.count)))


if __name__ == "__main__":
    main()
