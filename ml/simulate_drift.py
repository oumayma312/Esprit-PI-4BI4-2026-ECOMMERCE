from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

import httpx


async def run(url: str, count: int) -> None:
    target = f"{url.rstrip('/')}/drift/supplier"
    base_day = date.today()
    stats = {"ok": 0, "errors": 0}

    async with httpx.AsyncClient(timeout=30.0) as client:
        for index in range(count):
            payload = {
                "quantity": 1000 + (index * 50),
                "unit_price": 1 + (index % 3),
                "total_ht": 5000 + (index * 120),
                "total_ttc": 5900 + (index * 130),
                "governorate": f"GOV-{index % 5}",
                "city": f"CITY-{index % 8}",
            }
            try:
                response = await client.post(target, json=payload)
                if response.status_code < 400:
                    stats["ok"] += 1
                else:
                    stats["errors"] += 1
            except Exception:
                stats["errors"] += 1

    print({"target": target, "requests": count, **stats})


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate supplier drift detection checks.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(run(args.url, max(1, args.count)))


if __name__ == "__main__":
    main()
