from __future__ import annotations

import datetime as dt
import time

import run_pipeline


def _seconds_until_next_run(target_hour: int = 1, target_minute: int = 0) -> float:
    now = dt.datetime.now()
    next_run = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
    if next_run <= now:
        next_run = next_run + dt.timedelta(days=1)
    return (next_run - now).total_seconds()


def main() -> None:
    print("Scheduler started: runs every day at 01:00")
    while True:
        wait_s = _seconds_until_next_run(1, 0)
        next_at = dt.datetime.now() + dt.timedelta(seconds=wait_s)
        print(f"Next run at: {next_at:%Y-%m-%d %H:%M:%S}")
        time.sleep(wait_s)
        try:
            result = run_pipeline.run()
            print("Pipeline finished:", result)
        except Exception as e:
            print("Pipeline failed:", repr(e))


if __name__ == "__main__":
    main()
