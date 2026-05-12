import asyncio, time

async def main():
    domain = "pagerduty.com"

    from signals.hiring import get_hiring_signal
    from signals.pricing import get_pricing_signal
    from signals.reviews import get_reviews_signal
    from signals.traffic import get_traffic_signal
    from signals.headcount import get_headcount_signal

    for label, coro_fn in [
        ("traffic",   lambda: get_traffic_signal(domain,   force_refresh=True)),
        ("headcount", lambda: get_headcount_signal(domain, force_refresh=True)),
        ("hiring",    lambda: get_hiring_signal(domain,    force_refresh=True)),
        ("pricing",   lambda: get_pricing_signal(domain,   force_refresh=True)),
        ("reviews",   lambda: get_reviews_signal(domain,   force_refresh=True)),
    ]:
        t = time.time()
        try:
            result = await asyncio.wait_for(coro_fn(), timeout=25)
            elapsed = time.time() - t
            # Print key fields only
            summary = {k: v for k, v in result.items() if k not in ("raw_titles", "extra")}
            print(f"\n[{label}] {elapsed:.1f}s → {summary}")
        except asyncio.TimeoutError:
            print(f"\n[{label}] TIMED OUT after 25s")
        except Exception as e:
            print(f"\n[{label}] ERROR: {e}")

asyncio.run(main())