import asyncio
from fastmcp import FastMCP
from signals.hiring import get_hiring_signal

mcp = FastMCP("sri")


@mcp.tool
def health_check() -> dict:
    """check if the server is running"""
    return {"status": "ok", "server": "saas-revenue-intelligence"}


@mcp.tool
async def hiring_signal(domain: str) -> dict:
    """get hiring signal for a given domain"""
    return await get_hiring_signal(domain)


@mcp.tool
async def batch_hiring_signals(domains, limit=10) -> list:
    """get hiring signals for a batch of domains with concurrency control"""
    if isinstance(domains, str):
        domains = [d.strip() for d in domains.split(",") if d.strip()]
    semaphore = asyncio.Semaphore(limit)

    async def worker(domain):
        async with semaphore:
            try:
                result = await get_hiring_signal(domain)
                return {
                    "domain": domain,
                    "result": result
                }
            except Exception:
                return {
                    "domain": domain,
                    "result": {
                        "open_roles": 0,
                        "engineering_roles": 0,
                        "sales_roles": 0,
                        "source": "error",
                        "velocity_score": 0.0,
                        "raw_titles": []
                    }
                }

    tasks = [worker(d) for d in domains]
    results = await asyncio.gather(*tasks)

    return results

if __name__ == "__main__":
    mcp.run(transport="http", port=3000)
