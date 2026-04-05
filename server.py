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

if __name__ == "__main__":
    mcp.run(transport="http", port=3000)
