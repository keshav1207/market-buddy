"""A tiny MCP server exposing stock news from Yahoo Finance."""

import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("market-news")


def _normalize(item: dict) -> dict:
    """yfinance has used two shapes for news items; handle both."""
    c = item.get("content", item)
    provider = c.get("provider")
    url = c.get("canonicalUrl")
    return {
        "title": c.get("title"),
        "summary": (c.get("summary") or "")[:300],
        "publisher": (
            provider.get("displayName") if isinstance(provider, dict)
            else item.get("publisher")
        ),
        "published": c.get("pubDate") or item.get("providerPublishTime"),
        "url": url.get("url") if isinstance(url, dict) else item.get("link"),
    }


@mcp.tool()
def get_stock_news(ticker: str, limit: int = 5) -> list[dict]:
    """Get recent news headlines for a stock ticker (e.g. AAPL).
    Returns title, summary, publisher, and link for each story."""
    items = yf.Ticker(ticker).news or []
    return [_normalize(i) for i in items[:limit]]


if __name__ == "__main__":
    mcp.run(transport="stdio")