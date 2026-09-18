"""
Market Buddy - a simple stock Q&A agent built with LangChain.

"""

import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


# ---------- Tools ----------
# The model only sees each tool's name, docstring, and argument types.

@tool
def get_quote(ticker: str) -> dict:
    """Get the latest price, market cap, and 52-week range
    for a stock ticker (e.g. AAPL)."""
    info = yf.Ticker(ticker).fast_info
    return {
        "ticker": ticker.upper(),
        "price": round(info.last_price, 2),
        "market_cap": info.market_cap,
        "52w_high": round(info.year_high, 2),
        "52w_low": round(info.year_low, 2),
        "currency": info.currency,
    }


@tool
def get_performance(ticker: str, period: str = "1mo") -> dict:
    """Get a stock's % return over a period.
    Valid periods: 5d, 1mo, 3mo, 6mo, 1y, ytd, 5y."""
    hist = yf.Ticker(ticker).history(period=period)
    if hist.empty:
        return {"error": f"No data found for {ticker}"}
    start, end = hist["Close"].iloc[0], hist["Close"].iloc[-1]
    return {
        "ticker": ticker.upper(),
        "period": period,
        "start_price": round(start, 2),
        "end_price": round(end, 2),
        "return_pct": round((end - start) / start * 100, 2),
    }


@tool
def get_fundamentals(ticker: str) -> dict:
    """Get basic valuation and company info: sector, P/E, dividend yield, beta."""
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker.upper(),
        "name": info.get("longName"),
        "sector": info.get("sector"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
    }

@tool
def compare_performance(tickers: list[str], period: str = "1mo") -> list[dict]:
    """Compare the % return of MULTIPLE stock tickers over the same period.
    Use this instead of get_performance when the user asks about
    two or more stocks. Valid periods: 5d, 1mo, 3mo, 6mo, 1y, ytd, 5y."""
    results = []
    for ticker in tickers:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            results.append({"ticker": ticker.upper(), "error": "No data found"})
            continue
        start, end = hist["Close"].iloc[0], hist["Close"].iloc[-1]
        results.append({
            "ticker": ticker.upper(),
            "return_pct": round((end - start) / start * 100, 2),
        })
    return sorted(
        results,
        key=lambda r: r.get("return_pct", float("-inf")),
        reverse=True,
    )



TOOLS = [get_quote, get_performance, get_fundamentals, compare_performance]

SYSTEM_PROMPT = """You are Market Buddy, a friendly capital markets assistant.
Use your tools to fetch real data before answering; never guess numbers.
Explain results in plain English and keep answers short.
You provide information and analysis only, not buy/sell recommendations."""


# ---------- Agent ----------

def build_agent(checkpointer=None):
    return create_agent(
        model="openai:gpt-5-mini",  
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


# Used later by `langgraph dev` / Agent Chat UI (the server provides its own memory).
agent = build_agent()


# ---------- Terminal chat ----------

if __name__ == "__main__":
    # In the terminal we provide memory ourselves.
    chat_agent = build_agent(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "session-2"}}

    print("Market Buddy ready! Ask about any stock (type 'quit' to exit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        result = chat_agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )
        print(f"\nMarket Buddy: {result['messages'][-1].content}\n")
