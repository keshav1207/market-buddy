"""
Market Buddy - a simple stock Q&A agent built with LangChain.

"""

import yfinance as yf
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import AgentMiddleware, dynamic_prompt, ModelRequest
from dataclasses import dataclass
import base64
import mimetypes
from pathlib import Path

load_dotenv()



@dataclass
class UserProfile:
    name: str = "there"
    risk_profile: str = "balanced"     # conservative | balanced | aggressive
    base_currency: str = "USD"


def _fx_rate(to_currency: str) -> float:
    """USD -> to_currency rate. Returns 1.0 for USD or on failure."""
    code = to_currency.upper()
    if code == "USD":
        return 1.0
    try:
        return float(yf.Ticker(f"{code}=X").fast_info.last_price)
    except Exception:
        return 1.0

# ---------- Tools ----------
# The model only sees each tool's name, docstring, and argument types.



@tool
def get_quote(ticker: str, runtime: ToolRuntime[UserProfile]) -> dict:
    """Get the latest price, market cap, and 52-week range
    for a stock ticker (e.g. AAPL)."""
    profile = runtime.context or UserProfile()
    info = yf.Ticker(ticker).fast_info
    rate = _fx_rate(profile.base_currency) if info.currency == "USD" else 1.0

    return {
        "ticker": ticker.upper(),
        "price": round(info.last_price * rate, 2),
       "market_cap_usd": info.market_cap,
        "52w_high": round(info.year_high * rate, 2),
        "52w_low": round(info.year_low * rate, 2),
        "currency": profile.base_currency if rate != 1.0 else info.currency,
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


TONE = {
    "conservative": (
        "This user is risk-averse. Lead with downside: volatility, drawdowns, "
        "and dividend stability. Flag when something is speculative."
    ),
    "balanced": (
        "This user wants a balanced view. Mention both upside and risk."
    ),
    "aggressive": (
        "This user has a high risk tolerance. Focus on growth, momentum, "
        "and relative performance. Keep risk notes brief."
    ),
}

BASE_PROMPT = """You are Market Buddy, a friendly capital markets assistant.
Use your tools to fetch real data before answering; never guess numbers.
Explain results in plain English and keep answers short.

When the user sends a chart image:
- Say what you can actually see: ticker, timeframe, overall direction,
  notable spikes or drops, and any visible axis values.
- Never state precise prices or percentages read off the image.
- If you can identify the ticker and timeframe, call get_performance
  to confirm the real numbers, and give those instead.
- Say so plainly if the image is unreadable or isn't a price chart.
- If the image and the tool data disagree, say so explicitly and
  trust the tool data.

You provide information and analysis only, not buy/sell recommendations."""

@dynamic_prompt
def personalized_prompt(request: ModelRequest) -> str:
    profile = request.runtime.context or UserProfile()
    return f"""{BASE_PROMPT}

You are speaking with {profile.name}.
{TONE.get(profile.risk_profile, TONE["balanced"])}
Their base currency is {profile.base_currency}. Prices from tools are
already converted to it, so use that currency in your answers."""

def _fix_block(block):
    """Convert LangChain/UI-style image blocks into OpenAI image_url blocks."""
    if not isinstance(block, dict) or block.get("type") != "image":
        return block
    data = block.get("data") or block.get("base64")
    mime = block.get("mime_type") or block.get("mimeType") or "image/png"
    url = f"data:{mime};base64,{data}" if data else block.get("url")
    if not url:
        return block
    return {"type": "image_url", "image_url": {"url": url}}


def _fix_messages(messages):
    fixed = []
    for msg in messages:
        if isinstance(msg.content, list):
            msg = msg.model_copy(
                update={"content": [_fix_block(b) for b in msg.content]}
            )
        fixed.append(msg)
    return fixed

def build_message(text: str, image_path: str | None = None) -> dict:
    """Build a user message, optionally carrying an image."""
    if not image_path:
        return {"role": "user", "content": text}

    path = Path(image_path).expanduser()
    data = base64.b64encode(path.read_bytes()).decode()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"

    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            },
        ],
    }
class OpenAIImageBlocks(AgentMiddleware):
    """Rewrite image blocks on every model call (sync and async)."""

    def wrap_model_call(self, request, handler):
        return handler(request.override(messages=_fix_messages(request.messages)))

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(messages=_fix_messages(request.messages)))

# ---------- Agent ----------

def build_agent(checkpointer=None):
    return create_agent(
        model="openai:gpt-5-mini",  
        tools=TOOLS,
        middleware=[OpenAIImageBlocks(), personalized_prompt],
        checkpointer=checkpointer,
        context_schema=UserProfile,
    )


# Used later by `langgraph dev` / Agent Chat UI (the server provides its own memory).
agent = build_agent()

# ---------- Terminal chat ----------

if __name__ == "__main__":
    # In the terminal we provide memory ourselves.
    chat_agent = build_agent(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "session-2"}}

    print("Market Buddy ready! Ask about any stock (type 'quit' to exit).\n")
    profile = UserProfile(name="Keshav", risk_profile="aggressive", base_currency="CAD" \
    "")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"quit", "exit"}:
            break

        # Type: image:/path/to/chart.png What trend is this?
        image_path = None
        if question.startswith("image:"):
            _, rest = question.split(":", 1)
            image_path, question = rest.strip().split(" ", 1)

        result = chat_agent.invoke(
            {"messages": [build_message(question, image_path)]},
            config=config,
            context=profile,
        )
        print(f"\nMarket Buddy: {result['messages'][-1].content}\n")
