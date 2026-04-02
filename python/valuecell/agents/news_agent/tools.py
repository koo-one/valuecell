"""News-related tools for the News Agent."""

from datetime import datetime
from typing import Optional

from agno.agent import Agent
from loguru import logger

from valuecell.adapters.models import create_model


async def web_search(query: str) -> str:
    """Search web for the given query and return a summary of the top results.

    This function uses the OpenAI OAuth-backed model with built-in web search.

    Args:
        query: The search query string.

    Returns:
        A summary of the top search results.
    """
    model = create_model(
        provider="openai",
        model_id="gpt-5.4",
        max_tokens=4096,
    )
    response = await Agent(
        model=model,
        tools=[{"type": "web_search_preview"}],
    ).arun(
        "Use web search to gather current information and summarize the most relevant findings.\n\n"
        f"Query: {query}"
    )
    return response.content


async def get_breaking_news() -> str:
    """Get breaking news and urgent updates.

    Returns:
        Formatted string containing breaking news
    """
    try:
        search_query = "breaking news urgent updates today"
        logger.info("Fetching breaking news")

        news_content = await web_search(search_query)
        return news_content

    except Exception as e:
        logger.error(f"Error fetching breaking news: {e}")
        return f"Error fetching breaking news: {str(e)}"


async def get_financial_news(
    ticker: Optional[str] = None, sector: Optional[str] = None
) -> str:
    """Get financial and market news.

    Args:
        ticker: Stock ticker symbol for company-specific news
        sector: Industry sector for sector-specific news

    Returns:
        Formatted string containing financial news
    """
    try:
        search_query = "financial market news"

        if ticker:
            search_query = f"{ticker} stock news financial market"
        elif sector:
            search_query = f"{sector} sector financial news market"

        # Add time constraint for recent news
        today = datetime.now().strftime("%Y-%m-%d")
        search_query += f" {today}"

        logger.info(f"Searching for financial news with query: {search_query}")

        news_content = await web_search(search_query)
        return news_content

    except Exception as e:
        logger.error(f"Error fetching financial news: {e}")
        return f"Error fetching financial news: {str(e)}"
