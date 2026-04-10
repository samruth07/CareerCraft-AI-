"""
CareerCraft AI - Web Search Tool
A utility for agents to search the live web for up-to-date information.
"""

from langchain_tavily import TavilySearch
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

def perform_web_search(query: str, max_results: int = 3) -> str:
    """
    Perform a live web search using Tavily.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return
        
    Returns:
        str: Formatted search results with URLs and page snippets
    """
    if not settings.tavily_api_key:
        logger.warning("Web search skipped: TAVILY_API_KEY is not configured.")
        return "Web search is disabled. Rely on local knowledge base."
        
    try:
        # Initialize the Tavily search tool
        # In langchain-tavily, the tool is called TavilySearch
        search_tool = TavilySearch(max_results=max_results)
        
        # Execute the search
        logger.info(f"Searching web for: {query}")
        results = search_tool.invoke({"query": query})
        
        # Format the results into a string to inject into the LLM context
        if not results:
            return "No live web search results found."
            
        formatted_results = "LIVE WEB SEARCH RESULTS:\n"
        for i, res in enumerate(results):
            # Fallbacks for empty dictionary keys just in case
            title = res.get('title', 'Unknown Title') if isinstance(res, dict) else 'Result'
            url = res.get('url', 'Unknown URL') if isinstance(res, dict) else 'URL'
            content = res.get('content', '') if isinstance(res, dict) else str(res)
            
            formatted_results += f"[{i+1}] Title: {title}\n"
            formatted_results += f"    URL: {url}\n"
            formatted_results += f"    Content: {content}...\n\n"
            
        return formatted_results
        
    except Exception as e:
        logger.error(f"Web search failed: {str(e)}")
        return f"Web search failed: {str(e)}"
