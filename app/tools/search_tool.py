import requests
import re
import urllib.parse
from app.config import settings

class SearchTool:
    """
    Web Search Tool.
    Leverages Tavily Search API if configured, otherwise falls back to a free,
    API-keyless HTML search scraper via DuckDuckGo, guaranteeing search capability in any environment.
    """

    def __init__(self):
        self.tavily_key = settings.TAVILY_API_KEY

    def search(self, query: str, max_results: int = 4) -> str:
        """Runs the search query and returns structured results."""
        if self.tavily_key and self.tavily_key.strip():
            return self._tavily_search(query, max_results)
        else:
            return self._ddg_fallback_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> str:
        """Executes search query via Tavily Search API."""
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic"
            }
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                results = response.json().get("results", [])
                if not results:
                    return f"No Tavily search results found for: '{query}'"
                
                formatted = []
                for idx, r in enumerate(results):
                    formatted.append(
                        f"[{idx + 1}] Title: {r.get('title')}\n"
                        f"    URL: {r.get('url')}\n"
                        f"    Snippet: {r.get('content')}\n"
                    )
                return "\n".join(formatted)
        except Exception as e:
            # Fallback to DDG if Tavily request fails
            pass
            
        return self._ddg_fallback_search(query, max_results)

    def _ddg_fallback_search(self, query: str, max_results: int) -> str:
        """Free, API-keyless HTML scraper query to DuckDuckGo."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        try:
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code != 200:
                return f"⚠️ Search Error: Could not reach web search servers. (HTTP {response.status_code})"

            # Use regex to extract search result blocks from simple DDG HTML
            html = response.text
            # Results are contained inside <div class="result result-default...">
            result_blocks = re.findall(r'<div class="result result-default.*?">(.*?)</div>\s*</div>', html, re.DOTALL)
            
            if not result_blocks:
                return f"No search results returned for query: '{query}'"

            formatted = []
            count = 0
            for block in result_blocks:
                if count >= max_results:
                    break
                
                # Extract URL and Title
                url_title_match = re.search(r'<a class="result__url" href="(.*?)".*?>(.*?)</a>', block, re.DOTALL)
                snippet_match = re.search(r'<a class="result__snippet".*?>(.*?)</a>', block, re.DOTALL)
                
                if url_title_match:
                    # Clean up URL (sometimes DDG wraps links in redirect urls)
                    raw_url = url_title_match.group(1).strip()
                    url = raw_url
                    if "/l/?" in raw_url:
                        # Extract target url
                        parsed = urllib.parse.urlparse(raw_url)
                        query_params = urllib.parse.parse_qs(parsed.query)
                        if "uddg" in query_params:
                            url = query_params["uddg"][0]

                    # Strip HTML tags from title
                    title = re.sub(r'<[^>]*>', '', url_title_match.group(2)).strip()
                    
                    snippet = ""
                    if snippet_match:
                        snippet = re.sub(r'<[^>]*>', '', snippet_match.group(1)).strip()

                    formatted.append(
                        f"[{count + 1}] Title: {title}\n"
                        f"    URL: {url}\n"
                        f"    Snippet: {snippet}\n"
                    )
                    count += 1

            if not formatted:
                return f"No query result listings parsed for: '{query}'"
                
            return "\n".join(formatted)

        except Exception as e:
            return f"❌ Web search failed: {str(e)}"
