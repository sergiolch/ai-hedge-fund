import os
import pandas as pd
import requests

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
)

# Global cache instance
_cache = get_cache()

# Global valid ticker cache to avoid redundant checks
_ticker_validity_cache = {}

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or API."""
    # Check cache first
    if cached_data := _cache.get_prices(ticker):
        # Filter cached data by date range and convert to Price objects
        filtered_data = [Price(**price) for price in cached_data if start_date <= price["time"] <= end_date]
        if filtered_data:
            return filtered_data

    # Check if ticker is valid
    if not is_valid_ticker(ticker):
        return []

    # If not in cache or no data in range, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Warning: Error fetching price data for ticker '{ticker}': {response.status_code} - {response.text}")
            return []

        # Parse response with Pydantic model
        price_response = PriceResponse(**response.json())
        prices = price_response.prices

        if not prices:
            print(f"Warning: No price data found for ticker '{ticker}' in the specified date range")
            return []

        # Cache the results as dicts
        _cache.set_prices(ticker, [p.model_dump() for p in prices])
        return prices
    except Exception as e:
        print(f"Warning: Failed to fetch price data for ticker '{ticker}': {str(e)}")
        return []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or API."""
    # Check cache first
    if cached_data := _cache.get_financial_metrics(ticker):
        # Filter cached data by date and limit
        filtered_data = [FinancialMetrics(**metric) for metric in cached_data if metric["report_period"] <= end_date]
        filtered_data.sort(key=lambda x: x.report_period, reverse=True)
        if filtered_data:
            return filtered_data[:limit]

    # Check if ticker is valid
    if not is_valid_ticker(ticker):
        return []

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = f"https://api.financialdatasets.ai/financial-metrics/?ticker={ticker}&report_period_lte={end_date}&limit={limit}&period={period}"
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Warning: Error fetching financial metrics for ticker '{ticker}': {response.status_code} - {response.text}")
            return []

        # Parse response with Pydantic model
        metrics_response = FinancialMetricsResponse(**response.json())
        # Return the FinancialMetrics objects directly instead of converting to dict
        financial_metrics = metrics_response.financial_metrics
        
        # Set calendar_date to report_period if it's not provided
        for metric in financial_metrics:
            if metric.calendar_date is None:
                metric.calendar_date = metric.report_period

        if not financial_metrics:
            print(f"Warning: No financial metrics found for ticker '{ticker}' in the specified date range")
            return []

        # Cache the results as dicts
        _cache.set_financial_metrics(ticker, [m.model_dump() for m in financial_metrics])
        return financial_metrics
    except Exception as e:
        print(f"Warning: Failed to fetch financial metrics for ticker '{ticker}': {str(e)}")
        return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from API."""
    # Check if ticker is valid
    if not is_valid_ticker(ticker):
        return []
        
    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    try:
        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 200:
            print(f"Warning: Error fetching line items for ticker '{ticker}': {response.status_code} - {response.text}")
            return []

        # Parse response with Pydantic model
        line_item_response = LineItemResponse(**response.json())
        financial_line_items = line_item_response.line_items

        if not financial_line_items:
            print(f"Warning: No line items found for ticker '{ticker}' in the specified date range")
            return []

        return financial_line_items
    except Exception as e:
        print(f"Warning: Failed to fetch line items for ticker '{ticker}': {str(e)}")
        return []


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Check if ticker is valid
    if not is_valid_ticker(ticker):
        return []

    # Check cache first
    if cached_data := _cache.get_insider_trades(ticker):
        # Filter cached data by date range
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_trades = []
    current_end_date = end_date
    
    try:
        while True:
            url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
            if start_date:
                url += f"&filing_date_gte={start_date}"
            url += f"&limit={limit}"
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Warning: Error fetching insider trades for ticker '{ticker}': {response.status_code} - {response.text}")
                return all_trades
            
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
            
            if not insider_trades:
                break
                
            all_trades.extend(insider_trades)
            
            # Only continue pagination if we have a start_date and got a full page
            if not start_date or len(insider_trades) < limit:
                break
                
            # Update end_date to the oldest filing date from current batch for next iteration
            current_end_date = min(trade.filing_date for trade in insider_trades).split('T')[0]
        
        if not all_trades and start_date and end_date:
            print(f"Warning: No insider trades found for ticker '{ticker}' between {start_date} and {end_date}")
        elif not all_trades:
            print(f"Warning: No insider trades found for ticker '{ticker}'")
            
        # Cache the results
        if all_trades:
            _cache.set_insider_trades(ticker, [t.model_dump() for t in all_trades])
        
        return all_trades
    except Exception as e:
        print(f"Warning: Failed to fetch insider trades for ticker '{ticker}': {str(e)}")
        return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Check cache first
    if cached_data := _cache.get_company_news(ticker):
        # Filter cached data by date range
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # Check if ticker is valid
    if not is_valid_ticker(ticker):
        return []

    # If not in cache or insufficient data, fetch from API
    headers = {}
    if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
        headers["X-API-KEY"] = api_key

    all_news = []
    current_end_date = end_date
    
    try:
        while True:
            url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
            if start_date:
                url += f"&start_date={start_date}"
            url += f"&limit={limit}"
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Warning: Error fetching company news for ticker '{ticker}': {response.status_code} - {response.text}")
                return all_news
            
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
            
            if not company_news:
                break
                
            all_news.extend(company_news)
            
            # Only continue pagination if we have a start_date and got a full page
            if not start_date or len(company_news) < limit:
                break
                
            # Update end_date to the oldest news date from current batch for next iteration
            current_end_date = min(news.date for news in company_news)
            
            # If we've reached or passed the start_date, we can stop
            if current_end_date <= start_date:
                break

        if not all_news and start_date and end_date:
            print(f"Warning: No company news found for ticker '{ticker}' between {start_date} and {end_date}")
        elif not all_news:
            print(f"Warning: No company news found for ticker '{ticker}'")
            
        # Cache the results
        if all_news:
            _cache.set_company_news(ticker, [news.model_dump() for news in all_news])
            
        return all_news
    except Exception as e:
        print(f"Warning: Failed to fetch company news for ticker '{ticker}': {str(e)}")
        return []


def get_market_cap(
    ticker: str,
    end_date: str,
) -> float | None:
    """Fetch market cap from the API."""
    try:
        financial_metrics = get_financial_metrics(ticker, end_date)
        if not financial_metrics:
            print(f"Warning: No financial metrics available for ticker '{ticker}', cannot determine market cap")
            return None
            
        market_cap = financial_metrics[0].market_cap
        if not market_cap:
            print(f"Warning: Market cap data not available for ticker '{ticker}'")
            return None

        return market_cap
    except Exception as e:
        print(f"Warning: Failed to fetch market cap for ticker '{ticker}': {str(e)}")
        return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def is_valid_ticker(ticker: str) -> bool:
    """
    Check if a ticker is valid by attempting to get price data for it.
    Uses a cache to avoid redundant API calls for the same ticker.
    
    Args:
        ticker: The stock ticker symbol to check
        
    Returns:
        bool: True if the ticker is valid, False otherwise
    """
    # Check cache first
    if ticker in _ticker_validity_cache:
        return _ticker_validity_cache[ticker]
        
    try:
        import datetime
        # Get the current date and a date 5 days ago
        end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
        
        # Try to get price data with minimal date range
        url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={start_date}&end_date={end_date}"
        
        headers = {}
        if api_key := os.environ.get("FINANCIAL_DATASETS_API_KEY"):
            headers["X-API-KEY"] = api_key
            
        response = requests.get(url, headers=headers)
        
        # Check if we get a successful response with data
        is_valid = False
        if response.status_code == 200:
            data = response.json()
            if "prices" in data and len(data["prices"]) > 0:
                is_valid = True
        
        # Cache the result
        _ticker_validity_cache[ticker] = is_valid
        
        # Only print the warning message once per ticker
        if not is_valid:
            print(f"Warning: Invalid or unsupported ticker '{ticker}'. This API only supports company stocks, not ETFs or other instruments.")
            
        return is_valid
    except Exception:
        _ticker_validity_cache[ticker] = False
        print(f"Warning: Invalid or unsupported ticker '{ticker}'. This API only supports company stocks, not ETFs or other instruments.")
        return False


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    if not is_valid_ticker(ticker):
        print(f"Warning: Invalid ticker '{ticker}'")
        return pd.DataFrame()
    
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
