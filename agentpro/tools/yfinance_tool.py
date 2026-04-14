from .base_tool import Tool
import yfinance as yf
from typing import Any
import json

class YFinanceTool(Tool):
    name: str = "Yahoo Finance Data Fetcher"
    description: str = "Fetches stock/company info from Yahoo Finance with 'basic' or 'extended' detail level."
    action_type: str = "fetch_stock_info"
    input_format: str = (
        "A JSON with 'ticker' and optional 'detail_level' ('basic' or 'extended').\n"
        "Example: {\"ticker\": \"AAPL\", \"detail_level\": \"extended\"}"
    )

    def run(self, input_text: Any) -> str:
        if isinstance(input_text, str):
            try:
                input_text = json.loads(input_text)
            except json.JSONDecodeError:
                return "❌ Error: Expected JSON input like {\"ticker\": \"AAPL\"}."

        if not isinstance(input_text, dict) or "ticker" not in input_text:
            return "❌ Error: Missing 'ticker' field in input."

        ticker_symbol = input_text["ticker"].strip().upper()
        detail_level = input_text.get("detail_level", "basic").lower()

        stock = yf.Ticker(ticker_symbol)
        info = stock.info

        if not info:
            return f"No data found for ticker: {ticker_symbol}"

        company_name = info.get('shortName', 'N/A')
        current_price = info.get('currentPrice', 'N/A')
        market_cap = info.get('marketCap', 'N/A')

        output = (f"Company: {company_name}\n"
                  f"Ticker: {ticker_symbol}\n"
                  f"Current Price: {current_price}\n"
                  f"Market Cap: {market_cap}")

        if detail_level == "extended":
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            dividend_yield = info.get('dividendYield', 'N/A')

            output += (f"\nSector: {sector}"
                       f"\nIndustry: {industry}"
                       f"\nDividend Yield: {dividend_yield}")

            # Add last 5 days closing prices
            history = stock.history(period="5d")
            if not history.empty:
                output += "\nLast 5 Days Closing Prices:"
                for date, row in history.iterrows():
                    output += f"\n- {date.strftime('%Y-%m-%d')}: {row['Close']:.2f}"

        return output
