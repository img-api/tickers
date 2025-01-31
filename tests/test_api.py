
import pytest
import httpx

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session")
def client():
    """Creates an HTTPX test client."""
    with httpx.Client(base_url=BASE_URL) as client:
        yield client


import pytest


@pytest.mark.parametrize(
    "summaries", 
    [
        {
            "title": "Blockbuster Earnings",
            "summary": "Intel and Twitter report blockbuster earnings",
            "tickers": ["INTC", "TWTR"]
        }, {
            "title": "US Steel Reports Record Profits",
            "summary": "United States Steel Corporation reports record profits for the third quarter",
            "tickers": ["X", "USS"]
        }, {
            "title": "Apple Reports Record Earnings Amid Strong iPhone Sales",
            "summary": "Apple Inc. announced record quarterly earnings, driven by strong demand for iPhones.",
            "tickers": ["AAPL"]
        },
        {
            "title": "Tesla's Market Cap Surpasses $1 Trillion",
            "summary": "Tesla has officially joined the exclusive $1 trillion market cap club.",
            "tickers": ["TSLA"]
        },
        {
            "title": "JPMorgan Posts Strong Q1 Profits as Interest Rates Rise",
            "summary": "The largest U.S. bank reported a significant increase in profits due to higher interest income.",
            "tickers": ["JPM"]
        },

        {
            "title": "Rio Tinto Boosts Copper Production Amid Growing Demand",
            "summary": "The mining giant has increased its copper output to meet rising global demand.",
            "tickers": ["RIO"]
        }
    ]
    , ids=lambda article: article["title"]
)


def test_create_user(client, summaries):
    """Test creating a user."""
    response = client.post("/tickers", json=summaries)
    
    assert response.status_code == 200
    data = response.json()
    assert data.get("tickers") == summaries["tickers"]
