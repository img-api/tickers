"""Main server for API entrypoint"""
import os
import json
import uvicorn
from fastapi import FastAPI
from google.oauth2 import service_account

from tickers import Ticker, load_news


news = load_news()
ticker_extractor = Ticker(news=news, recreate_table=False)

app = FastAPI()


credentials = json.load(open("/home/dev/vertex-ai-config.json"))
google_credentials = credentials["GOOGLE_APPLICATION_CREDENTIALS"]


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_credentials


with open(google_credentials, 'r') as source:
    info = json.load(source)


# Auth using service account with json credentials
service_account.Credentials.from_service_account_info(info)


@app.get("/")
async def root():
    return ticker_extractor.get_tickers()


@app.post("/tickers")
async def root(title: str = "", summary: str = ""):
    return ticker_extractor.get_tickers(title, summary)


def start():
    """Launched with `poetry run start` at root level"""
    uvicorn.run("my_package.main:app", host="0.0.0.0", port=5521, reload=True)
