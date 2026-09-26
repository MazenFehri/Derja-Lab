from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from features.sentiment import predict as sentiment
from features.translit.predict import translate

WEB = Path(__file__).parent / "web"

app = FastAPI()
app.mount("/static", StaticFiles(directory=WEB), name="static")


class TranslateRequest(BaseModel):
    text: str = Field(max_length=200)  # same cap as the training data


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    model: str


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/translate")
def translate_page():
    return FileResponse(WEB / "translate.html")


@app.get("/sentiment")
def sentiment_page():
    return FileResponse(WEB / "sentiment.html")


@app.post("/api/translate")
def api_translate(req: TranslateRequest):
    return {"result": translate(req.text)}


@app.get("/api/sentiment/models")
def api_sentiment_models():
    return sentiment.available()


@app.post("/api/sentiment")
def api_sentiment(req: SentimentRequest):
    if req.model not in sentiment.MODELS:
        raise HTTPException(404, f"Unknown model '{req.model}'")
    return sentiment.predict(req.text, req.model)


if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))  # hosts like Render set PORT
