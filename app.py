import os
import time
import torch
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

MODEL = "cl-tohoku/bert-base-japanese"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2
)

start_time = time.time()
total_requests = 0

@app.post("/api/check")
def check(data: dict):
    global total_requests
    text = data.get("text", "")
    if not text:
        return {"error": "no text"}

    total_requests += 1

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        score = torch.softmax(logits, dim=1)[0][1].item()

    return {
        "text": text,
        "hate_score": score,
        "is_hate": score >= 0.6
    }

@app.get("/api/health")
def health():
    uptime = int(time.time() - start_time)
    return {
        "status": "ok",
        "totalRequests": total_requests,
        "uptime_seconds": uptime
    }

# 👇 Render対応
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))  # RenderはPORTを渡す
    uvicorn.run("main:app", host="0.0.0.0", port=port)
