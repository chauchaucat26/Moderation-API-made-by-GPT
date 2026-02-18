import os
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "cl-tohoku/bert-base-japanese-v3-small"
THRESHOLD = 0.6
MAX_LENGTH = 128

torch.set_num_threads(1)

app = FastAPI()

class CheckRequest(BaseModel):
    text: str

start_time = time.time()
total_requests = 0

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2,
    torch_dtype=torch.float32
)

model.eval()
print("Model loaded.")

@app.post("/api/check")
def check(req: CheckRequest):
    global total_requests

    text = req.text.strip()
    if not text:
        return {"error": "no text"}

    total_requests += 1

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        score = torch.softmax(logits, dim=1)[0][1].item()

    return {
        "hate_score": round(score, 4),
        "is_hate": score >= THRESHOLD
    }

@app.get("/api/health")
def health():
    uptime = int(time.time() - start_time)
    return {
        "status": "ok",
        "totalRequests": total_requests,
        "uptime_seconds": uptime
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
