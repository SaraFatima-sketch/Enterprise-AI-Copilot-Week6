from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import logging

# Logging set karein
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="LLM MLOps & Production API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class RAGQuery(BaseModel):
    question: str
    context: str

@app.get("/")
@limiter.limit("5/minute")
def home(request: Request):
    logger.info("Home endpoint accessed")
    return {"status": "API active hai!", "version": "1.0.0"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/evaluate")
@limiter.limit("5/minute")
def evaluate_rag(request: Request, query: RAGQuery):
    logger.info(f"Evaluating query: {query.question}")
    
    # Mock RAGAS Evaluation metrics score
    return {
        "question": query.question,
        "context": query.context,
        "metrics": {
            "faithfulness": 0.92,
            "answer_relevance": 0.88,
            "passed": True
        }
    }