import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph.types import Command
from graph import app as graph_app , actual_route     # renamed: FastAPI also uses "app"
import os

api = FastAPI(title="MedInfo Agent")

class AskRequest(BaseModel):
    question: str
    thread_id: str | None = None        # optional; a new one is created if missing

class ReviewRequest(BaseModel):
    approved: bool
    text: str | None = None             # optional edited answer


@api.post("/ask")
def ask(req: AskRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"app_version": os.getenv("APP_VERSION", "local")},
    }       
    result = graph_app.invoke({"question": req.question}, config)

    if "__interrupt__" in result:
        return {"status": "pending_review", "thread_id": thread_id}
    return {
        "status": "answered",
        "route": actual_route(result),     # "answer", "emergency", "fallback", "not_covered"
        "thread_id": thread_id,
        "reply": result["reply"],
    }

@api.post("/review/{thread_id}")
def review(thread_id:str, req: ReviewRequest):
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"app_version": os.getenv("APP_VERSION", "local")},
    }    
    result = graph_app.invoke(Command(resume=req.model_dump()), config)
    return {"status": "answered", "thread_id": thread_id, "reply": result["reply"]}


@api.get("/health")
def health():
    return {"status": "ok"}