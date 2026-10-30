from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.rag import ask_question

#***********************
# FastAPI Application  *
#***********************

app = FastAPI(
    title="AI/ML Research Navigator API",
    description=(
        "A Retrieval-Augmented Generation (RAG) API for "
        "answering questions from computer vision research papers."
    ),
    version="1.0.0",
)

#******************
# Request Model   *
#******************

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question about the research papers.",
    )

#*******************
# Response Model   *
#*******************

class Source(BaseModel):
    source: str
    page: int | str

class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]

#******************
# Root Endpoint   *
#******************

@app.get("/")
def root():
    return {
        "application": "AI/ML Research Navigator",
        "status": "running",
        "version": "1.0.0",
    }

#********************
# Health Endpoint   *
#********************

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "research-rag-api",
    }

#*****************
# Ask Endpoint   *
#*****************

@app.post(
    "/ask",
    response_model=QuestionResponse,
)
def ask(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        answer, documents = ask_question(question)

        sources = []

        for document in documents:

            source = document.metadata.get(
                "source",
                "Unknown",
            )

            page = document.metadata.get(
                "page",
                "Unknown",
            )

            sources.append(
                Source(
                    source=source,
                    page=page,
                )
            )

        return QuestionResponse(
            answer=answer,
            sources=sources,
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}",
        )