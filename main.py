# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# -----------------------------
# ENV
# -----------------------------
load_dotenv()

# -----------------------------
# APP
# -----------------------------
app = FastAPI(title="LLM Answer Verifier")

# -----------------------------
# MODEL
# -----------------------------
model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# -----------------------------
# PARSER
# -----------------------------
parser = JsonOutputParser()

# -----------------------------
# PROMPT
# -----------------------------
template = PromptTemplate(
    template="""
You are a strict answer verification engine.

Question:
{question}

Expected Answer / Concept:
{expected}

User Answer:
{answer}

{format_instruction}

Rules:
- verdict = 1 only if answer is correct or acceptable
- verdict = 0 otherwise
- Output ONLY valid JSON
""",
    input_variables=["question", "expected", "answer"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | model | parser

# -----------------------------
# SCHEMA
# -----------------------------
class VerifyRequest(BaseModel):
    question: str
    expected: str
    answer: str

class VerifyResponse(BaseModel):
    verdict: int

# -----------------------------
# ROUTE
# -----------------------------
@app.post("/verify", response_model=VerifyResponse)
async def verify_answer(payload: VerifyRequest):
    result = await chain.ainvoke({
        "question": payload.question,
        "expected": payload.expected,
        "answer": payload.answer
    })

    return {"verdict": int(result["verdict"])}

# -----------------------------
# RUN
# -----------------------------
# uvicorn app:app --host 0.0.0.0 --port 8000