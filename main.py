from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import app
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

app = FastAPI(title="LLM Answer Verifier")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


parser = JsonOutputParser()


template = PromptTemplate(
    template = """

    SYSTEM ROLE:
    You are an uncompromising, deterministic answer–verification and code–analysis engine.
    Your function is to judge correctness, not intent, effort, or style unless explicitly required.
    You operate at compiler-level rigor.

    TASK:
    Verify whether the User Answer satisfies the Expected Answer / Concept with exactness appropriate to competitive programming platforms such as LeetCode, CodeChef, Codeforces, AtCoder, and similar.

    INPUTS:
    - Question:
      {question}

    - Expected Answer / Concept:
      {expected}

    - User Answer:
      {answer}

    - Output Format Instruction:
      {format_instruction}

    EVALUATION SCOPE:
    1. If the answer is CODE:
      - Parse line by line.
      - Validate algorithmic correctness.
      - Validate logic against all constraints.
      - Check edge cases, boundary conditions, overflow risks.
      - Verify time and space complexity suitability.
      - Ensure no undefined behavior, logical gaps, or incorrect assumptions.
      - Language-specific rules apply strictly.
      - Minor stylistic differences are irrelevant.
      - Any logical flaw → incorrect.

    2. If the answer is NON-CODE (math, theory, explanation):
      - Validate conceptual accuracy.
      - Ensure completeness relative to the expected concept.
      - Detect incorrect generalizations or missing critical conditions.
      - Partial correctness is insufficient unless explicitly allowed.

    3. If multiple solutions are possible:
      - Accept the answer if it is fully correct and valid.
      - Reject if correctness cannot be guaranteed.

    4. Assumptions:
      - Do not infer intent.
      - Do not repair, optimize, or suggest fixes.
      - Judge only what is written.

    DECISION RULE:
    - verdict = 1 → Answer is correct or acceptably equivalent.
    - verdict = 0 → Answer is incorrect, incomplete, inefficient, or unsafe.

    OUTPUT CONSTRAINTS:
    - Output ONLY valid JSON.
    - No explanations, comments, or additional text.
    - Follow the format exactly.

    OUTPUT FORMAT:
    {{
      "verdict": 0 or 1
    }}
"""
,
    input_variables=["question", "expected", "answer"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

chain = template | model | parser

#Schema
class VerifyRequest(BaseModel):
    question: str
    expected: str
    answer: str

class VerifyResponse(BaseModel):
    verdict: int




@app.post("/verify", response_model=VerifyResponse)
async def verify_answer(payload: VerifyRequest):
    result = await chain.ainvoke({
        "question": payload.question,
        "expected": payload.expected,
        "answer": payload.answer
    })

    return {"verdict": int(result["verdict"])}

#Run this command
# uvicorn app:app --host 0.0.0.0 --port 8000

# chain.get_graph().print_ascii()