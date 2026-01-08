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
    temperature=0.8
)


parser = JsonOutputParser()


template = PromptTemplate(
    template = """

    SYSTEM ROLE:
You are a deterministic program-verification and answer-evaluation engine.
You judge semantic correctness under standard competitive-programming rules.
You do NOT perform string matching.

TASK:
Determine whether the User Answer is logically and functionally equivalent
to the Expected Answer / Concept as accepted by competitive programming judges
(LeetCode, CodeChef, Codeforces, AtCoder).

INPUTS:
- Question:
  {question}

- Expected Answer / Concept:
  {expected}

- User Answer:
  {answer}

EVALUATION PRINCIPLES (MANDATORY):

GENERAL:
- NEVER compare answers by raw string equality.
- NEVER reject due to:
  - Newlines (`\\n`)
  - Indentation
  - Whitespace differences
  - Line breaks
  - Formatting style
  - Braces placement
  - One-line vs multi-line formatting
- Treat formatting differences as non-semantic.

CODE-SPECIFIC RULES:
1. Parse the User Answer as code in its respective language.
2. Normalize the code mentally:
   - Ignore whitespace, indentation, and line breaks.
   - Focus on control flow, operations, and returned values.
3. Determine the implemented logic.
4. Compare the logic with the Expected Answer’s logic.
5. Accept if:
   - The algorithm computes the same result.
   - The behavior matches for all valid inputs implied by the problem.
6. Reject ONLY if:
   - The logic differs.
   - A required condition is missing.
   - The output differs for some valid input.
   - There is a clear logical or semantic error.

NON-CODE ANSWERS:
- Judge by conceptual correctness only.
- Ignore phrasing, wording, or ordering differences.

MULTIPLE VALID SOLUTIONS:
- Accept any solution that is logically correct and consistent with the expected concept.
- Do NOT require structural similarity.

STRICT PROHIBITIONS:
- Do NOT perform string matching.
- Do NOT require identical formatting.
- Do NOT assume incorrectness due to presentation.
- Do NOT reject due to harmless syntactic variation.

DECISION RULE:
- verdict = 1 → Semantically and logically correct.
- verdict = 0 → Semantically incorrect or missing required logic.

DEFAULT RULE (IMPORTANT):
If the User Answer implements the same logic as the Expected Answer
and no explicit contradiction or logical error exists,
YOU MUST return verdict = 1.

OUTPUT CONSTRAINTS:
- Output ONLY valid JSON.
- No explanations.
- No comments.
- No extra text.

OUTPUT FORMAT:
{{"verdict": 0}}
or
{{"verdict": 1}}

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

@app.get("/")
async def hello():
    return "API is Working"

#Run this command
# uvicorn app:app --host 0.0.0.0 --port 8000

# chain.get_graph().print_ascii()