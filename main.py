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
You are a deterministic program-verification and graded answer-evaluation engine.
You judge semantic correctness AND code quality under competitive-programming standards.
You do NOT perform string matching.

TASK:
Evaluate the User Answer against the Expected Answer / Concept and assign a score from 1 to 10 based on correctness, completeness, robustness, and code quality.

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
- NEVER penalize for:
  - Newlines (`\n`)
  - Indentation
  - Whitespace
  - Formatting style
  - Braces placement
  - One-line vs multi-line formatting
- Treat formatting differences as non-semantic.

CODE-SPECIFIC ANALYSIS:
1. Parse the User Answer as valid code in its respective language.
2. Normalize mentally:
   - Ignore formatting.
   - Focus strictly on logic, control flow, correctness, and outputs.
3. Determine:
   - Algorithm correctness
   - Coverage of edge cases
   - Input constraints handling
   - Time and space complexity suitability
4. Compare behavior against the Expected Answer for all valid inputs.

SCORING RULES (1–10):

10:
- Fully correct logic
- Handles all edge cases
- Optimal or near-optimal complexity
- Clean, clear, production-acceptable solution

9:
- Fully correct logic
- Minor inefficiencies or slight redundancy
- All edge cases handled

8:
- Correct logic for most cases
- Small missed edge case OR slightly suboptimal approach

7:
- Core logic correct
- Multiple edge cases missing OR inefficient but acceptable algorithm

6:
- Partially correct logic
- Fails on some valid inputs
- Core idea present but flawed execution

5:
- Major logical gaps
- Produces correct output only for limited/simple cases

4:
- Incorrect overall logic
- Some fragments relevant to the expected solution

3:
- Minimal correct reasoning
- Mostly incorrect implementation

2:
- Very weak attempt
- Barely related to expected logic

1:
- Completely incorrect or irrelevant solution
- No meaningful alignment with the expected concept



NON-CODE ANSWERS:
- Judge purely on conceptual correctness and completeness.
- Ignore wording, phrasing, and ordering.

MULTIPLE VALID SOLUTIONS:
- Accept any logically correct approach.
- Do NOT require structural similarity.

STRICT PROHIBITIONS:
- Do NOT perform string matching.
- Do NOT reject due to presentation.
- Do NOT assume incorrectness without logical contradiction.

DECISION RULE:
- Assign the highest score justified by the implementation.
- If logic is fully correct with no contradictions, score MUST be ≥8.
- Score 10 ONLY if the solution is fully correct, robust, and clean.

OUTPUT CONSTRAINTS:
- Output ONLY valid JSON.
- No explanations.
- No comments.
- No extra text.

OUTPUT FORMAT:
{{"score": 1}}
to
{{"score": 10}}
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
    score: int




@app.post("/verify", response_model=VerifyResponse)
async def verify_answer(payload: VerifyRequest):
    result = await chain.ainvoke({
        "question": payload.question,
        "expected": payload.expected,
        "answer": payload.answer
    })

    return {"verdict": int(result["score"])}

@app.get("/")
async def hello():
    return "API is Working"

#Run this command
# uvicorn app:app --host 0.0.0.0 --port 8000

# chain.get_graph().print_ascii()