from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List

import httpx
from pydantic import ValidationError

from app.core.schemas import AgentState, PlanModel, PlanStep

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://ollama:11434"
OLLAMA_MODEL = "llama3"
REQUEST_TIMEOUT = 120.0

_MOCK_CORPUS: List[Dict[str, Any]] = [
    {"id": "doc_001", "title": "Introduction to Large Language Models",
     "text": "Large language models (LLMs) are neural networks trained on vast text corpora. They demonstrate emergent capabilities such as reasoning, summarisation, and code generation. GPT-4, Claude, and Llama are prominent examples.",
     "source": "mock://corpus/llm-intro"},
    {"id": "doc_002", "title": "Retrieval-Augmented Generation (RAG)",
     "text": "RAG combines a retriever that fetches relevant documents with a generator that conditions its output on those documents. This grounds responses in verifiable external knowledge and reduces hallucination.",
     "source": "mock://corpus/rag-overview"},
    {"id": "doc_003", "title": "LangGraph Multi-Agent Orchestration",
     "text": "LangGraph extends LangChain with a stateful graph abstraction suitable for multi-agent workflows. Nodes represent agents; edges encode control flow. Conditional edges allow dynamic routing based on runtime state.",
     "source": "mock://corpus/langgraph"},
    {"id": "doc_004", "title": "Prompt Engineering Best Practices",
     "text": "Effective prompts specify the task, audience, output format, and constraints. Chain-of-thought encourages step-by-step reasoning. Few-shot examples anchor the model to the desired response structure.",
     "source": "mock://corpus/prompt-eng"},
    {"id": "doc_005", "title": "Vector Databases for Semantic Search",
     "text": "Vector databases such as Qdrant, Weaviate, and Pinecone store dense embeddings and support approximate nearest-neighbour search. They complement BM25 keyword retrieval in hybrid search pipelines.",
     "source": "mock://corpus/vector-db"},
    {"id": "doc_006", "title": "Evaluation Metrics for NLP Systems",
     "text": "Common NLP evaluation metrics include BLEU, ROUGE, BERTScore, and RAGAS. Human evaluation remains the gold standard for open-ended generation quality.",
     "source": "mock://corpus/nlp-eval"},
    {"id": "doc_007", "title": "FastAPI Asynchronous Web Services",
     "text": "FastAPI is a modern Python web framework based on Starlette and Pydantic. It supports async/await natively, automatic OpenAPI documentation, and Server-Sent Events for real-time streaming.",
     "source": "mock://corpus/fastapi"},
    {"id": "doc_008", "title": "Agentic AI: Planning and Tool Use",
     "text": "Agentic AI systems decompose high-level goals into sub-tasks, select tools, execute actions, and observe results in a feedback loop. ReAct, Toolformer, and function-calling APIs are representative architectures.",
     "source": "mock://corpus/agentic-ai"},
]


async def _ollama_generate(prompt: str, *, json_mode: bool = False, system: str | None = None) -> str:
    payload: Dict[str, Any] = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    if json_mode:
        payload["format"] = "json"
    if system:
        payload["system"] = system
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        return response.json().get("response", "")


def _tokenise(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_score(query_tokens, doc_tokens, avg_dl, df, N, k1=1.5, b=0.75) -> float:
    tf_map: Dict[str, int] = defaultdict(int)
    for t in doc_tokens:
        tf_map[t] += 1
    dl = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in tf_map:
            continue
        tf = tf_map[term]
        idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))
    return score


def _mock_bm25_search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    tokenised_corpus = [_tokenise(doc["text"] + " " + doc["title"]) for doc in _MOCK_CORPUS]
    N = len(_MOCK_CORPUS)
    avg_dl = sum(len(t) for t in tokenised_corpus) / N
    df: Dict[str, int] = defaultdict(int)
    for tokens in tokenised_corpus:
        for term in set(tokens):
            df[term] += 1
    query_tokens = _tokenise(query)
    scored = [(_bm25_score(query_tokens, doc_tokens, avg_dl, df, N), doc)
              for doc_tokens, doc in zip(tokenised_corpus, _MOCK_CORPUS)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def _build_healing_prompt(original_prompt: str, bad_output: str, schema_hint: str) -> str:
    return (f"{original_prompt}\n\n--- PREVIOUS ATTEMPT PRODUCED INVALID JSON ---\n"
            f"{bad_output}\n\n--- REQUIRED JSON SCHEMA ---\n{schema_hint}\n\n"
            "Your response MUST be valid JSON matching the schema exactly. Output raw JSON only.")


async def planner_node(state: AgentState) -> AgentState:
    logger.info("[planner] Starting for query: %s", state.query)
    schema_hint = PlanModel.model_json_schema()
    system_prompt = ("You are a meticulous research planner. Decompose the research question "
                     "into 3-5 focused sub-queries. Respond ONLY with JSON matching the schema.")
    base_prompt = (f"Research question: {state.query}\n\n"
                   f"JSON schema:\n{json.dumps(schema_hint, indent=2)}\n\nProduce the JSON plan now:")
    prompt = base_prompt
    raw_output = ""
    plan: PlanModel | None = None
    for attempt in range(4):
        try:
            raw_output = await _ollama_generate(prompt, json_mode=True, system=system_prompt)
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.MULTILINE)
            plan = PlanModel.model_validate_json(cleaned)
            break
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            state.retry_count += 1
            logger.warning("[planner] Parse failure attempt %d: %s", attempt + 1, exc)
            if attempt == 3:
                plan = PlanModel(steps=[PlanStep(index=1, sub_query=state.query)])
                break
            prompt = _build_healing_prompt(base_prompt, raw_output, json.dumps(schema_hint, indent=2))
    state.plan = plan.steps
    state.steps.append(f"[Planner] Decomposed query into {len(plan.steps)} sub-queries.")
    state.next_node = "retriever"
    return state


async def retriever_node(state: AgentState) -> AgentState:
    logger.info("[retriever] Fetching documents for %d sub-queries.", len(state.plan))
    seen_ids = {doc["id"] for doc in state.retrieved_documents}
    new_docs: List[Dict[str, Any]] = []
    queries = [step.sub_query for step in state.plan] if state.plan else [state.query]
    for sub_query in queries:
        for doc in _mock_bm25_search(sub_query, top_k=3):
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                new_docs.append(doc)
    state.retrieved_documents.extend(new_docs)
    state.steps.append(f"[Retriever] Retrieved {len(new_docs)} new documents (total: {len(state.retrieved_documents)}).")
    state.next_node = "summariser"
    return state


async def summariser_node(state: AgentState) -> AgentState:
    logger.info("[summariser] Synthesising %d documents.", len(state.retrieved_documents))
    doc_block = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['text']}" for d in state.retrieved_documents)
    feedback_section = f"\n\nPrevious critic feedback:\n{state.critic_feedback}\n" if state.critic_feedback else ""
    prompt = (f"You are a professional research analyst. Using ONLY the documents below, write a "
              f"well-structured markdown report answering:\n\n**{state.query}**\n\n"
              f"Documents:\n{doc_block}{feedback_section}\n\n"
              "Use markdown headers, cite document IDs inline like [doc_001], include intro and conclusion.")
    raw_output = await _ollama_generate(prompt)
    state.current_summary = raw_output.strip()
    state.steps.append("[Summariser] Draft summary produced.")
    state.next_node = "critic"
    return state


async def critic_node(state: AgentState) -> AgentState:
    logger.info("[critic] Evaluating summary quality.")
    prompt = (f"You are a rigorous research critic. Evaluate this summary against the original question.\n\n"
              f"Original question: {state.query}\n\nSummary:\n{state.current_summary}\n\n"
              'Respond with JSON: {"verdict": "ACCEPT" or "REVISE", "feedback": "instructions or empty string"}\n'
              "Output raw JSON only.")
    raw_output = ""
    verdict = "ACCEPT"
    feedback = ""
    for attempt in range(4):
        try:
            raw_output = await _ollama_generate(prompt, json_mode=True)
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_output.strip(), flags=re.MULTILINE)
            parsed = json.loads(cleaned)
            verdict = str(parsed.get("verdict", "ACCEPT")).upper()
            feedback = str(parsed.get("feedback", ""))
            break
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            state.retry_count += 1
            logger.warning("[critic] Parse failure attempt %d: %s", attempt + 1, exc)
            if attempt == 3:
                verdict = "ACCEPT"
                feedback = ""
                break
            prompt = _build_healing_prompt(prompt, raw_output, '{"verdict": "ACCEPT|REVISE", "feedback": "string"}')
    state.critic_feedback = feedback
    if verdict == "REVISE" and state.retry_count < 3:
        state.steps.append(f"[Critic] Requested revision: {feedback[:120]}")
        state.next_node = "summariser"
    else:
        state.steps.append("[Critic] Summary accepted.")
        state.next_node = "END"
    return state