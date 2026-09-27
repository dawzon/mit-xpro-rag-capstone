# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
r"""Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Design workflows that coordinate retrieval, reasoning, and tool use in a RAG system.
# 4. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 5. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
# 6. Build an agent-based system that integrates retrieval and external tools to complete user tasks.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Get a free OpenRouter key at <https://openrouter.ai/keys>. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.load import dumps, loads
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from langgraph.graph import END, StateGraph

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 5
LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

# DECIDE_SYSTEM = (
#     "You are an agent retrieving from a small document collection. Given the question, "
#     "the queries already run, and the documents found so far, decide what to do next. "
#     'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
#     '"reasoning": "..."}. Set done=true when you have enough to answer; otherwise give '
#     "1-2 new_queries targeting what is still missing (do not repeat past queries)."
# )
DECIDE_SYSTEM = (
    "You are an agent whose job is to answer questions about research papers. Given the question, "
    "the queries already run, and the documents found so far, decide which action to take next. "
    "Don't answer the question until you have good sources for answering. "
    'Respond with ONLY a JSON object: '
    '{'
    '  "action": "retrieve" | "clarify" | "answer", // The next action you want to take'
    '  "query": "query text", // Only for action "retrieve". This is a query that will run against a hybrid keyword/vector retrieval engine'
    '  "clarification": "question text", // Only for action "clarify". This is a question that the user will answer. You can also ask a question that would help the user '
    '  "reasoning": "brief explanation"'
    '}'
    'Here is what has been done so far:'
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents, "
    "quoting where you can. If they do not contain the answer, say so."
    "The answer that you give is the final step in the conversation, so do not ask any follow up questions or suggest any further steps."
    "The user may not want to continue the conversation, so don't suggest other things that you could possibly do."
    "However, if you think there are tangentially related topics that may supplement the answer,"
    "You may elect to include a bulleted list titled \"SEE ALSO\" at the end of the answer."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")

# %% [markdown]
# ## A tiny sample corpus + a keyword retriever (provided)

# %%
# SAMPLE_DOCS = [
#     {"id": "d1", "text": "Program synthesis generates programs from a specification, such as input-output examples."},
#     {"id": "d2", "text": "The sketching approach lets a programmer leave holes in a program for a synthesizer to fill."},
#     {"id": "d3", "text": "Retrieval-augmented generation grounds a model's answers in retrieved documents to reduce hallucination."},
#     {"id": "d4", "text": "An agentic retriever decides at each step whether it has enough information or should search again."},
#     {"id": "d5", "text": "A tool-using agent chooses among actions — search, look up by date, ask the user — to complete a task."},
#     {"id": "d6", "text": "Evaluating an agent compares its answers and cost against a fixed single-pass pipeline."},
# ]
# DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


# def retrieve(query: str, k: int = 2) -> list[str]:
#     q = set(re.findall(r"[a-z0-9]+", query.lower()))
#     scored = [(d["id"], len(q & set(re.findall(r"[a-z0-9]+", d["text"].lower())))) for d in SAMPLE_DOCS]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [doc_id for doc_id, s in scored[:k] if s > 0]

####################################### PDF LOADING ####################################### 
CACHE_DIR = './doc_cache'

def load_pdf_pages(pdf_paths: list[Path]) -> list[Document]:
    """Extract each PDF page into its own LangChain Document."""
    PDF_DIR = Path("../checkpoint_1.1/ResearchPapers/")
    files = sorted(PDF_DIR.glob("*.pdf"))

    documents = []

    os.makedirs(CACHE_DIR , exist_ok=True)

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        for page_index, page in enumerate(reader.pages):
            page_content = (page.extract_text() or "").strip()
            if not page_content:
                continue

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": str(pdf_path),
                        "file_name": pdf_path.name,
                        "page": page_index,          # zero-based, LangChain convention
                        "page_number": page_index + 1,  # one-based, for display
                        "total_pages": total_pages,
                    },
                )
            )

            doc_index = len(documents) - 1
            with open(f"{CACHE_DIR}/{doc_index}", "w", encoding="utf-8") as file:
                file.write(dumps(documents[doc_index]))

    print(f"Loaded {len(documents)} pages from {len(pdf_paths)} PDFs")
    return documents

def load_cache():
    print("Using document cache...")
    documents = []
    filenames = os.listdir(CACHE_DIR)
    filenames.sort(key=int)
    for filename in filenames:
        with open(f"{CACHE_DIR}/{filename}", encoding="utf-8") as file:
            documents.append(loads(file.read()))
    print(f"Loaded {len(documents)} documents")
    return documents



docs = load_cache() if os.path.exists(CACHE_DIR) else load_pdf_pages()
# Add index so we can get it from Chroma results
for index, doc in enumerate(docs):
    doc.metadata.update({
        "doc_index": index,
    })

####################################### DOCUMENT RETRIEVAL ####################################### 
TOP_K = 8

# Stopwords list from Lab 1.2
_STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "of", "by", "with", "from", "into", "onto", "upon",
    "about", "above", "below", "between", "through", "during", "before", "after",
    "under", "over", "around", "along", "across", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them",
    "my", "our", "your", "his", "its", "their", "this", "that", "these", "those",
    "as", "if", "up", "out", "not", "no",
}
def tokenize(text: str) -> list[str]:
    """Lowercase, split into alphanumeric tokens, drop stopwords. The same
    tokenizer is used to index documents and to tokenize queries."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]
bm25 = BM25Okapi([tokenize(doc.page_content) for doc in docs])
def retrieve_bm25(query: str, topK: int):
    scores = bm25.get_scores(tokenize(query))
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topK]
    return [(i, docs[i].page_content, scores[i]) for i in top_indexes]

CHROMA_DIR = './chroma_db'
EMBEDDING_MODEL = "openai/text-embedding-3-small"
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=check_api_key(),
    base_url=OPENROUTER_BASE_URL,
)
if not os.path.isdir(CHROMA_DIR):
    Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
def retrieve_vector(query: str, topK: int):
    chroma = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    top_vector = chroma.similarity_search_with_score(query, k=topK)
    return [(d.metadata.get("doc_index", "unknown"), d.page_content, s) for d, s in top_vector]

# Normalize function from Lab 2.2
def normalize(scores: list[float], invert: bool = False) -> list[float]:
    """Min-max scale a list of scores to [0, 1]. If invert is True, flip the scores 
    so a LOW raw value (e.g., a small vector distance = very similar) becomes a HIGH 
 normalized score."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.5] * len(scores)  # all equal → neutral
    norm = [(s - lo) / (hi - lo) for s in scores]
    return [1.0 - n for n in norm] if invert else norm

# Rank fusion adapted from Lab 2.2
def retrieve(query: str, k: int = TOP_K) -> list[tuple[int, float]]:
    """Hybrid retrieval"""
    top_bm25 = retrieve_bm25(query, k)
    top_vector = retrieve_vector(query, k)

    bm_norm: dict[int, float] = {}
    vec_norm: dict[int, float] = {}
    if top_bm25:
        for (index, content, _), val in zip(top_bm25, normalize([s for _, _, s in top_bm25])):
            bm_norm[index] = val
    if top_vector:
        for (index, content, _), val in zip(top_vector, normalize([d for _, _, d in top_vector], invert=True)):
            vec_norm[index] = val

    WEIGHT_BM25 = 0.3
    WEIGHT_VECTOR = 0.7
    all_found_ids = bm_norm.keys() | vec_norm.keys()
    fused = [
        (id, WEIGHT_BM25 * bm_norm.get(id, 0.0) + WEIGHT_VECTOR * vec_norm.get(id, 0.0))
        for id in all_found_ids
    ]
    fused.sort(key=lambda t: t[1], reverse=True)
    return fused[:k]

# def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str]) -> dict:
#     docs = "\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in collected) or "(none yet)"
#     user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nDocuments so far:\n{docs}"
#     raw = llm.invoke([SystemMessage(content=DECIDE_SYSTEM), HumanMessage(content=user)]).content.strip()
#     if raw.startswith("```"):
#         raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
#     try:
#         d = json.loads(raw)
#         return {"done": bool(d.get("done", True)), "new_queries": d.get("new_queries", []) or [],
#                 "reasoning": d.get("reasoning", "")}
#     except (json.JSONDecodeError, ValueError):
#         return {"done": True, "new_queries": [], "reasoning": "parse-fail -> stop"}


class AgentState(TypedDict):
    history: list[str]
    step_count: int
    answer: str
    # LLM decide fields
    action: str
    query: str
    clarification: str
    reasoning: str

def agentic_answer(llm: ChatOpenAI, question: str) -> str:
    """Provided: a minimal agentic loop — retrieve, decide whether to continue, repeat."""

    graph = StateGraph(AgentState)

    def node_decide(state: AgentState):
        step_count = state.get("step_count")
        if step_count >= MAX_STEPS:
            return {
                "action": "answer"
            }

        user_content = '\n'.join([
            *state.get("history")
        ])
        response = llm.invoke([
            SystemMessage(content=DECIDE_SYSTEM),
            HumanMessage(content=user_content)
        ])

        raw = (response.content if hasattr(response, "content") else str(response)).strip()
        print(f"Decide step: {raw}")
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            result = json.loads(raw)
            action = result.get("action", "answer")
        except (json.JSONDecodeError, ValueError):
            action, result = "answer", {}

        new_history = list(state.get("history"))
        new_history.append(f"Chosen action: {action}. Reasoning: {result.get("reasoning")}")

        new_step_count = step_count + 1
        common_state = {
            "action": action,
            "step_count": new_step_count,
            "history": new_history,
        }

        if action == 'retrieve':
            return {
                **common_state,
                "query": result.get("query")
            }
        if action == 'clarify':
            return {
                **common_state,
                "clarification": result.get("clarification"),
            }
        if action == 'answer':
            # Answer doesn't need special state
            return {
                **common_state,
            }
    graph.add_node("decide", node_decide)

    def node_retrieve(state: AgentState):
        query = state.get("query")
        new_history = list(state.get("history"))
        new_history.append(f"Retrieved documents for question: {query}")
        documents = retrieve(query, TOP_K)
        for doc_id, score in documents:
            new_history.append(f"RETRIEVED DOC (id: {id})\n{docs[doc_id]}")
        return {
            "history": new_history
        }
    graph.add_node("retrieve", node_retrieve)

    def node_clarify(state: AgentState):
        question = state.get("clarification")
        print(f"Assistant: {question}")
        user_answer = input("You: ").strip()
        new_history = list(state.get("history"))
        new_history.append(f"Question: {question} Answer: {user_answer}")
        return {
            "history": new_history
        }
    graph.add_node("clarify", node_clarify)

    def node_answer(state: AgentState):
        user_content = '\n'.join([
            *state.get("history")
        ])
        response = llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=user_content)
        ])
        raw = (response.content if hasattr(response, "content") else str(response)).strip()
        return {
            "answer": raw
        }
    graph.add_node("answer", node_answer)

    def get_next(state: AgentState):
        next_node = state.get("action")
        return next_node

    # All actions return to decide node
    graph.add_edge("retrieve", "decide")
    graph.add_edge("clarify", "decide")

    graph.add_conditional_edges("decide", get_next)
    graph.set_entry_point("decide")
    graph.add_edge("answer", END)

    initial_state = {
        "next_action": "decide",
        "history": [f"ORIGINAL QUESTION: {question}"],
        "step_count": 0,
    }

    answer = graph.compile().invoke(initial_state)["answer"]
    return answer

    # OLD CODE
    # collected: dict[str, str] = {}
    # executed: list[str] = []
    # pending = [question]
    # for step in range(MAX_STEPS):
    #     for q in pending:
    #         for doc_id in retrieve(q):
    #             collected[doc_id] = DOC_BY_ID[doc_id]["text"]
    #         executed.append(q)
    #     d = decide(llm, question, collected, executed)
    #     print(f"  step {step + 1}: have {sorted(collected)}  -> done={d['done']}  ({d['reasoning'][:60]})")
    #     if d["done"] or not d["new_queries"]:
    #         break
    #     pending = d["new_queries"]
    # context = "\n\n".join(f"[{i}] {collected[i]}" for i in collected)
    # return llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
    #                    HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
    """Return your agent design for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "tools": list[str]      — the actions/tools your agent can take (e.g.,
                                  ["semantic_search", "find_by_author", "follow_citation",
                                   "clarify", "answer"]).
      - "stop_condition": str   — how the agent decides it has enough to answer.
      - "system_prompt_idea": str — one or two sentences on how you'll instruct the agent
                                  to choose actions (this is what "guiding agent behavior
                                  with system prompts" means).
      - "test_tasks": list[str] — 2-3 questions for your scenario that need more than one
                                  retrieval step (so the agentic loop earns its keep).

    Example (illustrative — replace with your own scenario):
        return {
            "tools": ["semantic_search", "find_by_author", "clarify", "answer"],
            "stop_condition": "the retrieved passages cover every part of the question",
            "system_prompt_idea": "Search first; clarify only if the request is ambiguous; "
                                  "answer once the retrieved text supports a grounded reply.",
            "test_tasks": [
                "What changed between the v1 and v2 proposals, and who approved it?",
                "Summarize the budget decisions discussed across Q1 and Q2.",
            ],
        }

    Delete the raise NotImplementedError line once your code works.
    """
    return {
        "tools": ["retrieve", "clarify", "answer"],
        "stop_condition": "Either the LLM decides it is ready to answer, or the LLM runs the maximum allowed number of retrievals.",
        "system_prompt_idea": DECIDE_SYSTEM,
        "test_tasks": [
            # Should compare to other documents
            "What is unique about program synthesis by sketching? And how does it compare to other techniques?",
            # Vague question
            "Policy extraction?",
            # Simple question. The agent shouldn't get lost trying to answer this
            "What authors wrote the paper \"Verifiable Reinforcement Learning via Policy Extraction\"?",
        ],
    }


# %% [markdown]
# ## Step 3 — Run the agentic loop and capture the evidence
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def run() -> None:
    llm = make_llm()
    # question = "How does an agentic retriever differ from a fixed pipeline, and how is it evaluated?"
    print(f"Checkpoint 5.1 — agentic RAG demo  |  scenario: {SCENARIO}")
    # print(f"Question: {question}\n")
    # answer = agentic_answer(llm, question)
    # print(f"\nAgent answer:\n{answer}\n")
    # log("AGENTIC", f"Q: {question}\nA: {answer}")
    # try:
    #     print("Your agent plan:")
    #     print(json.dumps(my_agent_plan(), indent=2))
    # except NotImplementedError as e:
    #     print(f"[my_agent_plan not done yet] {e}")

    for question in my_agent_plan()["test_tasks"]:
        answer = agentic_answer(llm, question)
        log("AGENTIC", f"Q: {question}\nA: {answer}")

    print("=" * 72)
    print("Done. Build this agent for your real system and compare it to your 2.1 baseline.")


run()

# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **written submission** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).
