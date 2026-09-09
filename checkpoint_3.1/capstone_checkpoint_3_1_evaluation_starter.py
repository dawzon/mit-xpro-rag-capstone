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
r"""Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.

This demonstration system is not your capstone system and 
does not use the Research Paper Navigator or Wikipedia corpus.
"""

# %% [markdown]
# # Capstone Checkpoint 3.1 — Evaluation Infrastructure and Baseline Diagnosis
# **MO-LLM Module 3 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# #
# This mirrors **Lab 3.1** (an LLM-judge that scores answers pass/fail against grading
# notes), applied to your capstone system. 
# You have a baseline retrieval system from Checkpoint 2.1. In this checkpoint, 
# you will use a structured evaluation approach to measure baseline performance, diagnose strengths 
# and weaknesses, and examine whether the evaluation framework detects problematic outputs.
# The starter script includes a small demonstration corpus and retriever to illustrate the 
# evaluation workflow. Apply the same evaluation approach to your selected capstone scenario and 
# baseline retrieval system. The graded deliverable is the completed Capstone Checkpoint 3.1 worksheet.
#
# **Learning outcomes (Module 3):**
# 1. Define evaluation metrics that reflect the real-world performance requirements of an LLM-powered retrieval system.
# 2. Identify key variables that influence system performance during evaluation and development.
# 3. Use language models to support evaluation tasks while avoiding common pitfalls.
# 4. Evaluate the performance of a retrieval-augmented system during development using a structured evaluation framework.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario and baseline retriever** from Checkpoints 1.1 and 2.1.
#
# | Scenario | Corpus |
# |---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) |

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12.**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# Runs on a tiny built-in sample corpus, so you do not need to prepare your own dataset.
# It still requires an OpenRouter API key to run the LLM (it is not offline or free of API
# calls). You apply the same evaluation approach to your real system for the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANSWER_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
JUDGE_MODEL = "openai/gpt-5.4"
TEMPERATURE = 0.2
# TOP_K = 3
TOP_K = 8
LOG_PATH = Path.cwd() / "checkpoint_3_1_evaluation.log"

SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES describing "
    "what a correct answer must contain. Reply with exactly one word: 'pass' if the "
    "answer satisfies the grading notes, or 'fail' if it does not."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this program, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=ANSWER_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

def make_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")

# %% [markdown]
# ## Demonstration corpus + retriever (provided to illustrate the evaluation workflow)

# %%
# SAMPLE_DOCS = [
#     {"id": "doc1", "text": "Program synthesis: generating programs automatically from a specification, such as input-output examples or a logical formula."},
#     {"id": "doc2", "text": "The sketching approach lets a programmer write a partial program with holes, and a synthesizer fills the holes to satisfy a specification."},
#     {"id": "doc3", "text": "Retrieval-augmented generation grounds a language model's answers in documents retrieved from a corpus, reducing hallucination."},
#     {"id": "doc4", "text": "BM25 is a keyword ranking function that scores documents by term frequency and inverse document frequency."},
#     {"id": "doc5", "text": "Vector search embeds text into dense vectors and ranks documents by cosine similarity to the query embedding."},
#     {"id": "doc6", "text": "Evaluation of retrieval systems measures whether the retrieved documents actually contain the information needed to answer the query."},
# ]
# DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


# def _tokens(text: str) -> set[str]:
#     return set(re.findall(r"[a-z0-9]+", text.lower()))


# def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
#     q = _tokens(query)
#     scored = [(d["id"], float(len(q & _tokens(d["text"])))) for d in SAMPLE_DOCS]
#     scored.sort(key=lambda x: x[1], reverse=True)
#     return [(doc_id, score) for doc_id, score in scored[:k] if score > 0]

####################################### PDF LOADING ####################################### 
def load_pdf_pages(pdf_paths: list[Path]) -> list[Document]:
    """Extract each PDF page into its own LangChain Document."""
    documents = []

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

    return documents


PDF_DIR = Path("../checkpoint_1.1/ResearchPapers/")
files = sorted(PDF_DIR.glob("*.pdf"))
docs = load_pdf_pages(files)
# Add index so we can get it from Chroma results
for index, doc in enumerate(docs):
    doc.metadata.update({
        "doc_index": index,
    })

print(f"Loaded {len(docs)} pages from {len(files)} PDFs")

####################################### DOCUMENT RETRIEVAL ####################################### 
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
    chroma = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings) #TODO is this right
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


def answer(llm: ChatOpenAI, query: str, doc_ids: list[int]) -> str:
    context = "\n\n".join(f"[{i}] {docs[i].page_content}" for i in doc_ids)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
#     context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
#     messages = [
#         SystemMessage(content=ANSWER_SYSTEM),
#         HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
#     ]
#     return llm.invoke(messages).content

# %% [markdown]
# ## Step 2 — The evaluation metric (provided)
#
# A simple **LLM-judge** that returns pass/fail by checking an answer against grading
# notes — the same idea as Lab 3.1's DiscreteMetric, written directly here so the
# checkpoint needs no extra packages. Tuning this metric (stricter notes, a 'partial'
# level, a stronger judge model) is part of the diagnosis.

# %%
def judge(llm: ChatOpenAI, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=f"ANSWER:\n{answer_text}\n\nGRADING NOTES:\n{grading_notes}\n\nVerdict (pass/fail):"),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if "pass" in verdict else "fail"


# %% [markdown]
# ## Step 3 — Your evaluation set (TODO)
#
# Define the test set your evaluation runs on. Each item is a question plus
# **grading notes** — a short description of what a correct answer must contain (the
# judge checks the answer against these). Good evaluation sets include questions you
# expect to pass AND questions that probe known weaknesses.
#
# Return a list of 3-5 dicts: `{"question": "...", "grading_notes": "..."}`.

# %%
def my_eval_set() -> list[dict]:
    """Return 3-5 evaluation items for YOUR scenario.

    TODO — your turn. Each item is {"question": "...", "grading_notes": "..."}.
    The grading_notes describe what a correct answer MUST contain, in one sentence.
    Include at least one question you expect your baseline to get WRONG, so your
    diagnosis has something to find.

    Delete the raise NotImplementedError line once your code works.
    """
    return [
        # Exact question about a specific document (DBLP_conf_nips_BastaniPS18.pdf)
        {
            "question": "What authors wrote the paper \"Verifiable Reinforcement Learning via Policy Extraction\"?",
            "grading_notes": "This paper was written by Osbert Bastani, Yewen Pu, and Armando Solar-Lezama."
        },
        # Find a paper with a particular topic (DBLP_journals_corr_abs-1208-2925.pdf)
        {
            "question": "Has the study of program synthesis been applied to social networks?",
            "grading_notes": "Yes. In \"Using Program Synthesis for Social Recommendations\", program synthesis was used to learn users' preferences."
        },
        # Question about information that is not contained in the corpus
        {
            "question": "What are some program synthesis techniques that were explored before the year 2000? Provide direct quotations from original sources, including a citation of the original paper",
            "grading_notes": "I did not find any information about program synthesis techniques before the year 2000."
        },
        # Many possible answers, pushing the limit of what a pass/fail test can measure
        {
            "question": "What are some programming languages that are used in modern computer science research?",
            "grading_notes": "Some of the programming languages used include Haskell, Scala, Lua, Python, Racket, and C++." # These are some of the answers from the checkpoint 2.1 log
        },
        # Answer might contain reference to other papers (DBLP_conf_aplas_Solar-Lezama09.pdf)
        {
            "question": "What is unique about program synthesis by sketching?",
            "grading_notes": "Sketching invovles writing a high-level program that leaves low-level details to be solved automatically."
        },
    ]


# %% [markdown]
# Run the demonstration code to understand the evaluation workflow. Then apply the same evaluation 
# design to your own capstone baseline system, using its retrieval and answer-generation functions. 
# Record results from your capstone system in the worksheet.
#
# ## Step 4 — Run the baseline evaluation
#
# Answers each question with the baseline retriever, scores it with the judge, and
# reports the pass rate. This is your baseline diagnosis: the failures are what you
# analyse in the report.

# %%
def format_hit(hit: tuple) -> str:
    index = hit[0]
    score = hit[1]
    doc = docs[index]
    return f"""HIT:
    index: {index}
    score: {score}
    filename: {doc.metadata["file_name"]}
    page: {doc.metadata["page"]}"""

def run_evaluation() -> None:
    llm = make_llm()
    judge_llm = make_llm()
    eval_set = my_eval_set()
    passes = 0
    print(f"Checkpoint 3.1 — baseline evaluation  |  scenario: {SCENARIO}\n")
    for i, item in enumerate(eval_set, 1):
        hits = retrieve(item["question"], TOP_K)
        ans = answer(llm, item["question"], [doc_id for doc_id, _ in hits]) if hits else "(no documents retrieved)"
        verdict = judge(judge_llm, ans, item["grading_notes"])
        passes += verdict == "pass"
        print("=" * 72)
        print(f"Q{i}: {item['question']}")
        print(f"  retrieved={"\n".join([format_hit(hit) for hit in hits])}  verdict={verdict.upper()}")
        print(f"  answer: {ans}")
        log(f"Q{i}: {item['question']}", f"retrieved={"\n".join([format_hit(hit) for hit in hits])}\nverdict={verdict}\nanswer={ans}")
    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")


# %% [markdown]
# ## Step 5 — Validate the framework: can it catch a manipulated answer? (provided)
#
# A good evaluation framework must FAIL a wrong answer, not just pass good ones. This
# takes a question your corpus can answer, produces a correct answer, then feeds the
# judge a deliberately manipulated (false) answer — and checks that the judge flags it.
# This is your "evaluation framework validation" evidence for the report.

# %%
def validate_framework() -> None:
    llm = make_llm()
    judge_llm = make_judge_llm()
    q = "What are some programming languages that are used in modern computer science research?"
    notes = "Languages used in modern computer science research include Haskell, Scala, Lua, Python, Racket, and C++."
    good = answer(llm, q, [doc_id for doc_id, _ in retrieve(q)])
    manipulated = "COBOL is often used in modern computer science research."
    good_verdict = judge(judge_llm, good, notes)
    manip_verdict = judge(judge_llm, manipulated, notes)
    print("\n--- Framework validation ---")
    print(f"  correct answer   -> {good_verdict.upper()}   (expected PASS)")
    print(f"  manipulated answer -> {manip_verdict.upper()}   (expected FAIL)")
    print("  The framework works if it PASSES the correct answer and FAILS the manipulated one.")
    log("FRAMEWORK VALIDATION", f"good={good_verdict} manipulated={manip_verdict}")


# %%
run_evaluation()
validate_framework()

# %% [markdown]
# ## Step 6 — Your written responses in the Capstone Checkpoint 3.1 worksheet
#
# Complete the Capstone Checkpoint 3.1 worksheet using evidence from your capstone evaluation.
# Address the seven sections: system overview, evaluation design, testing approach, baseline results, 
# performance analysis, evaluation framework validation, and reflection and next steps.
#
# 1. **System overview** — your scenario and your 2.1 baseline retriever.
# 2. **Evaluation design** — your criteria and metric (what "correct" means; how the
#    judge decides pass/fail; any thresholds).
# 3. **Testing approach** — how you built your evaluation set and ran it.
# 4. **Baseline results** — the pass/fail outcomes and where the system falls short.
# 5. **Performance analysis** — what the results reveal about strengths, weaknesses,
#    and failure modes.
# 6. **Evaluation framework validation** — show your framework detects a degraded or
#    manipulated output (use the Step 5 result, or your own).
# 7. **Reflection and next steps** — limitations of your evaluation and what you'll
#    improve (this motivates the advanced retrieval in Checkpoint 4.1).
