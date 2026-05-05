# ERCOT Multi-Agent RAG System

A local multi-agent Retrieval-Augmented Generation (RAG) system built with LangGraph to answer questions over ERCOT market documents such as Nodal Protocols, Operating Guides, and Market Guides.

## Business Problem

ERCOT market rules are spread across long PDF documents, including Nodal Protocols, Operating Guides, Planning Guides, and Commercial Operations Market Guides. Analysts often need to manually search across these documents to answer operational, settlement, or market-rule questions.

This workflow is slow, inconsistent, and difficult to audit.

This project turns that workflow into a multi-agent RAG assistant that retrieves relevant ERCOT document sections, grades context quality, generates cited answers, supports memory, and provides traceability through an agent trace.

## Features

- Multi-agent workflow with LangGraph
- Local PDF ingestion with Chroma vector store
- ERCOT-specific query expansion and reranking
- Context grading before answer generation
- Source citations and confidence notes
- Optional Tavily web fallback when local context is weak
- Agent trace for debugging, demos, and auditability
- Local memory with SQLite checkpoints

## Demo

Example query output with source citations, confidence note, and agent trace:

![Demo](assets/demo.png)

## Architecture

The system uses a multi-agent RAG workflow with LangGraph and LangChain.

The workflow uses six agents:

1. **Router Agent** — decides whether the query should use local ERCOT documents or fallback search.
2. **Retrieval Agent** — retrieves relevant chunks from the local Chroma vector store.
3. **Grading Agent** — checks whether retrieved context is relevant enough to answer the question.
4. **Memory Agent** — maintains thread-level memory using SQLite checkpoints.
5. **Web Fallback Agent** — optionally searches externally when local document context is weak.
6. **Answer Agent** — generates a cited answer with confidence notes.

The user question enters the graph, passes through routing, retrieval, grading, memory, fallback when needed, and final answer generation. This makes the workflow easier to debug, inspect, and improve compared with a single-step chatbot.

## Key Technical Decisions

### 1. LangGraph instead of a single RAG chain

I used LangGraph because the workflow needed explicit control over routing, retrieval, grading, fallback, memory, and final answer generation. This made the system more debuggable and easier to extend than a simple one-step chatbot.

### 2. Local-first retrieval

ERCOT PDFs are ingested into a local Chroma vector store. This design supports private-document workflows and reduces dependency on external systems for retrieval.

### 3. Traceability by design

The system includes source citations, confidence notes, and an optional agent trace. This helps users verify where an answer came from and makes debugging easier when retrieval quality is weak.

## Biggest Technical Challenge

The biggest challenge was retrieval quality. ERCOT documents are dense and often use similar terminology across different sections. A normal vector search can retrieve chunks that sound relevant but do not actually answer the user’s question.

To reduce hallucination risk, I added ERCOT-specific query expansion, reranking, context grading, citations, confidence notes, and fallback behavior when local context is weak.

The goal was not only to generate an answer, but to avoid giving a confident answer when the evidence was insufficient.

## Production Evaluation and Guardrails

Moving this system into production, I would add a formal RAG evaluation and guardrail layer.

### RAG Evaluation

- Retrieval quality
- Citation accuracy
- Answer faithfulness
- Hallucination detection
- Fallback rate
- Latency

### Guardrails

- Prompt injection detection
- Source-grounding validation
- Response validation
- Escalation when evidence is weak

## Production Improvements

If rebuilding this for production, I would separate the system into more modular services:

```text
src/
  ingestion/
  retrieval/
  graph/
  memory/
  evaluation/
  guardrails/
  api/
  monitoring/
```

## Required Data

Create a `data/` folder and place these ERCOT PDFs inside it:

- `April-1-2026-Nodal-Protocols.pdf`
- `February-1-2026-Nodal-Operating-Guide.pdf`
- `February-1-2026-Commercial-Operations-Market-Guide.pdf`
- `February-1-2026-Planning-Guide.pdf`

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and add required keys:

```bash
cp .env.example .env
```

## Quick Start

### Ingest ERCOT documents

```bash
python multi_agent_rag_local_Ercot.py ingest --data-dir data
```

### Ask a question

```bash
python multi_agent_rag_local_Ercot.py ask \
  --query "What are the DAM and RTM responsibilities under ERCOT protocols?" \
  --thread-id demo \
  --show-trace
```

### Start interactive chat

```bash
python multi_agent_rag_local_Ercot.py chat --show-trace
```