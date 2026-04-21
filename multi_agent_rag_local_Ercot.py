
import argparse
import os
import re
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    route: Literal["manual_rag", "direct"]
    retrieved_docs: list[Document]
    filtered_docs: list[Document]
    memory_context: str
    web_context: str
    answer: str
    source_used: Literal["manual", "web", "direct", "none"]
    confidence_note: str
    trace: str


class RouteDecision(BaseModel):
    datasource: Literal["manual_rag", "direct"] = Field(
        ..., description="manual_rag for ERCOT document-backed questions; direct otherwise."
    )


class RelevanceDecision(BaseModel):
    relevant: Literal["yes", "no"] = Field(
        ..., description="yes only if the document chunk is genuinely useful for answering the question."
    )


ROUTER_SYSTEM = """
You route questions for an ERCOT document assistant.
Use manual_rag for ERCOT market rules, protocols, operating guides, market guides, DAM, RTM,
settlements, ancillary services, outages, emergency operations, planning, compliance, reliability,
definitions, procedures, responsibilities, or market process questions.
Use direct only for simple greetings or general chat that does not need document retrieval.
Return only structured output.
""".strip()

GRADER_SYSTEM = """
You grade whether a retrieved document chunk is relevant to the user question.
Be strict. Return yes only if the chunk genuinely helps answer the question.
Return only structured output.
""".strip()

MEMORY_SYSTEM = """
Summarize the recent conversation into a short ERCOT-focused memory note.
Keep only useful context such as DAM/RTM topic, document family, compared concepts, or unresolved issue.
If there is nothing important, say: No important memory yet.
""".strip()

ANSWER_SYSTEM = """
You are a grounded ERCOT support assistant.

Priority:
1) Use ERCOT document context first.
2) Use web fallback only if document context is weak or empty.
3) If neither supports the answer, say so clearly.

Output rules:
- Be concise and specific.
- Prefer short bullets for procedures, responsibilities, or definitions.
- Do not invent rules, market responsibilities, timelines, or citations.
- Use only the provided context.
""".strip()

DIRECT_SYSTEM = "You are a concise assistant. Answer clearly and briefly.".strip()


def get_llm(model_name: str) -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in .env")
    return ChatGroq(model=model_name, api_key=api_key, temperature=0)


def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    hf_token = os.getenv("HF_TOKEN")
    model_kwargs = {"token": hf_token} if hf_token else {}
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)


def load_documents(data_dir: str) -> list[Document]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader = DirectoryLoader(str(base), glob="**/*.pdf", loader_cls=PyPDFLoader, silent_errors=True)
    documents = loader.load()
    if not documents:
        raise ValueError("No PDFs found. Put the ERCOT PDFs in the data folder.")

    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))
        doc.metadata["source"] = source
        doc.metadata["filename"] = Path(source).name
    return documents


def split_documents(documents: list[Document], chunk_size: int = 1100, chunk_overlap: int = 180) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def deduplicate_documents(chunks: list[Document]) -> list[Document]:
    seen: set[tuple[str, str, int | None]] = set()
    deduped: list[Document] = []
    for doc in chunks:
        text = " ".join(doc.page_content.split())
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        key = (filename, text, page)
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
    return deduped


def get_vector_store(persist_dir: str, collection_name: str, embeddings: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )


def ingest_documents(data_dir: str, persist_dir: str, collection_name: str, chunk_size: int, chunk_overlap: int) -> None:
    embeddings = get_embeddings()
    raw_docs = load_documents(data_dir)
    chunks = split_documents(raw_docs, chunk_size, chunk_overlap)
    chunks = deduplicate_documents(chunks)

    try:
        existing = get_vector_store(persist_dir, collection_name, embeddings)
        existing.delete_collection()
    except Exception:
        pass

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    print(f"Indexed {len(chunks)} deduplicated chunks into local Chroma collection '{collection_name}'.")


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No ERCOT document context available."
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        page_text = f", page={page}" if page is not None else ""
        parts.append(f"[Doc {i} | source={source}{page_text}]\n{doc.page_content}")
    return "\n\n".join(parts)


def extract_doc_hint(question: str) -> str | None:
    q = question.lower()
    if any(t in q for t in ["dam", "day-ahead", "rtm", "real-time", "ancillary", "settlement", "resource", "qse", "crr", "protocol"]):
        return "protocols"
    if any(t in q for t in ["operating guide", "emergency operations", "reliability", "load shed", "outage", "black start"]):
        return "operating"
    if any(t in q for t in ["commercial operations", "market guide", "settlement statement", "invoice", "commercial"]):
        return "market_guide"
    if any(t in q for t in ["planning guide", "planning", "interconnection", "transmission planning"]):
        return "planning"
    if "ercot" in q:
        return "ercot"
    return None


def filename_matches_hint(filename: str, hint: str | None) -> bool:
    if not hint:
        return False
    name = filename.lower()
    if hint == "protocols":
        return "protocol" in name
    if hint == "operating":
        return "operating" in name
    if hint == "market_guide":
        return "commercial" in name or "market-guide" in name or "market_guide" in name
    if hint == "planning":
        return "planning" in name
    if hint == "ercot":
        return "ercot" in name or "protocol" in name or "guide" in name
    return False


def build_query_variants(question: str) -> list[str]:
    q = question.strip()
    variants = [q]
    q_lower = q.lower()

    topical_terms = []
    if any(term in q_lower for term in ["dam", "day-ahead"]):
        topical_terms.append("ERCOT day-ahead market DAM protocols responsibilities settlements")
    if any(term in q_lower for term in ["rtm", "real-time"]):
        topical_terms.append("ERCOT real-time market RTM protocols responsibilities dispatch settlements")
    if any(term in q_lower for term in ["protocol", "rule", "responsibilit", "obligation", "requirement"]):
        topical_terms.append("ERCOT nodal protocols responsibilities obligations definitions")
    if any(term in q_lower for term in ["operating", "emergency", "reliability", "outage"]):
        topical_terms.append("ERCOT operating guide emergency operations reliability procedures")
    if any(term in q_lower for term in ["commercial", "market guide", "settlement", "invoice"]):
        topical_terms.append("ERCOT commercial operations market guide settlements invoicing processes")
    if any(term in q_lower for term in ["planning"]):
        topical_terms.append("ERCOT planning guide transmission planning interconnection procedures")
    if any(term in q_lower for term in ["compare", "difference", "versus", "vs"]):
        topical_terms.append("compare across ERCOT documents protocols operating guide market guide")

    hint = extract_doc_hint(q)
    if hint == "protocols":
        variants.append(f"ERCOT Nodal Protocols {q}")
    elif hint == "operating":
        variants.append(f"ERCOT Operating Guide {q}")
    elif hint == "market_guide":
        variants.append(f"ERCOT Commercial Operations Market Guide {q}")
    elif hint == "planning":
        variants.append(f"ERCOT Planning Guide {q}")
    elif hint == "ercot":
        variants.append(f"ERCOT {q}")

    for extra in topical_terms:
        variants.append(f"{q} {extra}")

    seen = set()
    output = []
    for v in variants:
        v_norm = v.lower().strip()
        if v_norm not in seen:
            seen.add(v_norm)
            output.append(v)
    return output[:4]


def rerank_docs(question: str, docs: list[Document]) -> list[Document]:
    hint = extract_doc_hint(question)
    q_lower = question.lower()

    scored: list[tuple[int, int, Document]] = []
    seen = set()

    for idx, doc in enumerate(docs):
        text = doc.page_content.lower()
        filename = doc.metadata.get("filename", "unknown")

        signature = (filename, doc.metadata.get("page"), " ".join(text.split())[:500])
        if signature in seen:
            continue
        seen.add(signature)

        score = 0
        if filename_matches_hint(filename, hint):
            score += 6
        if any(k in text for k in [
            "day-ahead market", "dam", "real-time market", "rtm", "protocol", "operating guide",
            "commercial operations market guide", "planning guide", "ercot", "qualified scheduling entity",
            "settlement", "ancillary service", "resource entity", "load serving entity"
        ]):
            score += 4

        question_terms = [t for t in re.findall(r"[a-zA-Z0-9\-]+", q_lower) if len(t) > 3]
        term_hits = sum(1 for t in question_terms if t in text)
        score += min(term_hits, 6)

        scored.append((score, idx, doc))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [doc for _, _, doc in scored]


def build_web_context(question: str) -> str:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or TavilyClient is None:
        return ""

    client = TavilyClient(api_key=api_key)

    snippets: list[str] = []
    seen_urls: set[str] = set()

    try:
        site_results = client.search(
            query=f"{question} site:ercot.com",
            max_results=4,
            search_depth="advanced",
        )
        for item in site_results.get("results", []):
            url = item.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                snippets.append(
                    f"[Website] {item.get('title', '')}\nURL: {url}\n{item.get('content', '')}"
                )
    except Exception:
        pass

    if len(snippets) < 2:
        try:
            web_results = client.search(query=f"ERCOT {question}", max_results=3, search_depth="basic")
            for item in web_results.get("results", []):
                url = item.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    snippets.append(
                        f"[Web] {item.get('title', '')}\nURL: {url}\n{item.get('content', '')}"
                    )
        except Exception:
            pass

    return "\n\n".join(snippets)


def infer_confidence(filtered_docs: list[Document], web_context: str, route: str) -> str:
    if route == "direct":
        return "Direct answer: no retrieval used."
    if len(filtered_docs) >= 3:
        return "High confidence: answer is grounded in multiple relevant ERCOT document chunks."
    if len(filtered_docs) >= 1:
        return "Medium confidence: answer is grounded in limited ERCOT document context."
    if web_context:
        return "Low confidence: ERCOT document retrieval was weak, so web fallback was used."
    return "Low confidence: insufficient supporting context was found."


def collect_manual_citations(docs: list[Document]) -> list[str]:
    citations = []
    seen = set()
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        label = f"{filename}, p.{page}" if page is not None else filename
        if label not in seen:
            seen.add(label)
            citations.append(label)
    return citations


def collect_web_citations(web_context: str) -> list[str]:
    citations = []
    seen = set()
    for line in web_context.splitlines():
        if line.startswith("URL: "):
            url = line.replace("URL: ", "").strip()
            if url and url not in seen:
                seen.add(url)
                citations.append(url)
    return citations


def build_graph(collection_name: str, persist_dir: str, model_name: str, top_k: int, memory_db: str):
    llm = get_llm(model_name)
    embeddings = get_embeddings()
    vector_store = get_vector_store(persist_dir, collection_name, embeddings)

    router = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM),
        ("human", "Question: {question}"),
    ]) | llm.with_structured_output(RouteDecision)

    grader = ChatPromptTemplate.from_messages([
        ("system", GRADER_SYSTEM),
        ("human", "Question:\n{question}\n\nDocument:\n{document}"),
    ]) | llm.with_structured_output(RelevanceDecision)

    memory_prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_SYSTEM),
        ("human", "Recent thread history:\n{history}"),
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM),
        (
            "human",
            "Memory context:\n{memory_context}\n\nUser question:\n{question}\n\nDocument context:\n{manual_context}\n\nWeb fallback context:\n{web_context}",
        ),
    ])

    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", DIRECT_SYSTEM),
        ("human", "Conversation history:\n{history}\n\nUser question:\n{question}"),
    ])

    def router_agent(state: GraphState) -> dict[str, Any]:
        question = state["messages"][-1].content
        decision = router.invoke({"question": question})
        return {
            "question": question,
            "route": decision.datasource,
            "trace": f"router:{decision.datasource}",
        }

    def retrieve_agent(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        variants = build_query_variants(question)

        raw_docs: list[Document] = []
        for variant in variants:
            docs = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": top_k, "fetch_k": max(top_k * 4, 12)},
            ).invoke(variant)
            raw_docs.extend(docs)

        reranked = rerank_docs(question, raw_docs)
        selected = reranked[: max(top_k + 2, 6)]
        return {
            "retrieved_docs": selected,
            "trace": state["trace"] + f" -> retrieve:{len(selected)}",
        }

    def grade_agent(state: GraphState) -> dict[str, Any]:
        filtered: list[Document] = []
        for doc in state.get("retrieved_docs", []):
            decision = grader.invoke({"question": state["question"], "document": doc.page_content})
            if decision.relevant == "yes":
                filtered.append(doc)

        if not filtered and state.get("retrieved_docs"):
            filtered = state["retrieved_docs"][:2]

        return {
            "filtered_docs": filtered,
            "trace": state["trace"] + f" -> grade:{len(filtered)}",
        }

    def memory_agent(state: GraphState) -> dict[str, Any]:
        history_msgs = state.get("messages", [])[-8:]
        history_text = "\n".join(f"{m.__class__.__name__}: {m.content}" for m in history_msgs) or "No history"
        response = (memory_prompt | llm).invoke({"history": history_text})
        return {
            "memory_context": response.content,
            "trace": state["trace"] + " -> memory",
        }

    def web_fallback_agent(state: GraphState) -> dict[str, Any]:
        if state.get("filtered_docs"):
            return {
                "web_context": "",
                "trace": state["trace"] + " -> web:skipped",
            }

        web_context = build_web_context(state["question"])
        web_state = "used" if web_context else "none"
        return {
            "web_context": web_context,
            "trace": state["trace"] + f" -> web:{web_state}",
        }

    def answer_agent(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        history_text = "\n".join(f"{m.__class__.__name__}: {m.content}" for m in state.get("messages", [])[:-1][-8:])

        if state["route"] == "direct":
            answer = (direct_prompt | llm).invoke({"history": history_text or "No history", "question": question}).content
            source_used = "direct"
        else:
            manual_context = format_docs(state.get("filtered_docs", []))
            web_context = state.get("web_context", "")
            answer = (answer_prompt | llm).invoke(
                {
                    "memory_context": state.get("memory_context", "No important memory yet."),
                    "question": question,
                    "manual_context": manual_context,
                    "web_context": web_context or "No web fallback used.",
                }
            ).content

            if state.get("filtered_docs"):
                source_used = "manual"
            elif web_context:
                source_used = "web"
            else:
                source_used = "none"

        manual_citations = collect_manual_citations(state.get("filtered_docs", []))
        web_citations = collect_web_citations(state.get("web_context", ""))
        confidence_note = infer_confidence(state.get("filtered_docs", []), state.get("web_context", ""), state["route"])

        parts = [answer.strip()]
        if manual_citations:
            parts.append("Sources:\n- " + "\n- ".join(manual_citations))
        elif web_citations:
            parts.append("Sources:\n- " + "\n- ".join(web_citations))
        else:
            parts.append("Sources:\n- No supporting citations found")

        parts.append(f"Source used: {source_used}")
        parts.append(f"Confidence: {confidence_note}")
        final_answer = "\n\n".join(parts)

        return {
            "answer": final_answer,
            "source_used": source_used,
            "confidence_note": confidence_note,
            "trace": state["trace"] + f" -> answer:{source_used}",
            "messages": [AIMessage(content=final_answer)],
        }

    workflow = StateGraph(GraphState)
    workflow.add_node("router_agent", router_agent)
    workflow.add_node("retrieve_agent", retrieve_agent)
    workflow.add_node("grade_agent", grade_agent)
    workflow.add_node("memory_agent", memory_agent)
    workflow.add_node("web_fallback_agent", web_fallback_agent)
    workflow.add_node("answer_agent", answer_agent)

    workflow.add_edge(START, "router_agent")
    workflow.add_conditional_edges(
        "router_agent",
        lambda state: state["route"],
        {"manual_rag": "retrieve_agent", "direct": "answer_agent"},
    )
    workflow.add_edge("retrieve_agent", "grade_agent")
    workflow.add_edge("grade_agent", "memory_agent")
    workflow.add_edge("memory_agent", "web_fallback_agent")
    workflow.add_edge("web_fallback_agent", "answer_agent")
    workflow.add_edge("answer_agent", END)

    conn = sqlite3.connect(memory_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return workflow.compile(checkpointer=checkpointer)


def run_chat(app, thread_id: str, show_trace: bool) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    print("Local multi-agent RAG chat started. Type 'exit' to stop.\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        result = app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        print(f"Assistant:\n{result['answer']}\n")
        if show_trace:
            print(f"Agent trace:\n{result['trace']}\n")


def run_query(app, thread_id: str, query: str, show_trace: bool) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
    print(result["answer"])
    if show_trace:
        print(f"\nAgent trace:\n{result['trace']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local multi-agent RAG over ERCOT PDFs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("--data-dir", default="data")
    ingest.add_argument("--persist-dir", default="./chroma_store")
    ingest.add_argument("--collection", default="ercot_local")
    ingest.add_argument("--chunk-size", type=int, default=1100)
    ingest.add_argument("--chunk-overlap", type=int, default=180)

    chat = subparsers.add_parser("chat")
    chat.add_argument("--persist-dir", default="./chroma_store")
    chat.add_argument("--collection", default="ercot_local")
    chat.add_argument("--model", default="llama-3.1-8b-instant")
    chat.add_argument("--top-k", type=int, default=5)
    chat.add_argument("--thread-id", default="demo-thread")
    chat.add_argument("--memory-db", default="rag_memory_ercot.sqlite")
    chat.add_argument("--show-trace", action="store_true")

    ask = subparsers.add_parser("ask")
    ask.add_argument("--persist-dir", default="./chroma_store")
    ask.add_argument("--collection", default="ercot_local")
    ask.add_argument("--model", default="llama-3.1-8b-instant")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--thread-id", default="demo-thread")
    ask.add_argument("--memory-db", default="rag_memory_ercot.sqlite")
    ask.add_argument("--query", required=True)
    ask.add_argument("--show-trace", action="store_true")

    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.command == "ingest":
        ingest_documents(args.data_dir, args.persist_dir, args.collection, args.chunk_size, args.chunk_overlap)
        return

    app = build_graph(
        collection_name=args.collection,
        persist_dir=args.persist_dir,
        model_name=args.model,
        top_k=args.top_k,
        memory_db=args.memory_db,
    )

    if args.command == "chat":
        run_chat(app, args.thread_id, args.show_trace)
    else:
        run_query(app, args.thread_id, args.query, args.show_trace)


if __name__ == "__main__":
    main()
