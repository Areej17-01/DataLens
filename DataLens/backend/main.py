import json
import os
import sys
import uuid
from pathlib import Path
from typing import AsyncGenerator, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Ensure the repo root is importable when running from the DataLens/backend folder.
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from agentpro import ReactAgent, create_model

from .config import (
    CSV_EXTENSIONS,
    FRONTEND_ROOT,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPEN_ROUTER_KEY,
    OPEN_ROUTER_MODEL,
    PDF_EXTENSIONS,
    QDRANT_API_KEY,
    QDRANT_URL,
    TEMP_ROOT,
)
from .doc_tool.csv_analyst_tool import CsvAnalystTool
from .doc_tool.insights_dashboard import DashboardGeneratorTool
from .doc_tool.rag_tool import DataLensRAGTool
from .doc_tool.visualization_code_tool import PlotGeneratorTool
from .services.document_processing import (
    build_csv_metadata,
    build_text_documents,
    extract_csv_text,
    extract_images_from_pdf,
    extract_text_from_pdf,
    save_csv_metadata,
    save_text_content,
)
from .services.embeddings import EmbeddingService
from .services.qdrant_store import QdrantVectorStore
from agentpro.tools import AresInternetTool, SlideGenerationTool, YFinanceTool


# Create runtime directories before mounting them as static paths.
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
FRONTEND_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DataLens Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
app.mount("/temp", StaticFiles(directory=TEMP_ROOT), name="temp")
app.mount("/frontend", StaticFiles(directory=FRONTEND_ROOT), name="frontend")


if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in the environment for DataLens backend.")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set in the environment for DataLens backend.")


def chunk_text(text: str, chunk_size: int = 80):
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]


def extract_tool_output(agent_response):
    merged = {
        "llm_response": "",
        "retrieved_context": "",
        "context_used": "",
        "image_paths": [],
        "plot_paths": [],
        "dashboard_url": "",
        "dashboard_path": "",
        "dashboard": "",
        "insight": "",
        "code": "",
        "reasoning": "",
        "python_output": "",
    }

    for step in getattr(agent_response, "thought_process", []):
        if getattr(step, "action", None):
            result = getattr(step.observation, "result", None)
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except Exception:
                    result = None
            if not isinstance(result, dict):
                continue

            for key, value in result.items():
                if key in ["image_paths", "plot_paths"]:
                    if isinstance(value, list):
                        merged[key].extend(value)
                elif key in ["dashboard_url", "dashboard_path", "dashboard"]:
                    if value and not merged[key]:
                        merged[key] = value
                elif key in ["llm_response", "retrieved_context", "context_used", "insight", "code", "reasoning", "python_output"]:
                    if isinstance(value, str) and value.strip():
                        merged[key] = value
                else:
                    # Preserve other useful metadata if needed
                    if key not in merged:
                        merged[key] = value

    merged["image_paths"] = list(dict.fromkeys(merged["image_paths"]))
    merged["plot_paths"] = list(dict.fromkeys(merged["plot_paths"]))
    return merged


class DataLensAppState:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.qdrant_store = QdrantVectorStore(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            text_dim=self.embedding_service.text_dim,
            image_dim=self.embedding_service.image_dim,
        )
        self.rag_tool = DataLensRAGTool(
            qdrant_store=self.qdrant_store,
            embeddings=self.embedding_service,
        )
        self.csv_tool = CsvAnalystTool()
        self.dashboard_tool = DashboardGeneratorTool()
        self.plot_tool = PlotGeneratorTool()
        self.ares_tool = AresInternetTool()
        self.slide_tool = SlideGenerationTool()
        self.yfinance_tool = YFinanceTool()
        self.agent = ReactAgent(
            model=create_model(
                provider="litellm",
                model_name=GEMINI_MODEL,
                api_key=GEMINI_API_KEY,
                litellm_provider="gemini",
                max_tokens=8192,
            ),
            tools=[
                self.rag_tool,
                self.csv_tool,
                self.dashboard_tool,
                self.plot_tool,
                self.ares_tool,
                self.slide_tool,
                self.yfinance_tool,
            ],
            custom_system_prompt=(
            "You are DataLens, an intelligent document and data assistant.\n\n"
           """ ITERATION STRATEGY:
            - Use the minimum iterations needed — stop as soon as you have a confident, complete answer.
            - Simple questions  → 1–2 iterations max.
            - Medium tasks (multi-step analysis, dashboard + analysis) → use under 10 iterations.
            - Complex tasks (cross-tool, multi-source, full report) → up to 10 iterations.
            - Never pad iterations. If you have the answer, return it immediately.
            - If you hit the iteration limit i.e 19th iterationwithout a full answer, return the best partial answer you have — never return nothing.
           """
            "Use the available tools to answer the user's query as completely and accurately as possible.\n"
            "You may call multiple tools across iterations if needed — never limit yourself to one if more would help.\n\n"
            "Tool guide:\n"
            "- csv_analyst → CSV files, columns, rows, statistics, numeric analysis\n"
            "- datalens_rag → PDF documents, text content, images inside documents\n"
            "- dashboard_generator → creating interactive HTML dashboards from any data, give this tool as much details as possible because this is for reporting and insights generation\n"
            "- plot_generator → generating plots and visualizations from any dataset\n"
            "- ares_tool → live web data, news, anything not in uploaded files\n"
            "- slide_generation_tool → creating presentations or slide summaries\n"
            "- yfinance_tool → stock prices, financial market data, ticker symbols\n\n"
            "Routing rules:\n"
            "- If the query is clearly about a CSV or data → use csv_analyst\n"
            "- If the query is clearly about a document or PDF → use datalens_rag\n"
            "- If the query is ambiguous or could relate to both uploaded files → use both csv_analyst and datalens_rag and combine the results\n"
            "- If the query needs external information → use ares_tool\n\n"
            "All tools accept: {\"session_id\": \"<session_id>\", \"query\": \"<user question>\"}"
            "Donot add any paths in final response. The response should bever contain any path but normal natural answer."
        ),
            max_iterations=20,
        )

state = DataLensAppState()


def build_session_folder(session_id: str, file_type: str) -> Path:
    session_folder = TEMP_ROOT / session_id / file_type
    session_folder.mkdir(parents=True, exist_ok=True)
    return session_folder


def process_pdf_upload(session_id: str, pdf_path: Path) -> None:
    text_pages = extract_text_from_pdf(str(pdf_path))
    content_path = pdf_path.parent / "content.txt"
    save_text_content(text_pages, str(content_path))

    image_folder = pdf_path.parent / "images"
    image_folder.mkdir(parents=True, exist_ok=True)
    image_files = extract_images_from_pdf(str(pdf_path), str(image_folder))

    text_documents = build_text_documents(text_pages, "pdf", session_id)
    for idx, doc in enumerate(text_documents):
        doc["embedding"] = state.embedding_service.get_text_embeddings(doc["content"])

    state.qdrant_store.upload_text_points(session_id, text_documents)

    if image_files:
        relative_image_paths = []
        for path in image_files:
            rel_path = Path(path).relative_to(TEMP_ROOT / session_id)
            relative_image_paths.append(str(rel_path).replace("\\", "/"))

        embeddings = state.embedding_service.embed_image_list(image_files)
        state.qdrant_store.upload_image_points(session_id, relative_image_paths, embeddings)


def process_csv_upload(session_id: str, csv_path: Path) -> None:
    raw_text = extract_csv_text(str(csv_path))
    content_path = csv_path.parent / "content.txt"
    save_text_content(raw_text, str(content_path))

    text_documents = build_text_documents([raw_text], "csv", session_id)
    for doc in text_documents:
        doc["embedding"] = state.embedding_service.get_text_embeddings(doc["content"])

    state.qdrant_store.upload_text_points(session_id, text_documents)

    metadata_path = TEMP_ROOT / session_id / "json" / f"{csv_path.stem}.json"
    save_csv_metadata(str(csv_path), str(metadata_path))


def _count_existing_uploads(session_id: str) -> tuple[int, int]:
    pdf_folder = TEMP_ROOT / session_id / "pdf"
    csv_folder = TEMP_ROOT / session_id / "csv"
    existing_pdfs = len(list(pdf_folder.glob("*.pdf"))) if pdf_folder.exists() else 0
    existing_csv = len(list(csv_folder.glob("*.csv"))) if csv_folder.exists() else 0
    return existing_pdfs, existing_csv


@app.post("/api/upload")
async def upload_file(files: List[UploadFile] = File(...), session_id: str | None = Form(default=None)):
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded.")

    session_id = session_id or str(uuid.uuid4())
    state.qdrant_store.ensure_session_collections(session_id)

    pdf_files = [file for file in files if Path(file.filename).suffix.lower() in PDF_EXTENSIONS]
    csv_files = [file for file in files if Path(file.filename).suffix.lower() in CSV_EXTENSIONS]
    rejected = [file.filename for file in files if Path(file.filename).suffix.lower() not in PDF_EXTENSIONS | CSV_EXTENSIONS]

    if rejected:
        raise HTTPException(status_code=400, detail="Only PDF and CSV uploads are supported.")
    if len(pdf_files) > 2 or len(csv_files) > 1:
        raise HTTPException(status_code=400, detail="Upload limits exceeded: maximum 2 PDFs and 1 CSV allowed.")

    existing_pdfs, existing_csv = _count_existing_uploads(session_id)
    if existing_pdfs + len(pdf_files) > 2 or existing_csv + len(csv_files) > 1:
        raise HTTPException(
            status_code=400,
            detail="Upload limits exceeded for this session: maximum 2 PDFs and 1 CSV allowed.",
        )

    uploaded_names = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        file_contents = await file.read()

        if suffix in PDF_EXTENSIONS:
            pdf_folder = build_session_folder(session_id, "pdf")
            pdf_path = pdf_folder / file.filename
            pdf_path.write_bytes(file_contents)
            process_pdf_upload(session_id, pdf_path)
            uploaded_names.append(file.filename)
        elif suffix in CSV_EXTENSIONS:
            csv_folder = build_session_folder(session_id, "csv")
            csv_path = csv_folder / file.filename
            csv_path.write_bytes(file_contents)
            process_csv_upload(session_id, csv_path)
            uploaded_names.append(file.filename)

    return JSONResponse({"session_id": session_id, "uploaded_files": uploaded_names})


@app.get("/api/chat")
async def chat(session_id: str, query: str):
    if not session_id or not query:
        raise HTTPException(status_code=400, detail="session_id and query are required.")

    agent_query = (
        "You are a DataLens assistant that chooses the right tool for the question.\n"
        "Use csv_analyst for CSV dataset analysis, numeric questions, column names, and statistics.\n"
        "Use datalens_rag for PDF document text, images, and document retrieval questions.\n"
        "Return the action input as a JSON object with session_id and query.\n"
        f"session_id: {session_id}\n"
        f"query: {query}"
    )
    agent_response = state.agent.run(agent_query)
    tool_output = extract_tool_output(agent_response)
    if not tool_output.get("retrieved_context") and not tool_output.get("image_paths") and not tool_output.get("insight") and not tool_output.get("plot_paths") and not tool_output.get("dashboard_url") and not tool_output.get("code") and not tool_output.get("python_output"):
        tool_output = state.rag_tool.run({"session_id": session_id, "query": query})
    final_answer = getattr(agent_response, "final_answer", "") or tool_output.get("llm_response", "") or tool_output.get("insight", "")

    def build_sse_event(event_name: str, data: str) -> str:
        lines = data.splitlines()
        if not lines:
            return f"event: {event_name}\ndata:\n\n"

        event_lines = [f"event: {event_name}"]
        for line in lines:
            event_lines.append(f"data:{line}")
        if data.endswith("\n"):
            event_lines.append("data:")
        return "\n".join(event_lines) + "\n\n"

    async def event_generator() -> AsyncGenerator[str, None]:
        for chunk in chunk_text(final_answer):
            yield build_sse_event("token", chunk)

        yield build_sse_event("metadata", json.dumps(tool_output))
        yield "event: done\ndata:done\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
async def root():
    return {"message": "DataLens backend is running. Serve frontend from /frontend/index.html."}
