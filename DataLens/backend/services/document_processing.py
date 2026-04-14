import csv
import io
import json
import os
import uuid
from pathlib import Path
from typing import List, Any

import numpy as np
import pandas as pd
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

from ..config import TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE


def extract_images_from_pdf(pdf_path: str, images_folder: str) -> List[str]:
    pdf_document = fitz.open(pdf_path)
    os.makedirs(images_folder, exist_ok=True)
    saved_images = []

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        images = page.get_images(full=True)

        for image_index, image_info in enumerate(images):
            xref = image_info[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_filename = f"img_{len(saved_images) + 1}.png"
            image_path = os.path.join(images_folder, image_filename)

            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image.save(image_path, format="PNG")
            except Exception:
                with open(image_path, "wb") as image_file:
                    image_file.write(image_bytes)

            saved_images.append(image_path)

    pdf_document.close()
    return saved_images


def extract_text_from_pdf(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return texts


def extract_csv_text(csv_path: str) -> str:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            rows.append(", ".join(row))
    return "\n".join(rows)


def _make_json_safe(value: Any) -> Any:
    # Handle arrays/lists first before pd.isna() check to avoid ambiguous truth value error
    if isinstance(value, (np.ndarray, pd.Series)):
        return [
            _make_json_safe(v) for v in (value.tolist() if hasattr(value, "tolist") else list(value))
        ]
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    # Check for NaT (Not a Time) for numpy datetime types
    if isinstance(value, np.datetime64) and pd.isna(value):
        return None
    # Only check pd.isna() for scalar values to avoid ambiguous array truth values
    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        # pd.isna() can raise ValueError for array-like objects
        pass
    return value


def build_csv_metadata(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    metadata = {
        "file_name": Path(csv_path).name,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": [],
        "sample_rows": [],
    }

    for col in df.columns:
        series = df[col]
        col_stats = {
            "name": col,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_stats.update(
                {
                    "min": _make_json_safe(series.min()),
                    "max": _make_json_safe(series.max()),
                    "mean": _make_json_safe(series.mean()),
                    "median": _make_json_safe(series.median()),
                    "std": _make_json_safe(series.std()),
                }
            )
        else:
            mode_values = series.mode()
            col_stats.update(
                {
                    "top": _make_json_safe(mode_values.iloc[0] if len(mode_values) else None),
                    "sample_values": _make_json_safe(series.dropna().head(10).tolist()),
                }
            )

        metadata["columns"].append(col_stats)

    metadata["sample_rows"] = [
        {str(col): _make_json_safe(value) for col, value in row.items()}
        for row in df.head(5).fillna("").to_dict(orient="records")
    ]

    return metadata


def save_csv_metadata(csv_path: str, metadata_path: str) -> None:
    metadata = build_csv_metadata(csv_path)
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def save_text_content(raw_texts: List[str] | str, content_path: str) -> None:
    os.makedirs(os.path.dirname(content_path), exist_ok=True)
    if isinstance(raw_texts, list):
        raw_texts = "\n\n".join(raw_texts)

    with open(content_path, "w", encoding="utf-8") as output:
        output.write(raw_texts)


def build_text_documents(raw_texts: List[str], source_label: str, session_id: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", ".", ",", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""],
    )

    documents = text_splitter.create_documents(raw_texts)
    output_documents = []

    for document in documents:
        chunk_id = str(uuid.uuid4())
        output_documents.append(
            {
                "id": chunk_id,
                "content": document.page_content,
                "metadata": {
                    "source": source_label,
                    "session_id": session_id,
                    "uuid": chunk_id,
                },
            }
        )

    return output_documents


def build_csv_documents(csv_path: str, session_id: str):
    text = extract_csv_text(csv_path)
    save_text_content(text, str(Path(csv_path).with_suffix(".txt")))
    return build_text_documents([text], "csv", session_id)
