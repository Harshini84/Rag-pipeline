import os
from turtle import st
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain.schema import Document
import csv
try:
    import pandas as pd
except Exception:
    pd = None
from .config import SPECIFIC_FILES


def load_langchain(paths):
    docs = []
    loaded_sources = []

    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: File not found - {path}")
            continue
        print(f"Loading: {path}")
        loaded_sources.append(path)

        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
            loaded = loader.load()
            # Manually ensure source is in metadata
            for doc in loaded:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['source'] = path
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
            loaded = loader.load()
            for doc in loaded:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['source'] = path
        elif path.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['source'] = path
        elif path.endswith(".csv"):
            # try pandas first for robust parsing, else fallback to csv
            rows = []
            if pd is not None:
                try:
                    df = pd.read_csv(path, dtype=str, keep_default_na=False)
                    for _, r in df.iterrows():
                        rows.append(Document(page_content=", ".join([f"{k}: {v}" for k, v in r.items()]), metadata={"source": path}))
                except Exception:
                    pd = None
            if pd is None:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    for i, r in enumerate(reader):
                        if headers:
                            content = ", ".join([f"{h}: {v}" for h, v in zip(headers, r)])
                        else:
                            content = ", ".join(r)
                        rows.append(Document(page_content=content, metadata={"source": path, "row": i}))
            loaded = rows
        elif path.endswith(".xls") or path.endswith(".xlsx"):
            rows = []
            if pd is not None:
                try:
                    df = pd.read_excel(path, dtype=str, engine="openpyxl")
                    for _, r in df.iterrows():
                        rows.append(Document(page_content=", ".join([f"{k}: {v}" for k, v in r.items()]), metadata={"source": path}))
                    loaded = rows
                except Exception:
                    # if pandas/openpyxl not available or failed, skip
                    loaded = []
            else:
                loaded = []
        else:
            continue
        docs.extend(loaded)
        print(f"  Loaded {len(loaded)} pages/sections")

    if not docs:
        print("No documents were loaded!")
        return {'chunks': [], 'sources': []}

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    chunks = splitter.split_documents(docs)
    #print(chunks)
    print(f"Total chunks created: {len(chunks)}")
    return {'chunks': chunks, 'sources': loaded_sources}


def load_documents_from_directory(directory: str) -> List:
    paths = []

    # Use SPECIFIC_FILES only when no directory was provided
    if (not directory or directory is None) and SPECIFIC_FILES:
        files = [SPECIFIC_FILES] if isinstance(SPECIFIC_FILES, str) else SPECIFIC_FILES
        print(f"Using ONLY files from config: {files}")
        return load_langchain(files)

    # Otherwise add files from directory
    if directory and os.path.exists(directory):
        files_in_dir = os.listdir(directory)
        print(f"Directory '{directory}' contains: {files_in_dir}")
        for filename in files_in_dir:
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.pdf', '.docx', '.txt', '.csv', '.xls', '.xlsx')):
                paths.append(file_path)

    print(f"Total files to load: {len(paths)}")
    chunks_and_sources = load_langchain(paths)
    return chunks_and_sources
