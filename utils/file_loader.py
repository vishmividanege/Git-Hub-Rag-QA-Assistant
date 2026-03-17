# utils/file_loader.py
import os
from langchain_community.document_loaders import TextLoader
from utils.config import SUPPORTED_EXTENSIONS

def load_files(repo_path):
    documents = []

    for root, dirs, files in os.walk(repo_path):
        # Skip .git, node_modules, and other common build/test dirs
        if any(idx in root for idx in [".git", "node_modules", "__pycache__", "dist", "build", "venv"]):
            continue

        for file in files:
            if file.endswith(SUPPORTED_EXTENSIONS):
                path = os.path.join(root, file)
                try:
                    loader = TextLoader(path, encoding="utf-8")
                    docs = loader.load()

                    for d in docs:
                        d.metadata["source"] = os.path.relpath(path, repo_path)
                        d.metadata["full_path"] = path

                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

    return documents
