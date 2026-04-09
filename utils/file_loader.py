import os
import logging
from langchain_community.document_loaders import TextLoader
from utils.config import (
    EXCLUDED_DIRS,
    EXCLUDED_EXTENSIONS,
    EXCLUDED_FILENAMES,
    EXCLUDED_SECRET_EXTENSIONS,
    ENABLE_SECRET_SCANNING,
    SKIP_FILES_WITH_SECRETS,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)

_secrets_guard = None

def _get_secrets_guard():
    """Returns a cached Guard instance using only Guardrails AI for secret checking."""
    global _secrets_guard
    if _secrets_guard is None:
        try:
            from guardrails.validators import Validator, register_validator, PassResult, FailResult
            from guardrails import Guard

            @register_validator(name="simple_secret_check", data_type="string")
            class SimpleSecretCheck(Validator):
                def validate(self, value: str, metadata: dict = None):
                    
                    patterns = [
                        "sk-",         
                        "PRIVATE_KEY", 
                        "AWS_ACCESS_KEY", 
                        "AWS_SECRET_KEY",
                        "JWT"
                    ]
                    for p in patterns:
                        if p in value:
                            return FailResult(error_message=f"Secret detected: {p}")
                    return PassResult()

            _secrets_guard = Guard().use(SimpleSecretCheck(on_fail="exception"))
            logger.info("[GuardRail] Simple local secrets validator loaded successfully.")
        except Exception as e:
            logger.warning(f"[GuardRail] Could not initialize simple secrets guard: {e}")
            _secrets_guard = None

    return _secrets_guard

def _has_secrets(file_path: str) -> bool:
    """
    Scan file content for simple secrets using only Guardrails AI.
    Returns True if any secrets are detected, False otherwise.
    """
    guard = _get_secrets_guard()
    if guard is None:
        return False

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return False

        guard.validate(content)
        return False

    except Exception as e:
        error_msg = str(e).lower()
        if "secret" in error_msg or "fail" in error_msg:
            return True
        logger.warning(f"[GuardRail] Unexpected error scanning '{file_path}': {e}")
        return False

def _is_safe_path(base_path: str, file_path: str) -> bool:
    """Ensures file_path is strictly inside base_path, preventing directory traversal."""
    real_base = os.path.realpath(os.path.abspath(base_path))
    real_file = os.path.realpath(os.path.abspath(file_path))
    return real_file.startswith(real_base + os.sep)

def load_files(repo_path: str):
    """
    Loads all eligible text files from the cloned repository.

    Guardrail layers applied in order:
      1. Path safety  — repo_path must be inside UPLOAD_DIR.
      2. Filename/ext blocklist — skips known sensitive filenames & extensions.
      3. Per-file path check — each file must be a child of repo_path.
      4. Secret scanning (simplified Guardrails-only) — skips files containing secrets.
    """
    documents = []

    upload_dir_abs = os.path.realpath(os.path.abspath(str(UPLOAD_DIR)))
    repo_path_abs  = os.path.realpath(os.path.abspath(repo_path))
    if not repo_path_abs.startswith(upload_dir_abs):
        logger.error(
            f"[GuardRail] Path traversal blocked! '{repo_path_abs}' is outside "
            f"the allowed UPLOAD_DIR '{upload_dir_abs}'."
        )
        return []

    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        if any(ex_dir in root.split(os.sep) for ex_dir in EXCLUDED_DIRS):
            continue

        for file in files:
            filename_lower = file.lower()
            _, ext = os.path.splitext(filename_lower)

            if ext in EXCLUDED_EXTENSIONS:
                continue

            if ext in EXCLUDED_SECRET_EXTENSIONS:
                logger.info(f"[GuardRail] Blocked sensitive extension: {file}")
                continue

            if filename_lower in EXCLUDED_FILENAMES:
                logger.info(f"[GuardRail] Blocked sensitive filename: {file}")
                continue

            path = os.path.join(root, file)

            if not _is_safe_path(repo_path, path):
                logger.warning(f"[GuardRail] Path traversal blocked: {path}")
                continue

            if ENABLE_SECRET_SCANNING and _has_secrets(path):
                rel = os.path.relpath(path, repo_path)
                if SKIP_FILES_WITH_SECRETS:
                    logger.warning(f"[GuardRail] Secret detected — skipping: {rel}")
                    continue
                else:
                    logger.warning(f"[GuardRail] Secret detected (indexing anyway): {rel}")

            try:
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()

                for d in docs:
                    d.metadata["source"] = os.path.relpath(path, repo_path)
                    d.metadata["full_path"] = path

                documents.extend(docs)

            except (UnicodeDecodeError, Exception):
                pass

    return documents