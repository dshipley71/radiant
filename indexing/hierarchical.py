"""
Hierarchical indexing pipeline with improved error handling, validation, and resource management.

Key improvements:
- Specific exception handling with logging
- Configuration validation
- Resource cleanup with context managers
- Progress indicators
- Memory-efficient streaming
- Enhanced security checks
- Config alignment with config.fast.yaml (vectorstore/models/indexing/retrieval/advanced)
"""

from __future__ import annotations

# ---------- Fast env knobs ----------
import os as _os

_os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
_os.environ.setdefault("POSTHOG_DISABLED", "true")
_os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
_os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "false")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
_os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch as _torch
_torch.set_float32_matmul_precision("high")
if _torch.cuda.is_available():
    _torch.backends.cuda.matmul.allow_tf32 = True

# ---------- Standard imports ----------
import os
import sys
import json
import time
import yaml
import glob
import mimetypes
import hashlib
import numbers
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Email parsing
from email import policy
from email.parser import BytesParser

# Concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed

# Progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[WARN] tqdm not available. Install with: pip install tqdm")

# Unstructured parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.image import partition_image
from unstructured.partition.auto import partition as partition_auto

# Haystack core
from haystack import Document
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils.device import ComponentDevice

# Chroma document store
from haystack_integrations.document_stores.chroma import ChromaDocumentStore

# Transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForVision2Seq,
    CLIPProcessor,
    CLIPModel,
)

from PIL import Image
from html.parser import HTMLParser

# Optional OpenAI client
try:
    from openai import OpenAI as _OpenAIClient
except ImportError:
    _OpenAIClient = None

# Suppress noisy warnings from deps
import warnings
import requests

logging.getLogger(requests.packages.urllib3.__package__).setLevel(logging.ERROR)
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

requests.packages.urllib3.disable_warnings()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---------- Logging with ANSI colors (for our logger only) ----------

class ColorFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\033[32m",     # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
    }
    RESET = "\033[0m"

    def format(self, record):
        # base message (no level in front)
        msg = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""
        return f"{color}[{record.levelname}]{reset} {msg}"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)
# Avoid double-printing through root logger ("INFO:__main__" spam)
logger.propagate = False

# ---------------------------
# Config dataclasses (aligned with config.fast.yaml)
# ---------------------------

@dataclass
class VectorStoreConfig:
    # config: vectorstore.persist_path / collection_name
    persist_path: str = "./chroma_db_2"
    collection_name: Optional[str] = "leaves"
    # backend kept for future extensibility; not in config but harmless
    backend: str = "chroma"


@dataclass
class ParentVectorStoreConfig:
    # config: parent_vectorstore.persist_path / collection_name
    persist_path: str = "./chroma_db_parents_2"
    collection_name: Optional[str] = "parents"
    backend: str = "chroma"


@dataclass
class ModelConfig:
    # config.models.*
    embedder_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    e5_prefix: str = ""
    embedder_device: str = "cuda"

    use_local: bool = True
    llm_model: str = "Qwen/Qwen3-8B"
    llm_max_new_tokens: int = 128
    llm_temperature: float = 0.2

    api_base: Optional[str] = None
    api_key: Optional[str] = None
    api_model: Optional[str] = None


@dataclass
class IndexingConfig:
    # security limits
    max_file_size_mb: int = 500
    max_attachment_size_mb: int = 50
    max_email_recursion: int = 5

    # corpus
    corpus_dir: Optional[str] = "./corpus"
    files: List[str] = field(default_factory=list)
    split_by: str = "sentence"

    # sentence split
    parent_sentences: int = 10
    leaf_sentences: int = 5
    sentence_overlap: int = 1

    # word split
    chunk_sizes: List[int] = field(default_factory=lambda: [2000, 600])
    chunk_overlaps: List[int] = field(default_factory=lambda: [80, 40])

    # page split
    parent_pages: int = 4
    leaf_pages: int = 1
    page_overlap: int = 0

    # passage split
    parent_passages: int = 4
    leaf_passages: int = 2
    passage_overlap: int = 0

    # summaries
    store_summaries: bool = False
    persist_meta_path: str = "./run_meta"

    # limits
    max_files: int = 0
    max_pdf_pages: int = 0

    # OCR & languages
    ocr_fallback: bool = True
    languages: List[str] = field(
        default_factory=lambda: ["eng", "spa", "rus", "chi_sim", "chi_tra"]
    )
    num_workers: int = 0

    # vision
    enable_vision_captions: bool = False
    vision_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    vision_backend: str = "causal"
    vision_prompt: str = (
        "Describe the image in detail in 2-3 sentences. Do not include the user prompt "
        "in the response. Provide only the answer."
    )
    vision_max_new_tokens: int = 128
    clip_labels: List[str] = field(
        default_factory=lambda: [
            "document",
            "poster",
            "diagram",
            "handwritten",
            "invoice",
            "spreadsheet",
            "map",
            "building",
            "cat",
            "dog",
            "person",
        ]
    )

    # summarization knobs (not heavily used in indexer beyond store_summaries)
    summarize_leaves: bool = False
    summarize_parents: bool = False
    summarizer_batch_size: int = 16
    summarizer_max_input_tokens: int = 512
    summarizer_concurrency: int = 0
    summarize_only_topk_leaves: int = 8

    # parent embedding toggle
    embed_parents: bool = True

    # OCR heuristics (used in _parse_pdf_into_page_docs)
    ocr_min_chars_total: int = 2000
    ocr_min_pages_text_ratio: float = 0.15


@dataclass
class RetrievalHints:
    """
    These are primarily for retrieval_automerging.py, but we declare them
    so hier_indexer.py doesn't spam warnings for unknown keys.
    """

    # Chroma paths (used by retrieval)
    leaf_chroma_path: str = "./chroma_db"
    leaf_collection: str = "leaves"
    parent_chroma_path: str = "./chroma_db_parents"
    parent_collection: str = "parents"
    leaf_only: bool = True
    parent_sidecar_path: str = "./run_meta/parents_sidecar.json"

    # Retrieval embedder (query side)
    embedder_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    embedder_device: str = "cuda"

    # Core retrieval knobs
    leaf_top_k: int = 50
    enable_hybrid: bool = True
    bm25_top_k: int = 50

    # Auto-merge thresholds
    merge_threshold: float = 0.5
    automerge_threshold: float = 0.5

    # Reranking
    enable_rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L12-v2"
    rerank_top_k: int = 50
    rerank_device: str = "cuda"
    predownload_reranker: bool = True

    # Display / normalization
    show_n_others: int = 20
    normalize_query: bool = True

    # PRF
    prf_enable: bool = True
    prf_docs: int = 8
    prf_terms: int = 6

    # QE (nested config stored as dict from YAML)
    query_expansion: Dict[str, Any] = field(default_factory=dict)

    # RAG generation (retrieval_automerging.py)
    gen_max_new_tokens: int = 512
    gen_temperature: float = 0.05
    context_max_chars: int = 3600

    # Optional E5 prefix for queries
    e5_query_prefix: str = ""


@dataclass
class AdvancedConfig:
    batch_size_docs: int = 1
    parent_batch_size_docs: int = 1
    warmup_embedder: bool = True
    normalize_embeddings: bool = True

    # New: per-document batching for the indexer
    # 0 = legacy mode (parse all docs → split → embed)
    # N > 0 = process N files at a time
    doc_batch_size: int = 50


@dataclass
class PipelineConfig:
    vectorstore: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    parent_vectorstore: ParentVectorStoreConfig = field(default_factory=ParentVectorStoreConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    retrieval: RetrievalHints = field(default_factory=RetrievalHints)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)


# ---------------------------
# Configuration validation
# ---------------------------

class ConfigValidationError(ValueError):
    """Raised when configuration is invalid."""
    pass


def validate_config(cfg: PipelineConfig) -> None:
    """Validate configuration with clear error messages."""
    # Vision settings
    if cfg.indexing.enable_vision_captions and not cfg.indexing.vision_model:
        raise ConfigValidationError("indexing.enable_vision_captions=True requires vision_model to be set")

    # Remote LLM settings
    if not cfg.models.use_local:
        if not cfg.models.api_base:
            raise ConfigValidationError("models.use_local=False requires models.api_base")
        if not cfg.models.api_key:
            raise ConfigValidationError("models.use_local=False requires models.api_key")
        if _OpenAIClient is None:
            raise ConfigValidationError(
                "models.use_local=False requires 'openai' package. Install with: pip install openai"
            )

    # Splitting parameters
    mode = cfg.indexing.split_by.lower()
    if mode == "sentence":
        if cfg.indexing.parent_sentences <= cfg.indexing.leaf_sentences:
            raise ConfigValidationError("indexing.parent_sentences must exceed leaf_sentences")
        if cfg.indexing.sentence_overlap >= cfg.indexing.leaf_sentences:
            raise ConfigValidationError("indexing.sentence_overlap must be less than leaf_sentences")
    elif mode == "word":
        if not cfg.indexing.chunk_sizes or len(cfg.indexing.chunk_sizes) < 2:
            raise ConfigValidationError("indexing.chunk_sizes must have at least 2 values for word mode")
        if cfg.indexing.chunk_sizes[0] <= cfg.indexing.chunk_sizes[1]:
            raise ConfigValidationError("Parent chunk size must exceed leaf chunk size")
    elif mode == "page":
        if cfg.indexing.parent_pages <= cfg.indexing.leaf_pages:
            raise ConfigValidationError("indexing.parent_pages must exceed leaf_pages")
    elif mode == "passage":
        if cfg.indexing.parent_passages <= cfg.indexing.leaf_passages:
            raise ConfigValidationError("indexing.parent_passages must exceed leaf_passages")
    else:
        raise ConfigValidationError(
            f"Invalid indexing.split_by value: {mode}. Must be one of: sentence, word, page, passage"
        )

    # Corpus / files
    if cfg.indexing.corpus_dir and not os.path.isdir(cfg.indexing.corpus_dir):
        if cfg.indexing.files:
            logger.warning(
                f"indexing.corpus_dir '{cfg.indexing.corpus_dir}' not found, using explicit files only"
            )
        else:
            raise ConfigValidationError(
                f"indexing.corpus_dir '{cfg.indexing.corpus_dir}' does not exist and no explicit files provided"
            )

    # Security limits
    if cfg.indexing.max_file_size_mb < 0:
        raise ConfigValidationError("indexing.max_file_size_mb must be non-negative")
    if cfg.indexing.max_attachment_size_mb < 0:
        raise ConfigValidationError("indexing.max_attachment_size_mb must be non-negative")
    if cfg.indexing.max_email_recursion < 1:
        raise ConfigValidationError("indexing.max_email_recursion must be at least 1")

    logger.info("Configuration validation passed")


# ---------------------------
# Utilities
# ---------------------------

def _json_dumps(obj: Any) -> str:
    """Safe UTF-8 JSON dump without ASCII escaping."""
    return json.dumps(obj, ensure_ascii=False)


def _check_file_size(path: str, max_size_mb: int) -> None:
    """Validate file size before processing."""
    if max_size_mb <= 0:
        return

    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValueError(f"File {path} exceeds max size: {size_mb:.1f}MB > {max_size_mb}MB")
    except OSError as e:
        raise IOError(f"Cannot check size of {path}: {e}")


def _sanitize_path(path: str) -> str:
    """Validate and normalize file path to prevent traversal attacks."""
    try:
        normalized = os.path.normpath(path)
        # Check for path traversal
        if ".." in normalized or normalized.startswith("/"):
            if not os.path.isabs(path):
                raise ValueError(f"Suspicious path detected: {path}")
        return normalized
    except Exception as e:
        raise ValueError(f"Invalid path '{path}': {e}")


class _LinkExtractor(HTMLParser):
    """Extract links from HTML."""
    def __init__(self):
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._in_a = False
        self._current_href: Optional[str] = None
        self._current_text: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._in_a = True
            self._current_href = None
            for k, v in attrs:
                if k.lower() == "href":
                    self._current_href = v
                    break
            self._current_text = []

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._in_a:
            text = "".join(self._current_text).strip()
            if self._current_href:
                self.links.append((self._current_href, text))
            self._in_a = False
            self._current_href = None
            self._current_text = []

    def handle_data(self, data):
        if self._in_a and data:
            self._current_text.append(data)


def load_config(path: Optional[str]) -> PipelineConfig:
    """Load and validate YAML config (config.fast.yaml)."""
    raw: Dict[str, Any] = {}
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigValidationError(f"Invalid YAML in {path}: {e}")
        except IOError as e:
            raise ConfigValidationError(f"Cannot read config file {path}: {e}")

    def merge(dc, raw_section):
        if raw_section and isinstance(raw_section, dict):
            for k, v in raw_section.items():
                if hasattr(dc, k):
                    setattr(dc, k, v)
                else:
                    # Silently ignore truly unknown keys for this dataclass (hier_indexer
                    # doesn't need to warn about retrieval- or agentic-only fields).
                    # logger.debug(f"Unknown config key '{k}' in section {type(dc).__name__}, ignoring")
                    pass
        return dc

    cfg = PipelineConfig()
    merge(cfg.vectorstore, raw.get("vectorstore"))
    merge(cfg.parent_vectorstore, raw.get("parent_vectorstore"))
    merge(cfg.models, raw.get("models"))
    merge(cfg.indexing, raw.get("indexing"))
    merge(cfg.retrieval, raw.get("retrieval"))
    merge(cfg.advanced, raw.get("advanced"))
    # NOTE: 'agentic' section is intentionally ignored by the indexer

    # Backward-compat aliases for API base URL keys
    models_raw = raw.get("models") or {}
    if not cfg.models.api_base and isinstance(models_raw, dict):
        for alias in ("api_base_url", "base_url"):
            if models_raw.get(alias):
                cfg.models.api_base = models_raw[alias]
                break

    # Validate before returning
    validate_config(cfg)
    return cfg


def ensure_dirs(*paths):
    """Create directories with error handling."""
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
        except OSError as e:
            raise IOError(f"Cannot create directory {p}: {e}")


def list_corpus_files(corpus_dir: Optional[str], explicit_files: List[str], max_files: int = 0) -> List[str]:
    """Collect files with progress indication."""
    files: List[str] = []

    # Add explicit files
    if explicit_files:
        for f in explicit_files:
            try:
                sanitized = _sanitize_path(f)
                if os.path.isfile(sanitized) or str(f).startswith("http"):
                    files.append(sanitized)
            except ValueError as e:
                logger.warning(f"Skipping invalid path '{f}': {e}")

    # Scan corpus directory
    if corpus_dir and os.path.isdir(corpus_dir):
        patterns = [
            "**/*.pdf",
            "**/*.png",
            "**/*.jpg",
            "**/*.jpeg",
            "**/*.tif",
            "**/*.tiff",
            "**/*.txt",
            "**/*.docx",
            "**/*.pptx",
            "**/*.xlsx",
            "**/*.eml",
            "**/*.html",
            "**/*.htm",
        ]
        for pat in patterns:
            try:
                found = glob.glob(os.path.join(corpus_dir, pat), recursive=True)
                files.extend(found)
            except Exception as e:
                logger.warning(f"Error scanning pattern {pat}: {e}")

    # Dedupe while preserving order
    seen, out = set(), []
    for f in files:
        if f not in seen:
            out.append(f)
            seen.add(f)

    result = out[:max_files] if max_files and len(out) > max_files else out
    logger.info(f"Found {len(result)} files to index")
    return result


def export_docs_jsonl(docs: List[Document], path: str):
    """Write JSONL with error handling."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            for d in docs:
                emb = getattr(d, "embedding", None)
                emb_out = (
                    emb.tolist()
                    if hasattr(emb, "tolist")
                    else (list(emb) if isinstance(emb, (list, tuple)) else None)
                )
                rec = {
                    "id": d.id,
                    "content": d.content,
                    "meta": d.meta or {},
                    "relationships": getattr(d, "relationships", {}),
                    "embedding": emb_out,
                }
                f.write(_json_dumps(rec) + "\n")
        logger.info(f"Exported {len(docs)} documents to {path}")
    except IOError as e:
        raise IOError(f"Failed to write JSONL to {path}: {e}")


def export_parents_sidecar(parents: List[Document], path: str):
    """Save parent sidecar JSON with error handling."""
    try:
        payload = [
            {
                "id": d.id,
                "content": d.content,
                "meta": d.meta or {},
                "relationships": getattr(d, "relationships", {}),
            }
            for d in parents
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(parents)} parents to sidecar: {path}")
    except IOError as e:
        raise IOError(f"Failed to write sidecar to {path}: {e}")


def _device_for_embedder(mcfg: ModelConfig) -> Optional[ComponentDevice]:
    """Resolve ComponentDevice for embedder."""
    try:
        if mcfg.embedder_device.lower() == "cuda" and _torch.cuda.is_available():
            return ComponentDevice.from_str("cuda:0")
    except Exception as e:
        logger.warning(f"Failed to resolve CUDA device: {e}, falling back to CPU")
    return None


# ---------------------------
# Metadata sanitization
# ---------------------------

def _sanitize_doc_meta(doc: Document) -> Document:
    """
    Coerce doc.meta to Chroma-supported types with logging.
    Prevents: 'contains meta values of unsupported types' warnings.
    """
    meta = dict(doc.meta or {})
    dropped_keys = []

    for k, v in list(meta.items()):
        # Already safe
        if isinstance(v, (str, int, float, bool)):
            continue

        # Numeric types
        if isinstance(v, numbers.Integral):
            meta[k] = int(v)
            continue
        if isinstance(v, numbers.Real):
            meta[k] = float(v)
            continue

        # Remove None
        if v is None:
            meta.pop(k, None)
            dropped_keys.append(k)
            continue

        # Containers → JSON
        if isinstance(v, (list, dict, tuple, set)):
            try:
                meta[k] = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize meta['{k}']: {e}, converting to string")
                meta[k] = str(v)
            continue

        # Unknown → string
        meta[k] = str(v)

    # Special handling for known fields
    if "_split_overlap" in meta:
        try:
            meta["_split_overlap"] = int(meta["_split_overlap"])
        except (TypeError, ValueError):
            meta.pop("_split_overlap", None)
            dropped_keys.append("_split_overlap")

    if dropped_keys:
        logger.debug(f"Sanitized doc {doc.id}: dropped keys {dropped_keys}")

    doc.meta = meta
    return doc


def _sanitize_meta_for_all(docs: List[Document]) -> List[Document]:
    """Apply sanitization to all documents."""
    return [_sanitize_doc_meta(d) for d in (docs or [])]


# ---------------------------
# Language detection
# ---------------------------

_LANG_MAP_TO_TESS = {
    "en": "eng",
    "eng": "eng",
    "es": "spa",
    "spa": "spa",
    "fr": "fra",
    "fra": "fra",
    "de": "deu",
    "deu": "deu",
    "it": "ita",
    "ita": "ita",
    "pt": "por",
    "por": "por",
    "ar": "ara",
    "ara": "ara",
    "ru": "rus",
    "rus": "rus",
    "zh": "chi_sim",
    "zh-cn": "chi_sim",
    "zh-hans": "chi_sim",
    "zh-tw": "chi_tra",
    "zh-hant": "chi_tra",
}


def _has_cjk(text: str) -> bool:
    """Check for CJK characters."""
    for ch in text:
        code = ord(ch)
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0x20000 <= code <= 0x2A6DF):
            return True
    return False


def _dedupe_keep_order(seq: List[str]) -> List[str]:
    """Deduplicate while preserving order."""
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _detect_langs_from_text(text: str, max_out: int = 4) -> List[str]:
    """Detect languages with fallbacks."""
    text = (text or "").strip()
    if not text:
        return []

    langs: List[str] = []
    try:
        from langdetect import detect_langs

        det_sorted = sorted(detect_langs(text[:4000]), key=lambda x: x.prob, reverse=True)
        for d in det_sorted[:max_out]:
            code = d.lang.lower()
            mapped = _LANG_MAP_TO_TESS.get(code)
            if mapped:
                langs.append(mapped)
            if code.startswith("zh"):
                langs.extend(["chi_sim", "chi_tra"])
    except ImportError:
        logger.debug("langdetect not available, using heuristics")
    except Exception as e:
        logger.debug(f"Language detection failed: {e}")

    # Fallback heuristics
    if _has_cjk(text):
        langs.extend(["chi_sim", "chi_tra"])
    if "eng" not in langs:
        langs.append("eng")

    return _dedupe_keep_order(langs)[: max_out + 2]


# ---------------------------
# Summarizer with unified LLM routing
# ---------------------------

class LocalSummarizer:
    """
    Summarizer supporting local HF or remote OpenAI-compatible API.
    """

    def __init__(self, mcfg: ModelConfig):
        self.max_new = int(mcfg.llm_max_new_tokens)
        self.temp = float(mcfg.llm_temperature)
        self.use_local = bool(getattr(mcfg, "use_local", True))

        if self.use_local:
            self._init_local(mcfg)
        else:
            self._init_remote(mcfg)

    def _init_local(self, mcfg: ModelConfig):
        """Initialize local HF model."""
        self.model_id = mcfg.llm_model
        try:
            cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=False)
            self._is_seq2seq = bool(getattr(cfg, "is_encoder_decoder", False))

            dtype = _torch.bfloat16 if _torch.cuda.is_available() else _torch.float32
            self.tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

            if not self._is_seq2seq:
                try:
                    self.tok.padding_side = "left"
                except AttributeError:
                    pass

            if self._is_seq2seq:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_id, device_map="auto", torch_dtype=dtype, trust_remote_code=False
                )
                task = "text2text-generation"
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, device_map="auto", torch_dtype=dtype, trust_remote_code=False
                )
                try:
                    self.model.config.use_cache = False
                except AttributeError:
                    pass
                task = "text-generation"

            self.pipe = hf_pipeline(
                task,
                model=self.model,
                tokenizer=self.tok,
                max_new_tokens=self.max_new,
                do_sample=(self.temp > 0.0),
                temperature=self.temp,
            )
            self._client = None
            self._remote_model = None
            logger.info(f"Initialized local summarizer: {self.model_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize local summarizer '{self.model_id}': {e}")

    def _init_remote(self, mcfg: ModelConfig):
        """Initialize remote OpenAI-compatible client."""
        if _OpenAIClient is None:
            raise RuntimeError("openai package required for remote LLM. Install with: pip install openai")
        if not mcfg.api_base or not mcfg.api_key:
            raise RuntimeError("models.use_local=False requires models.api_base and models.api_key")

        try:
            self._client = _OpenAIClient(base_url=str(mcfg.api_base), api_key=str(mcfg.api_key))
            self._remote_model = mcfg.api_model or mcfg.llm_model
            self._is_seq2seq = False  # Not applicable for remote
            logger.info(f"Initialized remote summarizer: {self._remote_model} @ {mcfg.api_base}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize remote summarizer: {e}")

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Token-safe truncation (local only)."""
        if self.use_local and max_tokens > 0:
            try:
                ids = self.tok.encode(text, add_special_tokens=False)
                if len(ids) <= max_tokens:
                    return text
                return self.tok.decode(ids[:max_tokens], skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Token truncation failed: {e}, using character truncation")
                return text[: max_tokens * 4]  # Rough approximation
        return text

    def summarize_batch(self, texts: List[str]) -> List[str]:
        """Summarize texts in batch."""
        if not texts:
            return []

        if self.use_local:
            return self._summarize_batch_local(texts)
        else:
            return self._summarize_batch_remote(texts)

    def _summarize_batch_local(self, texts: List[str]) -> List[str]:
        """Local HF summarization."""
        try:
            payload = [
                (
                    f"Summarize the following passage in 1-3 concise sentences:\n\n{t}"
                    if self._is_seq2seq
                    else (
                        "You are a concise summarizer.\n"
                        "Summarize the following passage in 1-3 sentences.\n\n"
                        f"Passage:\n{t}\n\nSummary:"
                    )
                )
                for t in texts
            ]
            outs = self.pipe(payload, batch_size=max(1, len(texts)))
            results = []
            for out in outs:
                gen = out[0]["generated_text"] if isinstance(out, list) else out["generated_text"]
                if self._is_seq2seq:
                    results.append((gen or "").strip())
                else:
                    g = (gen or "")
                    idx = g.lower().rfind("summary:")
                    summary = g[idx + len("summary:") :].strip() if idx >= 0 else g
                    results.append(summary.strip().split("\n\n")[0][:1000])
            return results
        except Exception as e:
            logger.error(f"Local summarization failed: {e}")
            return ["" for _ in texts]

    def _summarize_batch_remote(self, texts: List[str]) -> List[str]:
        """Remote API summarization."""
        results = []
        for t in texts:
            try:
                prompt = "Summarize the following passage in 1–3 concise sentences:\n\n" + t
                resp = self._client.chat.completions.create(
                    model=self._remote_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temp,
                    max_tokens=self.max_new,
                )
                results.append((resp.choices[0].message.content or "").strip())
            except Exception as e:
                logger.warning(f"Remote summarization failed for one text: {e}")
                results.append("")
        return results


# ---------------------------
# Main indexer with improvements
# ---------------------------

class HierarchicalIndexer:
    """
    Hierarchical indexer with:
    - Improved error handling
    - Resource cleanup
    - Memory-efficient streaming
    - Progress indicators
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._cleanup_handlers = []  # Track resources for cleanup

        try:
            # Vector stores (aligned with vectorstore / parent_vectorstore config)
            self.leaf_store = ChromaDocumentStore(
                persist_path=cfg.vectorstore.persist_path,
                collection_name=cfg.vectorstore.collection_name,
            )
            self.parent_store = ChromaDocumentStore(
                persist_path=cfg.parent_vectorstore.persist_path,
                collection_name=cfg.parent_vectorstore.collection_name,
            )

            # Embedder
            device = _device_for_embedder(cfg.models)
            model_id = cfg.models.embedder_model
            is_jina = "jina-embeddings-" in (model_id or "")
            embed_prefix = (cfg.models.e5_prefix or "").strip()

            embedder_kwargs = dict(
                model=model_id,
                prefix=embed_prefix,
                suffix="",
                trust_remote_code=True,
            )

            if is_jina:
                # For jinaai/jina-embeddings-v4: prevent HF hub calls for "task", we just set default_task via model_kwargs
                embedder_kwargs["model_kwargs"] = {"default_task": "retrieval"}

            if isinstance(device, ComponentDevice):
                embedder_kwargs["device"] = device

            self.embedder = SentenceTransformersDocumentEmbedder(**embedder_kwargs)

            # Configure embedder
            try:
                self.embedder.normalize_embeddings = bool(cfg.advanced.normalize_embeddings)
            except AttributeError:
                pass

            self.embedder.prefix = getattr(self.embedder, "prefix", "") or ""
            self.embedder.suffix = getattr(self.embedder, "suffix", "") or ""

            try:
                self.embedder.batch_size = int(cfg.advanced.batch_size_docs or 1)
            except AttributeError:
                pass

            if cfg.advanced.warmup_embedder:
                logger.info("Warming up embedder...")
                self.embedder.warm_up()

            # Summarizer (optional, if indexing.store_summaries=true)
            self.summarizer = None
            if cfg.indexing.store_summaries:
                try:
                    self.summarizer = LocalSummarizer(cfg.models)
                except Exception as e:
                    logger.error(f"Failed to initialize summarizer: {e}")
                    raise

            # Splitter
            self.splitter = self._init_splitter(cfg.indexing)

            # I/O paths
            ensure_dirs(cfg.indexing.persist_meta_path)
            self.sidecar_path = os.path.join(cfg.indexing.persist_meta_path, "parents_sidecar.json")
            self.leaf_dump = os.path.join(cfg.indexing.persist_meta_path, "leaf_docs.jsonl")
            self.parent_dump = os.path.join(cfg.indexing.persist_meta_path, "parent_docs.jsonl")

            # Vision captioning
            self.vision_enabled = bool(cfg.indexing.enable_vision_captions)
            self._vision_model_id = cfg.indexing.vision_model
            self._vision_backend = (cfg.indexing.vision_backend or "blip").lower()
            self._vision_prompt = cfg.indexing.vision_prompt or "Describe this image."
            self._vision_max_new = int(cfg.indexing.vision_max_new_tokens or 64)
            self._clip_labels = list(cfg.indexing.clip_labels or [])

            # Vision models (lazy init)
            self._blip_pipe = None
            self._cap_model = None
            self._cap_processor = None
            self._clip_proc = None

            if self.vision_enabled:
                self._init_vision_backend()

            logger.info("Indexer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize indexer: {e}")
            self.cleanup()
            raise

    def _init_splitter(self, icfg: IndexingConfig) -> HierarchicalDocumentSplitter:
        """Initialize hierarchical splitter based on config."""
        mode = (icfg.split_by or "sentence").lower()

        if mode == "sentence":
            return HierarchicalDocumentSplitter(
                block_sizes=[
                    max(1, int(icfg.parent_sentences or 10)),
                    max(1, int(icfg.leaf_sentences or 5)),
                ],
                split_overlap=max(0, int(icfg.sentence_overlap or 0)),
                split_by="sentence",
            )
        elif mode == "word":
            sizes = list(icfg.chunk_sizes or [2000, 600])
            overlaps = list(icfg.chunk_overlaps or [80, 40])
            return HierarchicalDocumentSplitter(
                block_sizes=sizes,
                split_overlap=int(max(overlaps) if overlaps else 80),
                split_by="word",
            )
        elif mode == "page":
            return HierarchicalDocumentSplitter(
                block_sizes=[
                    max(1, int(icfg.parent_pages or 4)),
                    max(1, int(icfg.leaf_pages or 1)),
                ],
                split_overlap=max(0, int(icfg.page_overlap or 0)),
                split_by="page",
            )
        elif mode == "passage":
            return HierarchicalDocumentSplitter(
                block_sizes=[
                    max(1, int(icfg.parent_passages or 4)),
                    max(1, int(icfg.leaf_passages or 2)),
                ],
                split_overlap=max(0, int(iccfg.passage_overlap or 0)),
                split_by="passage",
            )
        else:
            logger.warning(f"Unknown split_by '{mode}', defaulting to sentence")
            return HierarchicalDocumentSplitter(block_sizes=[10, 5], split_overlap=1, split_by="sentence")

    def _init_vision_backend(self) -> None:
        """Initialize vision captioning backend with fallbacks."""
        try:
            backend = self._vision_backend
            if backend == "blip":
                _device = 0 if _torch.cuda.is_available() else -1
                try:
                    self._blip_pipe = hf_pipeline("image-to-text", model=self._vision_model_id, device=_device)
                    logger.info(f"Initialized BLIP backend: {self._vision_model_id}")
                except Exception as e:
                    logger.warning(f"Failed to load {self._vision_model_id}, trying base BLIP: {e}")
                    self._blip_pipe = hf_pipeline(
                        "image-to-text", model="Salesforce/blip-image-captioning-base", device=_device
                    )
                self._vision_backend = "blip"
                return

            if backend == "clip":
                self._clip_proc = CLIPProcessor.from_pretrained(self._vision_model_id)
                self._cap_model = CLIPModel.from_pretrained(self._vision_model_id, device_map="auto")
                self._vision_backend = "clip"
                if not self._clip_labels:
                    self._clip_labels = ["document", "poster", "photo", "diagram"]
                logger.info(f"Initialized CLIP backend: {self._vision_model_id}")
                return

            # Try generic image-text backends
            self._cap_processor = AutoProcessor.from_pretrained(self._vision_model_id)
            for name, cls in (
                ("imagetext2text", AutoModelForImageTextToText),
                ("vision2seq", AutoModelForVision2Seq),
                ("causal", AutoModelForCausalLM),
            ):
                try:
                    self._cap_model = cls.from_pretrained(self._vision_model_id, device_map="auto")
                    self._vision_backend = name
                    logger.info(f"Initialized {name} backend: {self._vision_model_id}")
                    return
                except Exception:
                    continue

            # Final fallback to BLIP base
            logger.warning("All vision backends failed, falling back to BLIP base")
            _device = 0 if _torch.cuda.is_available() else -1
            self._blip_pipe = hf_pipeline(
                "image-to-text", model="Salesforce/blip-image-captioning-base", device=_device
            )
            self._vision_backend = "blip"

        except Exception as e:
            self.vision_enabled = False
            logger.error(f"Vision captioning disabled (failed to load {self._vision_model_id}): {e}")

    def cleanup(self):
        """Release resources and cleanup temporary files."""
        logger.info("Cleaning up resources...")

        # Clear vision models
        if hasattr(self, "_blip_pipe") and self._blip_pipe is not None:
            del self._blip_pipe
        if hasattr(self, "_cap_model") and self._cap_model is not None:
            del self._cap_model
        if hasattr(self, "_cap_processor") and self._cap_processor is not None:
            del self._cap_processor
        if hasattr(self, "_clip_proc") and self._clip_proc is not None:
            del self._clip_proc

        # Clear summarizer
        if hasattr(self, "summarizer") and self.summarizer is not None:
            if hasattr(self.summarizer, "model"):
                del self.summarizer.model
            if hasattr(self.summarizer, "pipe"):
                del self.summarizer.pipe

        # Clear embedder
        if hasattr(self, "embedder"):
            del self.embedder

        # Clear CUDA cache
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        logger.info("Cleanup complete")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False

    # ---------------------------
    # Language helpers
    # ---------------------------

    def _configured_languages(self) -> List[str]:
        """Return configured languages."""
        langs = [
            l.strip()
            for l in (self.cfg.indexing.languages or [])
            if isinstance(l, str) and l.strip()
        ]
        return _dedupe_keep_order(langs)

    def _languages_for_path(self, _: str, hint_text: str = "") -> List[str]:
        """Get languages for parsing."""
        cfg_langs = self._configured_languages()
        if cfg_langs:
            return cfg_langs
        detected = _detect_langs_from_text(hint_text or "")
        return detected or ["eng", "chi_sim", "chi_tra"]

    # ---------------------------
    # Image processing
    # ---------------------------

    @staticmethod
    def _normalize_image(path: str) -> Optional[Image.Image]:
        """Normalize image with error handling."""
        try:
            img = Image.open(path)
        except Exception as e:
            logger.warning(f"Failed to open image {path}: {e}")
            return None

        try:
            from PIL import ImageOps

            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        try:
            if img.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                alpha = img.getchannel("A") if "A" in img.getbands() else None
                bg.paste(img, mask=alpha)
                img = bg
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            max_side = 1536
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side))
            if img.mode != "RGB":
                img = img.convert("RGB")
        except Exception as e:
            logger.warning(f"Image normalization failed for {path}: {e}")

        return img

    def _caption_image(self, path: str) -> str:
        """Generate image caption with error handling."""
        if not self.vision_enabled:
            return ""

        img = self._normalize_image(path)
        if img is None:
            return ""

        caption = ""
        backend = self._vision_backend
        max_new = self._vision_max_new

        try:
            if backend == "blip" and self._blip_pipe is not None:
                res = self._blip_pipe(img, max_new_tokens=min(64, max_new))
                if isinstance(res, list) and res and isinstance(res[0], dict):
                    caption = (res[0].get("generated_text") or "").strip()

            elif backend == "clip" and self._clip_proc is not None and self._cap_model is not None:
                labels = self._clip_labels or ["photo", "document", "diagram"]
                text_inputs = [f"a photo of {lbl}" for lbl in labels]
                inputs = self._clip_proc(text=text_inputs, images=img, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._cap_model.device) for k, v in inputs.items()}
                with _torch.no_grad():
                    out = self._cap_model(**inputs)
                    logits = out.logits_per_image.squeeze(0)
                    probs = logits.softmax(dim=-1)
                    topk = min(5, probs.shape[-1])
                    _, idxs = _torch.topk(probs, k=topk)
                tags = [labels[i] for i in idxs.tolist()]
                caption = ("Tags: " + ", ".join(tags)).strip()

            elif self._cap_processor is not None and self._cap_model is not None:
                prompt = self._vision_prompt or "Describe this image."
                if hasattr(self._cap_processor, "apply_chat_template"):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    text = self._cap_processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = self._cap_processor(text=[text], images=[img], return_tensors="pt")
                else:
                    inputs = self._cap_processor(images=img, text=prompt, return_tensors="pt")
                inputs = {k: v.to(self._cap_model.device) for k, v in inputs.items()}
                with _torch.no_grad():
                    out = self._cap_model.generate(**inputs, max_new_tokens=max_new)
                if hasattr(self._cap_processor, "batch_decode"):
                    caption = self._cap_processor.batch_decode(
                        out, skip_special_tokens=True
                    )[0].strip()
                else:
                    tok = AutoTokenizer.from_pretrained(self._vision_model_id, use_fast=True)
                    caption = tok.decode(out[0], skip_special_tokens=True).strip()

        except Exception as e:
            logger.warning(f"Caption generation failed for {path}: {e}")
            caption = ""

        # Strip prompt prefix if present
        vp = (self._vision_prompt or "").strip()
        if caption and vp and caption.lower().startswith(vp.lower()):
            caption = caption[len(vp) :].lstrip(" :,-.")

        return caption.strip()

    # ---------------------------
    # PDF parsing
    # ---------------------------

    def _pdf_elements(self, path: str, use_ocr: bool = False, langs: Optional[List[str]] = None):
        """Partition PDF with error handling."""
        kwargs = dict(
            filename=path,
            infer_table_structure=False,
            extract_images=False,
            include_page_breaks=False,
            starting_page_number=1,
            languages=list(langs or self._configured_languages() or ["eng"]),
        )

        try:
            elements = partition_pdf(strategy=("hi_res" if use_ocr else "fast"), **kwargs)
        except Exception as e:
            logger.error(f"PDF partition failed for {path}: {e}")
            raise

        # Honor max_pdf_pages
        max_pages = int(getattr(self.cfg.indexing, "max_pdf_pages", 0) or 0)
        if max_pages > 0:
            trimmed = []
            for el in elements:
                try:
                    md = el.metadata.to_dict() if hasattr(el, "metadata") else {}
                    pg = md.get("page_number") or md.get("page")
                    if pg is None or int(pg) <= max_pages:
                        trimmed.append(el)
                except (ValueError, TypeError):
                    trimmed.append(el)
            elements = trimmed

        return elements

    @staticmethod
    def _elements_page_and_text(elements) -> Tuple[Dict[int, List[str]], List[str], List[int]]:
        """Extract page-wise text from elements."""
        from collections import defaultdict

        page_texts = defaultdict(list)
        concat_parts: List[str] = []
        page_numbers: List[int] = []

        for el in elements:
            txt = getattr(el, "text", "") or ""
            if txt:
                concat_parts.append(txt)

            md = {}
            try:
                md = el.metadata.to_dict() if hasattr(el, "metadata") else {}
            except Exception:
                pass

            pg = md.get("page_number") or md.get("page")
            if pg is not None:
                try:
                    page_numbers.append(int(pg))
                except (ValueError, TypeError):
                    pass

            if txt.strip() and pg is not None:
                try:
                    page_texts[int(pg)].append(txt)
                except (ValueError, TypeError):
                    pass

        return page_texts, concat_parts, page_numbers

    @staticmethod
    def _page_docs_from_buckets(
        page_texts: Dict[int, List[str]], abs_path: str, fname: str
    ) -> Tuple[List[Document], int, int]:
        """Create per-page documents."""
        docs: List[Document] = []
        total_chars = 0
        pages_with_text = 0

        for pg in sorted(page_texts.keys()):
            content = "\n\n".join(page_texts[pg]).strip()
            if not content:
                continue
            total_chars += len(content)
            pages_with_text += 1
            docs.append(
                Document(
                    content=content,
                    meta={"source_path": abs_path, "filename": fname, "page": int(pg), "__level": 0},
                )
            )

        return docs, total_chars, pages_with_text

    def _elements_to_page_docs(
        self, elements, abs_path: str, fname: str
    ) -> Tuple[List[Document], int, int, str, int]:
        """Convert elements to page docs with stats."""
        page_texts, concat_parts, page_numbers = self._elements_page_and_text(elements)
        docs, total_chars, pages_with_text = self._page_docs_from_buckets(page_texts, abs_path, fname)
        total_pages = (max(page_numbers) if page_numbers else 0)
        return docs, total_chars, pages_with_text, ("\n".join(concat_parts) if concat_parts else ""), total_pages

    def _parse_pdf_into_page_docs(self, path: str) -> List[Document]:
        """Parse PDF with multi-stage fallback (fast then OCR if needed)."""
        abs_path, fname = os.path.abspath(path), os.path.basename(path)

        # Initial pass
        initial_langs = self._configured_languages() or ["eng"]
        try:
            elements = self._pdf_elements(path, use_ocr=False, langs=initial_langs)
        except Exception as e:
            logger.error(f"PDF parsing failed for {path}: {e}")
            return []

        docs, total_chars, pages_with_text, concat_text, total_pages = self._elements_to_page_docs(
            elements, abs_path, fname
        )

        cfg_has_langs = bool(self._configured_languages())
        dynamic_langs = (
            self._languages_for_path(path, hint_text=concat_text) if not cfg_has_langs else initial_langs
        )

        # OCR decision (use config.indexing.ocr_min_* knobs)
        min_chars = int(getattr(self.cfg.indexing, "ocr_min_chars_total", 2000) or 0)
        ratio = float(getattr(self.cfg.indexing, "ocr_min_pages_text_ratio", 0.15) or 0.0)
        pages_ratio = (pages_with_text / max(1, total_pages)) if total_pages else 0.0
        needs_ocr = self.cfg.indexing.ocr_fallback and (
            (min_chars and total_chars < min_chars) or (ratio and pages_ratio < ratio)
        )

        if needs_ocr:
            try:
                logger.info(
                    f"Applying OCR to {fname} (chars={total_chars}, page_ratio={pages_ratio:.2f})"
                )
                elements_ocr = self._pdf_elements(path, use_ocr=True, langs=dynamic_langs)
                docs_ocr, total_chars_ocr, _, _, _ = self._elements_to_page_docs(
                    elements_ocr, abs_path, fname
                )
                if total_chars_ocr > total_chars:
                    docs = docs_ocr
                    logger.info(
                        f"OCR improved extraction: {total_chars} → {total_chars_ocr} chars"
                    )
            except Exception as e:
                logger.warning(f"OCR fallback failed for {fname}: {e}")

        if not docs:
            logger.warning(f"No content extracted from {fname}, trying generic parser")
            single = self._parse_file(path)
            docs = [single] if single else []

        return docs

    # ---------------------------
    # Image and file parsing
    # ---------------------------

    def _parse_image_into_doc(self, path: str) -> Optional[Document]:
        """Parse image with OCR and optional captioning."""
        abs_path, fname = os.path.abspath(path), os.path.basename(path)

        # OCR
        ocr_text = ""
        try:
            langs = self._languages_for_path(path, hint_text="")
            els = partition_image(filename=path, languages=langs, strategy="ocr_only")
            texts = [((getattr(e, "text", "") or "").strip()) for e in els]
            ocr_text = "\n".join([t for t in texts if t])
        except Exception as e:
            logger.warning(f"Image OCR failed for {fname}: {e}")

        # Caption
        caption = ""
        if self.vision_enabled:
            try:
                caption = self._caption_image(path)
            except Exception as e:
                logger.warning(f"Image captioning failed for {fname}: {e}")

        meta = {
            "source_path": abs_path,
            "filename": fname,
            "page": 1,
            "__level": 0,
            "mime_guess": mimetypes.guess_type(path)[0] or "",
            "ext": os.path.splitext(path)[1].lower(),
        }

        parts = []
        if caption.strip():
            parts.append(f"[VISION_CAPTION] {caption.strip()}")
            meta["vision_caption"] = caption.strip()
        if ocr_text.strip():
            parts.append(ocr_text.strip())

        content = "\n\n".join(parts).strip() or "[IMAGE]"
        return Document(content=content, meta=meta)

    def _parse_file(self, path: str) -> Document:
        """Generic file parser."""
        ext = os.path.splitext(path)[1].lower()

        # Images
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
            doc = self._parse_image_into_doc(path)
            if doc:
                return doc
            return Document(
                content="[IMAGE]",
                meta={
                    "source_path": os.path.abspath(path),
                    "filename": os.path.basename(path),
                    "ext": ext,
                    "mime_guess": mimetypes.guess_type(path)[0] or "",
                    "__level": 0,
                    "page": 1,
                },
            )

        # HTML
        if ext in (".html", ".htm"):
            return self._parse_html(path)

        # Generic
        try:
            texts: List[str] = []
            elements = partition_auto(filename=path, languages=self._configured_languages() or ["eng"])
            for el in elements:
                txt = (getattr(el, "text", None) or "").strip()
                if txt:
                    texts.append(txt)
            content = "\n\n".join(texts).strip()
        except Exception as e:
            logger.warning(f"Generic parsing failed for {path}: {e}")
            content = ""

        meta = {
            "source_path": os.path.abspath(path),
            "filename": os.path.basename(path),
            "ext": ext,
            "mime_guess": mimetypes.guess_type(path)[0] or "",
            "__level": 0,
        }
        return Document(content=content or "[EMPTY]", meta=meta)

    def _parse_html(self, path: str) -> Document:
        """Parse HTML with link extraction."""
        ext = os.path.splitext(path)[1].lower()
        texts: List[str] = []

        try:
            elements = partition_auto(filename=path, languages=self._configured_languages() or ["eng"])
            for el in elements:
                txt = (getattr(el, "text", "") or "").strip()
                if txt:
                    texts.append(txt)
        except Exception as e:
            logger.warning(f"HTML text extraction failed for {path}: {e}")

        content_text = "\n\n".join(texts).strip()

        # Extract links
        links: List[Tuple[str, str]] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw_html = f.read()
            parser = _LinkExtractor()
            parser.feed(raw_html)
            links = [
                (href.strip(), (text or "").strip())
                for href, text in parser.links
                if href and href.strip()
            ]
        except Exception as e:
            logger.warning(f"HTML link extraction failed for {path}: {e}")

        link_lines = [f"[LINK] {t or '(no title)'} — {u}" for (u, t) in links]
        parts = [p for p in (content_text, "\n".join(link_lines)) if p]
        content = "\n\n".join(parts) if parts else "[HTML]"

        meta = {
            "source_path": os.path.abspath(path),
            "filename": os.path.basename(path),
            "ext": ext,
            "mime_guess": mimetypes.guess_type(path)[0] or "",
            "__level": 0,
            "links": [{"url": u, "title": t} for (u, t) in links],
        }
        return Document(content=content, meta=meta)

    # ---------------------------
    # Email parsing with security limits
    # ---------------------------

    def _parse_eml_into_docs(self, path: str, depth: int = 0) -> List[Document]:
        """Parse email with recursion limits."""
        if depth > self.cfg.indexing.max_email_recursion:
            logger.warning(
                f"Max email recursion depth ({self.cfg.indexing.max_email_recursion}) exceeded for {path}"
            )
            return []

        abs_path, fname = os.path.abspath(path), os.path.basename(path)

        try:
            with open(path, "rb") as f:
                msg = BytesParser(policy=policy.default).parse(f)
        except Exception as e:
            logger.error(f"Failed to parse email {path}: {e}")
            return []

        def _html_to_text(s: str) -> str:
            try:
                import html2text

                return html2text.html2text(s)
            except ImportError:
                import re

                s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", s)
                s = re.sub(r"(?is)<br\s*/?>", "\n", s)
                s = re.sub(r"(?is)</p>", "\n\n", s)
                s = re.sub(r"(?is)<.*?>", " ", s)
                return " ".join(s.split())

        def _extract_body(m) -> str:
            texts_plain, texts_html = [], []
            if m.is_multipart():
                for part in m.walk():
                    disp = (part.get_content_disposition() or "").lower()
                    ctype = (part.get_content_type() or "").lower()
                    if disp == "attachment":
                        continue
                    try:
                        if ctype == "text/plain":
                            texts_plain.append(part.get_content())
                        elif ctype == "text/html":
                            texts_html.append(_html_to_text(part.get_content()))
                    except Exception:
                        pass
            else:
                ctype = (m.get_content_type() or "").lower()
                try:
                    if ctype == "text/plain":
                        texts_plain.append(m.get_content())
                    elif ctype == "text/html":
                        texts_html.append(_html_to_text(m.get_content()))
                except Exception:
                    pass

            body = "\n\n".join([t for t in texts_plain if t.strip()]) or "\n\n".join(
                [t for t in texts_html if t.strip()]
            )
            return (body or "").strip()

        body_text = _extract_body(msg)

        email_meta = {
            "email_subject": (msg.get("subject") or "").strip(),
            "email_from": (msg.get("from") or "").strip(),
            "email_to": (msg.get("to") or "").strip(),
            "email_date": (msg.get("date") or "").strip(),
            "email_message_id": (msg.get("message-id") or "").strip(),
        }

        parent_meta = {
            "source_path": abs_path,
            "filename": fname,
            "mime_guess": "message/rfc822",
            "__level": 0,
            **email_meta,
        }
        parent_content = (
            f"[EMAIL] Subject: {email_meta['email_subject']}\n\n{body_text}"
            if body_text
            else "[EMAIL]"
        )
        parent_doc = Document(content=parent_content, meta=parent_meta)

        docs: List[Document] = [parent_doc]
        attachments_index: List[Dict[str, Any]] = []

        attach_root = os.path.join(
            self.cfg.indexing.persist_meta_path,
            "eml_attachments",
            hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:12],
        )
        ensure_dirs(attach_root)

        # Track total attachment size
        total_attach_size = 0
        max_attach_mb = self.cfg.indexing.max_attachment_size_mb

        def _is_body_part(part) -> bool:
            ctype = (part.get_content_type() or "").lower()
            disp = (part.get_content_disposition() or "").lower()
            fname = part.get_filename()
            if part.is_multipart():
                return False
            return (ctype in ("text/plain", "text/html")) and (disp != "attachment") and (not fname)

        for idx, part in enumerate(msg.walk(), 1):
            if part.is_multipart() or _is_body_part(part):
                continue

            ctype = (part.get_content_type() or "").lower()
            disp = (part.get_content_disposition() or "").lower()
            fname = part.get_filename()
            cid = (part.get("Content-ID") or "").strip().strip("<>")

            # Get payload
            try:
                payload = part.get_content()
                if isinstance(payload, str):
                    payload = payload.encode(part.get_content_charset() or "utf-8", errors="ignore")
                elif payload is None:
                    payload = part.get_payload(decode=True) or b""
            except Exception:
                payload = part.get_payload(decode=True) or b""

            # Check size limits
            if max_attach_mb > 0:
                size_mb = len(payload) / (1024 * 1024)
                if size_mb > max_attach_mb:
                    logger.warning(
                        f"Skipping large attachment in {fname}: {size_mb:.1f}MB > {max_attach_mb}MB"
                    )
                    continue
                total_attach_size += len(payload)
                if total_attach_size / (1024 * 1024) > max_attach_mb * 10:  # Total limit
                    logger.warning(
                        f"Total attachment size exceeded for {fname}, skipping remaining attachments"
                    )
                    break

            # Synthesize filename
            if not fname:
                base = f"inline-{idx}" if not cid else f"cid-{cid}"
                ext = mimetypes.guess_extension(ctype) or (".eml" if ctype == "message/rfc822" else "")
                fname = base + (ext or "")
            if ctype == "message/rfc822" and not fname.lower().endswith(".eml"):
                fname += ".eml"

            # Save to disk
            safe = os.path.basename(fname).replace("\x00", "")
            out_path = os.path.join(attach_root, safe)
            base, ext = os.path.splitext(out_path)
            n = 1
            while os.path.exists(out_path):
                out_path = f"{base}__{n}{ext}"
                n += 1

            try:
                with open(out_path, "wb") as fo:
                    fo.write(payload)
            except IOError as e:
                logger.error(f"Failed to save attachment {fname}: {e}")
                continue

            # Update index
            attachments_index.append(
                {
                    "filename": os.path.basename(out_path),
                    "saved_path": out_path,
                    "content_type": ctype,
                    "disposition": disp or "",
                    "content_id": cid or "",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )

            # Parse recursively
            try:
                adocs = self._parse_path_recursive(out_path, depth + 1)
            except Exception as e:
                logger.warning(f"Failed to parse attachment {fname}: {e}")
                adocs = []

            # Add linkage
            for d in adocs:
                m = dict(d.meta or {})
                m.update(
                    {
                        "attachment_of": parent_doc.id,
                        "attachment_filename": os.path.basename(out_path),
                        **email_meta,
                    }
                )
                d.meta = m

            docs.extend(adocs)

        pm = dict(parent_doc.meta or {})
        pm["attachments"] = attachments_index
        parent_doc.meta = pm
        return docs

    def _parse_path_recursive(self, path: str, depth: int = 0) -> List[Document]:
        """Parse with recursion depth tracking."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            return self._parse_pdf_into_page_docs(path)
        elif ext == ".eml":
            return self._parse_eml_into_docs(path, depth=depth)
        else:
            return [self._parse_file(path)]

    def _parse_path(self, p: str) -> List[Document]:
        """Top-level parse dispatcher with size checks."""
        try:
            _check_file_size(p, self.cfg.indexing.max_file_size_mb)
        except (ValueError, IOError) as e:
            logger.warning(f"Skipping file {p}: {e}")
            return []

        return self._parse_path_recursive(p, depth=0)

    # ---------------------------
    # Splitting and hierarchy
    # ---------------------------

    def _split_hierarchical(self, root_docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """Run hierarchical splitter."""
        try:
            out = self.splitter.run(documents=root_docs)
            parts: List[Document] = out.get("documents", []) or []
        except Exception as e:
            logger.error(f"Hierarchical splitting failed: {e}")
            return [], []

        if not parts:
            return [], []

        max_level = max(int((p.meta or {}).get("__level", 0)) for p in parts)
        leaves, non_leaves = [], []
        parents_by_id: Dict[str, Document] = {}
        children_map: Dict[str, List[str]] = {}

        # Partition by level
        for d in parts:
            lvl = int((d.meta or {}).get("__level", 0))
            if lvl == max_level:
                leaves.append(d)
            else:
                non_leaves.append(d)
                parents_by_id[d.id] = d

        # Build children map
        for d in parts:
            pid = (d.meta or {}).get("__parent_id")
            if pid:
                children_map.setdefault(pid, []).append(d.id)

        # Normalize relationships
        for d in parts:
            rel = getattr(d, "relationships", {}) or {}
            rel.setdefault("parent_id", (d.meta or {}).get("__parent_id"))
            if d.id in children_map:
                rel["children_ids"] = children_map[d.id]
            d.relationships = rel

        # Serialize children to meta
        for pid, pdoc in parents_by_id.items():
            m = dict(pdoc.meta or {})
            m["__children_ids"] = json.dumps(children_map.get(pid, []))
            pdoc.meta = m

        # Ensure leaves have __parent_id
        for leaf in leaves:
            m = dict(leaf.meta or {})
            pid = m.get("__parent_id")
            if pid is not None:
                m["__parent_id"] = str(pid)
            leaf.meta = m

        # Propagate vision captions
        cap_by_file = {
            (r.meta or {}).get("filename"): (r.meta or {}).get("vision_caption")
            for r in root_docs
            if (r.meta or {}).get("filename")
            and isinstance((r.meta or {}).get("vision_caption"), str)
        }

        def _apply_caption(d: Document):
            if not cap_by_file:
                return
            m = d.meta or {}
            fn = m.get("filename")
            if fn and "vision_caption" not in m and fn in cap_by_file:
                m = dict(m)
                m["vision_caption"] = cap_by_file[fn].strip()
                d.meta = m

        for d in non_leaves + leaves:
            _apply_caption(d)

        return non_leaves, leaves

    def _propagate_pages_and_ranges(
        self, roots: List[Document], parents: List[Document], leaves: List[Document]
    ) -> None:
        """Propagate page metadata."""
        id_to_page: Dict[str, int] = {}
        for d in roots + parents + leaves:
            pg = (d.meta or {}).get("page")
            try:
                if pg is not None:
                    id_to_page[d.id] = int(pg)
            except (ValueError, TypeError):
                pass

        parent_to_children: Dict[str, List[str]] = {}
        for d in parents + roots:
            rel = getattr(d, "relationships", {}) or {}
            kids = rel.get("children_ids")
            if not kids:
                try:
                    kids = json.loads((d.meta or {}).get("__children_ids", "[]"))
                except (json.JSONDecodeError, TypeError):
                    kids = []
            if kids:
                parent_to_children[d.id] = list(kids)

        # Resolve leaf pages
        leaf_pages: Dict[str, int] = {}
        for lf in leaves:
            m = lf.meta or {}
            pg = m.get("page")
            if pg is not None:
                try:
                    leaf_pages[lf.id] = int(pg)
                    continue
                except (ValueError, TypeError):
                    pass
            pid = m.get("__parent_id")
            if pid and pid in id_to_page:
                leaf_pages[lf.id] = id_to_page[pid]

        # Write back
        for lf in leaves:
            if "page" not in (lf.meta or {}) and lf.id in leaf_pages:
                m = dict(lf.meta or {})
                m["page"] = int(leaf_pages[lf.id])
                lf.meta = m

        # Compute ranges
        def collect_desc_leaf_pages(pid: str, acc: List[int]):
            for cid in parent_to_children.get(pid, []):
                if cid in parent_to_children:
                    collect_desc_leaf_pages(cid, acc)
                else:
                    if cid in leaf_pages:
                        acc.append(leaf_pages[cid])

        for pdoc in parents:
            pages: List[int] = []
            collect_desc_leaf_pages(pdoc.id, pages)
            if pages:
                m = dict(pdoc.meta or {})
                m["min_page"] = int(min(pages))
                m["max_page"] = int(max(pages))
                pdoc.meta = m

    # ---------------------------
    # Embedding with streaming
    # ---------------------------

    def _embed_and_write_streaming(
        self, docs: List[Document], store: ChromaDocumentStore, batch_size: int, desc: str
    ) -> int:
        """Embed and write in batches to reduce memory footprint."""
        total_written = 0
        iterator = range(0, len(docs), max(1, batch_size))

        if TQDM_AVAILABLE:
            iterator = tqdm(iterator, desc=desc, unit="batch")

        for i in iterator:
            chunk = docs[i : i + batch_size]
            try:
                res = self.embedder.run(documents=chunk)
                embedded = res["documents"]
                embedded = _sanitize_meta_for_all(embedded)
                DocumentWriter(store).run(documents=embedded)
                total_written += len(embedded)
            except Exception as e:
                logger.error(f"Failed to embed/write batch at index {i}: {e}")

        return total_written

    # ---------------------------
    # Main index method
    # ---------------------------

    def index(self) -> Dict[str, Any]:
        """Execute end-to-end indexing pipeline."""
        t0 = time.time()

        files = list_corpus_files(
            self.cfg.indexing.corpus_dir,
            self.cfg.indexing.files,
            max_files=int(self.cfg.indexing.max_files or 0),
        )

        if not files:
            logger.warning("No files found to index")
            return {}

        doc_batch_size = int(self.cfg.advanced.doc_batch_size or 0)
        if doc_batch_size <= 0:
            doc_batch_size = len(files)

        logger.info(f"Parsing {len(files)} files sequentially in batches of {doc_batch_size}...")

        all_roots: List[Document] = []
        all_parents: List[Document] = []
        all_leaves: List[Document] = []

        # Process files in batches (parse → split → accumulate)
        for start in range(0, len(files), doc_batch_size):
            batch_paths = files[start : start + doc_batch_size]
            batch_roots: List[Document] = []

            iterator = batch_paths
            if TQDM_AVAILABLE:
                iterator = tqdm(batch_paths, desc=f"Parsing files {start+1}-{start+len(batch_paths)}", unit="file")

            for p in iterator:
                try:
                    batch_roots.extend(self._parse_path(p))
                except Exception as e:
                    logger.error(f"Failed to parse {p}: {e}")

            if not batch_roots:
                continue

            all_roots.extend(batch_roots)

            # Split batch
            logger.info(f"Splitting {len(batch_roots)} root documents hierarchically (batch)...")
            parents, leaves = self._split_hierarchical(batch_roots)
            self._propagate_pages_and_ranges(batch_roots, parents, leaves)

            # Filter empty
            leaves = [d for d in leaves if (d.content or "").strip()]
            parents = [d for d in parents if (d.content or "").strip()]

            all_parents.extend(parents)
            all_leaves.extend(leaves)

        if not all_roots:
            logger.warning("No root documents produced after parsing")
            return {}

        logger.info(f"Parsed {len(all_roots)} root documents")
        logger.info(f"Split into {len(all_parents)} parents and {len(all_leaves)} leaves")

        # Embed and write leaves (streaming)
        logger.info("Embedding and writing leaf documents...")
        leaves_written = self._embed_and_write_streaming(
            all_leaves,
            self.leaf_store,
            self.cfg.advanced.batch_size_docs,
            "Embedding leaves",
        )

        # Embed and write parents (optional, streaming)
        parents_written = 0
        if bool(self.cfg.indexing.embed_parents):
            logger.info("Embedding and writing parent documents...")
            parents_written = self._embed_and_write_streaming(
                all_parents,
                self.parent_store,
                self.cfg.advanced.parent_batch_size_docs,
                "Embedding parents",
            )
        else:
            logger.info("Skipping parent embedding (indexing.embed_parents=false)")
            # Still sanitize for export
            all_parents = _sanitize_meta_for_all(all_parents)

        # Exports
        logger.info("Exporting metadata...")
        try:
            export_docs_jsonl(all_leaves, self.leaf_dump)
            export_docs_jsonl(all_parents, self.parent_dump)
            export_parents_sidecar(all_parents, self.sidecar_path)
        except Exception as e:
            logger.error(f"Export failed: {e}")

        elapsed = round(time.time() - t0, 2)

        info = {
            "files_indexed": len(files),
            "docs_root": len(all_roots),
            "docs_parents": len(all_parents),
            "docs_leaves": len(all_leaves),
            "leaves_written": leaves_written,
            "parents_written": parents_written,
            "vector_backend": "chroma",
            "leaf_chroma_path": self.cfg.vectorstore.persist_path,
            "leaf_collection": self.cfg.vectorstore.collection_name,
            "parent_chroma_path": self.cfg.parent_vectorstore.persist_path,
            "parent_collection": self.cfg.parent_vectorstore.collection_name,
            "parent_sidecar_path": self.sidecar_path,
            "leaf_dump": self.leaf_dump,
            "parent_dump": self.parent_dump,
            "embedder_model": self.cfg.models.embedder_model,
            "elapsed_sec": elapsed,
            "embed_parents": bool(self.cfg.indexing.embed_parents),
        }

        logger.info("Indexing complete:")
        print(json.dumps(info, indent=2))
        return info


# ---------------------------
# Main
# ---------------------------

def main(cfg_path: Optional[str]):
    """
    Main entry point with context manager cleanup.
    Run: python hier_indexer.py config.fast.yaml
    """
    try:
        cfg = load_config(cfg_path)
        with HierarchicalIndexer(cfg) as indexer:
            indexer.index()
    except ConfigValidationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(cfg_path)
