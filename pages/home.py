import streamlit as st
import tempfile
import os
import json
import difflib
import time
import re
import traceback
import hashlib
import requests
from collections import OrderedDict
from typing import Any, List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from docx_revisions import RevisionDocument, RevisionParagraph
from pydantic import BaseModel
from supabase import create_client, Client
from db_tables import get_table_name
from config import (
    LLM_PROVIDERS,
    OPENROUTER_MODELS,
    GOOGLE_VERTEX_MODELS,
    GOOGLE_VERTEX_MODEL_LIMITS,
    DEFAULT_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    DEFAULT_MAX_WORKERS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    TOKEN_CUSHION_FACTOR,
    PROMPT_TEMPLATE_OVERHEAD,
    MIN_COMPLETION_RESERVE,
    DEFAULT_PARAGRAPHS_PER_PAGE,
    DEFAULT_EDITS_PER_PAGE
)


ALLOWED_EMAIL_DOMAIN = "@red-publish.com"

class Edit(BaseModel):
    paragraph_index: int
    corrected_text: str
    reason: str

def compute_paragraph_hash(text: str) -> str:
    """Compute a short hash of paragraph text for verification."""
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

class ProofreadingResponse(BaseModel):
    edits: List[Edit]
    summary: str

# ============================================================================
# Prompt Management Functions (Supabase)
# ============================================================================

@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize and cache the Supabase client."""
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)

def load_prompts() -> Dict[str, Dict[str, Any]]:
    """
    Load prompts from Supabase.
    Returns ordered dict with default prompt first.
    """
    try:
        supabase = get_supabase_client()
        response = supabase.table(get_table_name("prompts")).select("name, content, is_protected").order("created_at").execute()
        
        prompts = OrderedDict()
        default_name = "預設"
        
        # Put default prompt first
        for row in response.data:
            if row["name"] == default_name:
                prompts[default_name] = {
                    "content": row["content"],
                    "protected": row["is_protected"],
                }
                break
        
        # Add remaining prompts
        for row in response.data:
            if row["name"] != default_name:
                prompts[row["name"]] = {
                    "content": row["content"],
                    "protected": row["is_protected"],
                }
        
        if not prompts:
            st.error("無法載入提示：資料庫中無提示資料")
            st.stop()
        
        return prompts
    except Exception as e:
        st.error(f"無法載入提示：{str(e)}")
        st.stop()

_MAX_PROMPT_NAME_LENGTH = 50

def _validate_prompt_name(name: str) -> Tuple[bool, str]:
    """Validate prompt name."""
    name = name.strip()
    if not name:
        return False, "提示名稱不能為空"
    if len(name) > _MAX_PROMPT_NAME_LENGTH:
        return False, f"提示名稱不能超過 {_MAX_PROMPT_NAME_LENGTH} 個字元"
    return True, ""

def add_prompt(prompts: Dict[str, Dict[str, Any]], name: str, content: str) -> Tuple[bool, str]:
    """
    Add a new prompt to Supabase.
    Returns (success, message).
    """
    valid, msg = _validate_prompt_name(name)
    if not valid:
        return False, msg
    
    if not content or not content.strip():
        return False, "提示內容不能為空"
    
    if name in prompts:
        return False, f"提示 '{name}' 已存在"
    
    try:
        supabase = get_supabase_client()
        supabase.table(get_table_name("prompts")).insert({
            "name": name,
            "content": content,
            "is_protected": False,
        }).execute()
        return True, f"已新增提示 '{name}'"
    except Exception as e:
        return False, f"儲存失敗：{str(e)}"

def update_prompt(prompts: Dict[str, Dict[str, Any]], name: str, content: str) -> Tuple[bool, str]:
    """
    Update an existing prompt's content in Supabase (if not protected).
    Returns (success, message).
    """
    if name not in prompts:
        return False, f"提示 '{name}' 不存在"
    
    if prompts[name].get("protected", False):
        return False, f"提示 '{name}' 受保護，無法編輯"
    
    if not content or not content.strip():
        return False, "提示內容不能為空"
    
    try:
        supabase = get_supabase_client()
        supabase.table(get_table_name("prompts")).update({
            "content": content,
        }).eq("name", name).execute()
        return True, f"已更新提示 '{name}'"
    except Exception as e:
        return False, f"儲存失敗：{str(e)}"

def delete_prompt(prompts: Dict[str, Dict[str, Any]], name: str) -> Tuple[bool, str]:
    """
    Delete a prompt from Supabase (if not protected).
    Returns (success, message).
    """
    if name not in prompts:
        return False, f"提示 '{name}' 不存在"
    
    if prompts[name].get("protected", False):
        return False, f"提示 '{name}' 受保護，無法刪除"
    
    try:
        supabase = get_supabase_client()
        supabase.table(get_table_name("prompts")).delete().eq("name", name).execute()
        return True, f"已刪除提示 '{name}'"
    except Exception as e:
        return False, f"刪除失敗：{str(e)}"

def get_openrouter_client(api_key: str) -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

def parse_google_service_account_info(raw_service_account: Any) -> Dict[str, Any]:
    """Parse and validate Google service account JSON from secrets."""
    if isinstance(raw_service_account, str):
        try:
            service_account_info = json.loads(raw_service_account)
        except json.JSONDecodeError as decode_error:
            raise ValueError("google_vertex.service_account_json 不是有效的 JSON") from decode_error
    elif isinstance(raw_service_account, dict):
        service_account_info = raw_service_account
    elif hasattr(raw_service_account, "items"):
        service_account_info = dict(raw_service_account.items())
    else:
        raise ValueError("google_vertex.service_account_json 格式錯誤，需為 JSON 字串或物件")

    required_fields = ["client_email", "token_uri", "private_key"]
    missing_fields = [field for field in required_fields if not service_account_info.get(field)]
    if missing_fields:
        raise ValueError(
            "google_vertex.service_account_json 缺少必要欄位："
            + ", ".join(missing_fields)
            + "。請使用 GCP 服務帳戶金鑰 JSON（不要使用 OAuth Client ID 設定）。"
        )

    return service_account_info

def get_google_vertex_client() -> Any:
    """Initialize Google Vertex AI client from secrets.toml [google_vertex]."""
    try:
        from google import genai
        from google.oauth2 import service_account
    except ImportError as import_error:
        raise ValueError(
            "缺少 Google Vertex 相依套件：請安裝 google-genai 與 google-auth，"
            "例如執行 `pip install google-genai google-auth`"
        ) from import_error

    settings = st.secrets.get("google_vertex", {})
    project_id = settings.get("project_id")
    location = settings.get("location")
    raw_service_account = settings.get("service_account_json")

    if not project_id:
        raise ValueError("缺少 Google Vertex 設定：請在 secrets.toml 的 [google_vertex] 提供 project_id")
    if not location:
        raise ValueError("缺少 Google Vertex 設定：請在 secrets.toml 的 [google_vertex] 提供 location")

    if not raw_service_account:
        raise ValueError("缺少 Google Vertex 設定：請在 secrets.toml 的 [google_vertex] 提供 service_account_json")

    service_account_info = parse_google_service_account_info(raw_service_account)

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials,
    )

def fetch_google_vertex_models(client: Any) -> List[str]:
    """Fetch available Gemini models from Vertex AI for the current project/location."""
    try:
        model_names: List[str] = []
        for model in client.models.list():
            full_name = getattr(model, "name", "") or ""
            short_name = full_name.split("/")[-1] if full_name else ""
            if not short_name.startswith("gemini"):
                continue

            supported_actions = getattr(model, "supported_actions", None) or []
            if supported_actions and not any("generate" in str(action).lower() for action in supported_actions):
                continue

            model_names.append(short_name)

        return sorted(set(model_names))
    except Exception as e:
        raise ValueError(f"無法從 Google Vertex 取得模型清單：{str(e)}") from e

def check_api_credits(api_key: str) -> Optional[Dict]:
    """
    Check OpenRouter API key credits.
    Returns dict with credit info or None if failed.
    """
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def fetch_google_model_info(client: Any, model_id: str) -> Tuple[Optional[Dict], str]:
    """
    Fetch model metadata for Google Vertex models.
    Primary: lookup from GOOGLE_VERTEX_MODEL_LIMITS (hardcoded, instant).
    Fallback: query Vertex API for unknown models.
    """
    # Primary: hardcoded lookup (API metadata often returns None for token limits)
    if model_id in GOOGLE_VERTEX_MODEL_LIMITS:
        input_limit, output_limit = GOOGLE_VERTEX_MODEL_LIMITS[model_id]
        return {
            "id": model_id,
            "name": model_id,
            "context_length": input_limit,
            "max_completion_tokens": output_limit,
        }, f"來源=hardcoded, input_token_limit={input_limit:,}, output_token_limit={output_limit:,}"

    # Fallback: try Vertex API for models not in the lookup table
    diagnostics: List[str] = []
    for candidate in [model_id, f"publishers/google/models/{model_id}"]:
        try:
            model_obj = client.models.get(model=candidate)
            input_limit = getattr(model_obj, "input_token_limit", 0) or 0
            output_limit = getattr(model_obj, "output_token_limit", 0) or 0
            if input_limit > 0:
                return {
                    "id": model_id,
                    "name": getattr(model_obj, "display_name", model_id),
                    "context_length": int(input_limit),
                    "max_completion_tokens": int(output_limit),
                }, f"來源=API/{candidate}, input_token_limit={int(input_limit):,}, output_token_limit={int(output_limit):,}"
            diagnostics.append(f"{candidate}: input_token_limit=0")
        except Exception as e:
            diagnostics.append(f"{candidate}: {str(e)}")

    return None, " | ".join(diagnostics) if diagnostics else "未取得任何模型限制資訊"

def fetch_model_info(api_key: str, model_id: str) -> Optional[Dict]:
    """
    Fetch model metadata from OpenRouter API.
    Returns dict with context_length, max_completion_tokens, etc. or None if failed.
    """
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if response.status_code != 200:
            return None
        
        data = response.json()
        for model in data.get("data", []):
            if model.get("id") == model_id:
                return {
                    "id": model["id"],
                    "name": model.get("name", model_id),
                    "context_length": model.get("context_length", 0),
                    "max_completion_tokens": model.get("top_provider", {}).get("max_completion_tokens", 0),
                    "tokenizer": model.get("architecture", {}).get("tokenizer", "unknown"),
                }
        return None
    except Exception:
        return None

_tiktoken_encoder = None
_tiktoken_load_attempted = False

def _get_tiktoken_encoder():
    """Lazy-load tiktoken cl100k_base encoder (used by GPT-4 family). Cached after first call."""
    global _tiktoken_encoder, _tiktoken_load_attempted
    if _tiktoken_load_attempted:
        return _tiktoken_encoder
    _tiktoken_load_attempted = True
    try:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _tiktoken_encoder = None
    return _tiktoken_encoder

def _estimate_tokens_cjk_heuristic(text: str) -> int:
    """
    CJK-aware heuristic fallback for token estimation.
    CJK characters ~1.5 tokens each; ASCII ~0.25 tokens per char (1 token per 4 chars).
    """
    cjk_count = 0
    ascii_count = 0
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
                0x2E80 <= cp <= 0x2EFF or 0x3000 <= cp <= 0x303F or
                0xFF00 <= cp <= 0xFFEF or 0xF900 <= cp <= 0xFAFF or
                0x20000 <= cp <= 0x2A6DF):
            cjk_count += 1
        else:
            ascii_count += 1
    return int(cjk_count * 1.5 + ascii_count * 0.25) or 1

def estimate_tokens(text: str) -> int:
    """
    Estimate token count using tiktoken (cl100k_base) with CJK heuristic fallback.
    tiktoken gives accurate counts for most LLM tokenizers.
    Falls back to a CJK-aware character heuristic if tiktoken is unavailable.
    """
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    return _estimate_tokens_cjk_heuristic(text)

def compute_dynamic_chunk_size(
    rdoc: RevisionDocument,
    system_prompt: str,
    context_length: int,
    max_completion_tokens: int,
    process_percentage: int = 100
) -> int:
    """
    Compute optimal paragraphs-per-chunk based on model token limits.
    
    Budget: context_length - completion_reserve - system_prompt_tokens - prompt_overhead
    Then apply cushion factor and divide by average tokens per paragraph.
    Falls back to DEFAULT_CHUNK_SIZE if computation yields unreasonable results.
    """
    # Reserve tokens for the LLM's response
    completion_reserve = max(max_completion_tokens, MIN_COMPLETION_RESERVE)
    
    # Estimate system prompt tokens
    system_prompt_tokens = estimate_tokens(system_prompt)
    
    # Available tokens for paragraph content in the user prompt
    available = context_length - completion_reserve - system_prompt_tokens - PROMPT_TEMPLATE_OVERHEAD
    safe_budget = int(available * TOKEN_CUSHION_FACTOR)
    
    if safe_budget <= 0:
        return DEFAULT_CHUNK_SIZE
    
    # Estimate average tokens per paragraph from the actual document
    total_paragraphs = len(rdoc.paragraphs)
    paragraphs_to_process = max(1, int(total_paragraphs * process_percentage / 100))
    
    # Sample up to 200 paragraphs to get average token count per paragraph
    sample_size = min(paragraphs_to_process, 200)
    total_tokens = 0
    for i in range(sample_size):
        text = rdoc.paragraphs[i].text.strip()
        if text:
            # Estimate tokens for the full line as it appears in the prompt: "[i|hash] text"
            line = f"[{i}|abcd1234] {text}"
            total_tokens += estimate_tokens(line)
        else:
            # Empty paragraph marker "[i] (empty paragraph)"
            total_tokens += estimate_tokens(f"[{i}] (empty paragraph)")
    
    if total_tokens == 0:
        return DEFAULT_CHUNK_SIZE
    
    avg_tokens_per_para = max(1, total_tokens // sample_size)
    
    if avg_tokens_per_para <= 0:
        return DEFAULT_CHUNK_SIZE
    
    chunk_size = int(safe_budget / avg_tokens_per_para)
    
    # Clamp to reasonable bounds
    chunk_size = max(10, min(chunk_size, MAX_CHUNK_SIZE))
    
    return chunk_size

def check_for_tracked_changes(rdoc: RevisionDocument) -> Tuple[bool, int]:
    """
    Check if the document has any pending tracked changes (insertions or deletions).
    Returns (has_changes, count) where has_changes is True if tracked changes exist.
    """
    from docx import Document
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.ns import qn
    
    change_count = 0
    
    # Access the underlying document
    doc = rdoc._document
    
    # Check all paragraphs for tracked changes
    for paragraph in doc.paragraphs:
        # Check for insertions (w:ins)
        insertions = paragraph._element.findall(qn('w:ins'))
        change_count += len(insertions)
        
        # Check for deletions (w:del)
        deletions = paragraph._element.findall(qn('w:del'))
        change_count += len(deletions)
        
        # Check for move from/to
        move_from = paragraph._element.findall(qn('w:moveFrom'))
        change_count += len(move_from)
        
        move_to = paragraph._element.findall(qn('w:moveTo'))
        change_count += len(move_to)
    
    # Also check in tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    insertions = paragraph._element.findall(qn('w:ins'))
                    change_count += len(insertions)
                    
                    deletions = paragraph._element.findall(qn('w:del'))
                    change_count += len(deletions)
                    
                    move_from = paragraph._element.findall(qn('w:moveFrom'))
                    change_count += len(move_from)
                    
                    move_to = paragraph._element.findall(qn('w:moveTo'))
                    change_count += len(move_to)
    
    return (change_count > 0, change_count)

def read_document_paragraphs(rdoc: RevisionDocument) -> str:
    lines = []
    for i, para in enumerate(rdoc.paragraphs):
        text = para.text.strip()
        if text:
            lines.append(f"[{i}] {text}")
    return "\n".join(lines)

def proofread_chunk_with_retry(
    client: Any,
    provider: str,
    model: str,
    chunk_text: str,
    system_prompt: str,
    chunk_info: str = "",
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_completion_tokens: Optional[int] = None,
    input_token_limit: int = 0
) -> Tuple[Optional[ProofreadingResponse], List[str], Dict[str, Any]]:
    """
    Wrapper function that retries chunk processing with exponential backoff.
    Returns (result, warnings, debug_info) tuple. Warnings are collected instead of calling
    st.warning() directly, since this function may run in worker threads where
    Streamlit calls are not thread-safe.
    """
    warnings = []
    debug_info: Dict[str, Any] = {"user_prompt": "", "raw_response": None, "retries": 0}
    chunk_start = time.time()
    for attempt in range(max_retries):
        try:
            if provider == "google":
                result, chunk_warnings, chunk_debug = proofread_chunk_with_google_llm(client, model, chunk_text, system_prompt, chunk_info, input_token_limit=input_token_limit)
            else:
                result, chunk_warnings, chunk_debug = proofread_chunk_with_llm(client, model, chunk_text, system_prompt, chunk_info, max_completion_tokens=max_completion_tokens)
            warnings.extend(chunk_warnings)
            debug_info.update(chunk_debug)
            debug_info["retries"] = attempt
            if result is not None:
                debug_info["duration_seconds"] = time.time() - chunk_start
                debug_info["status"] = "success"
                return result, warnings, debug_info
            
            # If result is None but no exception, still retry
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                warnings.append(f"重試 {attempt + 1}/{max_retries}{chunk_info}：未取得結果（等待 {delay} 秒）")
                time.sleep(delay)
                
        except Exception as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                warnings.append(f"重試 {attempt + 1}/{max_retries}{chunk_info} 發生錯誤：{str(e)[:100]}...（等待 {delay} 秒）")
                time.sleep(delay)
            else:
                warnings.append(f"失敗{chunk_info}，已重試 {max_retries} 次：{str(e)}")
                debug_info["retries"] = attempt
                debug_info["duration_seconds"] = time.time() - chunk_start
                debug_info["status"] = "failed"
                debug_info["error_message"] = str(e)
                return None, warnings, debug_info
    
    debug_info["retries"] = max_retries - 1
    debug_info["duration_seconds"] = time.time() - chunk_start
    debug_info["status"] = "failed"
    return None, warnings, debug_info

def chunk_paragraphs(rdoc: RevisionDocument, chunk_size: int = 100, process_percentage: int = 100) -> List[Tuple[int, int, str]]:
    """
    Split document into chunks for processing.
    Returns list of (start_index, end_index, text) tuples.
    
    Args:
        rdoc: RevisionDocument to process
        chunk_size: Number of paragraphs per chunk
        process_percentage: Percentage of document to process (1-100)
    
    IMPORTANT: Includes ALL paragraphs (even empty ones) to maintain correct indexing.
    Empty paragraphs are marked as [i] (empty) so the LLM knows to skip them.
    """
    chunks = []
    total_paragraphs = len(rdoc.paragraphs)
    
    # Calculate how many paragraphs to process based on percentage
    paragraphs_to_process = max(1, int(total_paragraphs * process_percentage / 100))
    
    for start_idx in range(0, paragraphs_to_process, chunk_size):
        end_idx = min(start_idx + chunk_size, paragraphs_to_process)
        
        lines = []
        for i in range(start_idx, end_idx):
            text = rdoc.paragraphs[i].text.strip()
            if text:
                h = compute_paragraph_hash(text)
                lines.append(f"[{i}|{h}] {text}")
            else:
                # Include empty paragraphs to maintain correct indexing
                lines.append(f"[{i}] (empty paragraph)")
        
        chunk_text = "\n".join(lines)
        chunks.append((start_idx, end_idx, chunk_text))
    
    return chunks

def proofread_chunk_with_llm(
    client: OpenAI,
    model: str,
    chunk_text: str,
    system_prompt: str,
    chunk_info: str = "",
    max_completion_tokens: Optional[int] = None
) -> Tuple[Optional[ProofreadingResponse], List[str], Dict[str, Any]]:
    """
    Process a single chunk with the LLM.
    Returns (result, warnings, debug_info) tuple. Warnings are collected instead of calling
    st.*() directly, since this function may run in worker threads where
    Streamlit calls are not thread-safe.
    """
    warnings = []
    debug_info: Dict[str, Any] = {"user_prompt": "", "raw_response": None}
    try:
        user_prompt = f"""以下是需要校對的段落{chunk_info}:

{chunk_text}

"""

        debug_info["user_prompt"] = user_prompt

        create_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }
        if max_completion_tokens:
            create_kwargs["max_tokens"] = max_completion_tokens
        
        response = client.chat.completions.create(**create_kwargs)
        
        content = response.choices[0].message.content
        debug_info["raw_response"] = content
        
        if not content:
            # Gather diagnostic info from the response
            choice = response.choices[0] if response.choices else None
            diag = {
                "finish_reason": getattr(choice, "finish_reason", None) if choice else None,
                "refusal": getattr(choice.message, "refusal", None) if choice else None,
                "model": getattr(response, "model", None),
                "usage": getattr(response, "usage", None),
            }
            warnings.append(f"LLM 回傳空白回應{chunk_info}。診斷資訊：{diag}")
            return None, warnings, debug_info
        
        # Check for truncation before parsing
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            warnings.append(f"LLM 回應被截斷（finish_reason=length）{chunk_info}。"
                            f"區塊可能超出模型的輸出上限。")
            return None, warnings, debug_info
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError as je:
            warnings.append(f"無法解析 LLM 回應的 JSON{chunk_info}：{str(je)}")
            return None, warnings, debug_info
        
        if "edits" not in data:
            warnings.append(f"LLM 回應缺少 'edits' 欄位{chunk_info}")
            return ProofreadingResponse(edits=[], summary=data.get("summary", "未提供修改")), warnings, debug_info
        
        try:
            edits = [Edit(**edit) for edit in data.get("edits", [])]
        except Exception as validation_error:
            warnings.append(f"驗證修改資料失敗{chunk_info}：{str(validation_error)}")
            return None, warnings, debug_info
        
        return ProofreadingResponse(
            edits=edits,
            summary=data.get("summary", "無需修正。")
        ), warnings, debug_info
    except Exception as e:
        warnings.append(f"呼叫 LLM 時發生錯誤{chunk_info}：{str(e)}")
        return None, warnings, debug_info

def count_tokens_google(client: Any, model: str, text: str) -> Optional[int]:
    """
    Count exact tokens using Google Vertex AI count_tokens API.
    Returns token count or None if the call fails.
    """
    try:
        response = client.models.count_tokens(model=model, contents=text)
        return response.total_tokens
    except Exception:
        return None

def _build_google_user_prompt(chunk_text: str, chunk_info: str) -> str:
    """Build the user prompt for Google Vertex proofreading (shared by main and split paths)."""
    return f"""以下是需要校對的段落{chunk_info}:

{chunk_text}
"""

def proofread_chunk_with_google_llm(
    client: Any,
    model: str,
    chunk_text: str,
    system_prompt: str,
    chunk_info: str = "",
    input_token_limit: int = 0
) -> Tuple[Optional[ProofreadingResponse], List[str], Dict[str, Any]]:
    """Process a single chunk with Google Vertex AI Gemini model."""
    warnings = []
    debug_info: Dict[str, Any] = {"user_prompt": "", "raw_response": None}
    try:
        user_prompt = _build_google_user_prompt(chunk_text, chunk_info)
        debug_info["user_prompt"] = user_prompt

        # Verify token count before sending (safety net)
        if input_token_limit > 0:
            prompt_tokens = count_tokens_google(client, model, system_prompt + "\n" + user_prompt)
            if prompt_tokens is not None and prompt_tokens > int(input_token_limit * TOKEN_CUSHION_FACTOR):
                # Chunk too large — split in half and process each part
                lines = chunk_text.split("\n")
                mid = len(lines) // 2
                if mid > 0:
                    warnings.append(
                        f"區塊 token 數 ({prompt_tokens:,}) 超過安全上限 "
                        f"({int(input_token_limit * TOKEN_CUSHION_FACTOR):,}){chunk_info}，"
                        f"自動拆分為兩半重新處理。"
                    )
                    first_half = "\n".join(lines[:mid])
                    second_half = "\n".join(lines[mid:])
                    r1, w1, d1 = proofread_chunk_with_google_llm(
                        client, model, first_half, system_prompt,
                        f"{chunk_info}[上半]", input_token_limit
                    )
                    r2, w2, d2 = proofread_chunk_with_google_llm(
                        client, model, second_half, system_prompt,
                        f"{chunk_info}[下半]", input_token_limit
                    )
                    warnings.extend(w1)
                    warnings.extend(w2)
                    # Combine raw responses from sub-calls
                    sub_responses = [d1.get("raw_response"), d2.get("raw_response")]
                    debug_info["raw_response"] = "\n---SPLIT---\n".join(r for r in sub_responses if r)
                    combined_edits = []
                    combined_summary_parts = []
                    if r1:
                        combined_edits.extend(r1.edits)
                        combined_summary_parts.append(r1.summary)
                    if r2:
                        combined_edits.extend(r2.edits)
                        combined_summary_parts.append(r2.summary)
                    if combined_edits or combined_summary_parts:
                        return ProofreadingResponse(
                            edits=combined_edits,
                            summary="；".join(combined_summary_parts) if combined_summary_parts else "無需修正"
                        ), warnings, debug_info
                    return None, warnings, debug_info

        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={
                # "thinking_level": "medium",
                "system_instruction": system_prompt,
                "temperature": 0.1,
                "response_mime_type": "application/json",
            },
        )

        content = response.text
        debug_info["raw_response"] = content
        if not content:
            warnings.append(f"Google Vertex 回傳空白回應{chunk_info}")
            return None, warnings, debug_info

        try:
            data = json.loads(content)
        except json.JSONDecodeError as je:
            warnings.append(f"無法解析 Google Vertex 回應的 JSON{chunk_info}：{str(je)}")
            return None, warnings, debug_info

        if "edits" not in data:
            warnings.append(f"Google Vertex 回應缺少 'edits' 欄位{chunk_info}")
            return ProofreadingResponse(edits=[], summary=data.get("summary", "未提供修改")), warnings, debug_info

        try:
            edits = [Edit(**edit) for edit in data.get("edits", [])]
        except Exception as validation_error:
            warnings.append(f"驗證 Google Vertex 修改資料失敗{chunk_info}：{str(validation_error)}")
            return None, warnings, debug_info

        return ProofreadingResponse(
            edits=edits,
            summary=data.get("summary", "無需修正。")
        ), warnings, debug_info
    except Exception as e:
        error_text = str(e)
        if "NOT_FOUND" in error_text and "Publisher Model" in error_text:
            supported_models = ", ".join(GOOGLE_VERTEX_MODELS)
            warnings.append(
                f"呼叫 Google Vertex 時發生錯誤{chunk_info}：模型 `{model}` 在目前專案/區域不可用。"
                f"請改用可用模型（{supported_models}）或確認 Vertex AI 區域與模型存取權限。"
            )
        else:
            warnings.append(f"呼叫 Google Vertex 時發生錯誤{chunk_info}：{error_text}")
        return None, warnings, debug_info

def proofread_with_llm(
    client: Any,
    provider: str,
    model: str,
    rdoc: RevisionDocument,
    system_prompt: str,
    max_workers: int = 5,
    process_percentage: int = 100,
    model_info: Optional[Dict] = None,
    run_id: Optional[str] = None,
    supabase_client: Any = None
) -> Optional[ProofreadingResponse]:
    """
    Proofread document in chunks using parallel processing.
    Chunk size is computed dynamically from model token limits when model_info is available.
    If run_id and supabase_client are provided, logs per-chunk debug data.
    """
    from debug_logging import insert_chunk as log_insert_chunk

    total_paragraphs = len(rdoc.paragraphs)
    paragraphs_to_process = max(1, int(total_paragraphs * process_percentage / 100))
    
    # Compute chunk size dynamically or fall back to default
    if model_info and model_info.get("context_length", 0) > 0:
        chunk_size = compute_dynamic_chunk_size(
            rdoc, system_prompt,
            model_info["context_length"],
            model_info.get("max_completion_tokens", 0),
            process_percentage
        )
        st.info(f"📐 動態區塊大小：{chunk_size} 個段落"
                f"（模型上下文：{model_info['context_length']:,} tokens）")
    else:
        chunk_size = DEFAULT_CHUNK_SIZE
        st.info(f"📐 使用預設區塊大小：{chunk_size} 個段落（無法取得模型資訊）")
        if provider == "google":
            debug_key = f"model_info_debug_{provider}_{model}"
            debug_message = st.session_state.get(debug_key, "")
            if debug_message:
                st.caption(f"Google 模型資訊除錯：{debug_message}")
    
    chunks = chunk_paragraphs(rdoc, chunk_size, process_percentage)
    total_chunks = len(chunks)
    
    if total_chunks == 0:
        return ProofreadingResponse(edits=[], summary="文件為空")
    
    if process_percentage < 100:
        st.info(f"🧪 測試模式：處理文件的 {process_percentage}%（{paragraphs_to_process}/{total_paragraphs} 個段落）")
    
    actual_workers = min(max_workers, total_chunks)
    st.info(f"📦 將文件分為 {total_chunks} 個區塊（每區塊 {chunk_size} 個段落），使用 {actual_workers} 個平行工作執行緒")
    
    all_edits = []
    chunk_summaries = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    completed_chunks = 0
    
    # Process chunks in parallel with retry logic
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_chunk = {}
        for chunk_idx, (start_idx, end_idx, chunk_text) in enumerate(chunks):
            chunk_info = f" (區塊 {chunk_idx + 1}/{total_chunks}, 段落範圍 {start_idx}-{end_idx})"
            future = executor.submit(
                proofread_chunk_with_retry,
                client,
                provider,
                model,
                chunk_text,
                system_prompt,
                chunk_info,
                max_retries=DEFAULT_MAX_RETRIES,
                initial_delay=DEFAULT_RETRY_DELAY,
                max_completion_tokens=model_info.get("max_completion_tokens") if model_info else None,
                input_token_limit=model_info.get("context_length", 0) if model_info else 0
            )
            future_to_chunk[future] = (chunk_idx, start_idx, end_idx, chunk_text)
        
        # Collect results as they complete
        chunk_results = {}
        all_warnings = []
        for future in as_completed(future_to_chunk):
            chunk_idx, start_idx, end_idx, chunk_text = future_to_chunk[future]
            completed_chunks += 1
            
            status_text.text(f"已完成 {completed_chunks}/{total_chunks} 個區塊（最新：段落 {start_idx}-{end_idx}）...")
            
            try:
                result, warnings, debug_info = future.result()
                all_warnings.extend(warnings)
                if result:
                    chunk_results[chunk_idx] = result

                # Log chunk to Supabase (fire-and-forget)
                if run_id and supabase_client:
                    try:
                        log_insert_chunk(
                            supabase_client,
                            run_id=run_id,
                            chunk_index=chunk_idx,
                            start_paragraph=start_idx,
                            end_paragraph=end_idx,
                            chunk_text=chunk_text,
                            user_prompt=debug_info.get("user_prompt", ""),
                            system_prompt=system_prompt,
                            raw_response=debug_info.get("raw_response"),
                            parsed_result=result.model_dump() if result else None,
                            warnings=warnings,
                            retries=debug_info.get("retries", 0),
                            status=debug_info.get("status", "unknown"),
                            duration_seconds=debug_info.get("duration_seconds", 0.0),
                            error_message=debug_info.get("error_message"),
                        )
                    except Exception:
                        pass  # Never let logging break proofreading
            except Exception as e:
                all_warnings.append(f"處理區塊 {chunk_idx + 1} 時發生錯誤：{str(e)}")
            
            progress_bar.progress(completed_chunks / total_chunks)
    
    # Display collected warnings from the main thread (thread-safe)
    for warning_msg in all_warnings:
        st.warning(warning_msg)
    
    # Combine results in order
    for chunk_idx in sorted(chunk_results.keys()):
        result = chunk_results[chunk_idx]
        all_edits.extend(result.edits)
        if result.summary and result.summary != "無需修正。":
            chunk_summaries.append(f"區塊 {chunk_idx + 1}：{result.summary}")
    
    status_text.text("✅ 所有區塊處理完成！")
    
    # Combine summaries
    if chunk_summaries:
        combined_summary = f"已平行處理 {total_chunks} 個區塊。\n" + "\n".join(chunk_summaries)
    else:
        combined_summary = f"已平行處理 {total_chunks} 個區塊，無需修正。"
    
    return ProofreadingResponse(
        edits=all_edits,
        summary=combined_summary
    )

def render_tracked_changes_html(original: str, corrected: str) -> str:
    """
    Render inline tracked-changes HTML like Word's track changes view.
    Deletions shown as red strikethrough, insertions as green underline.
    """
    import html as html_mod
    matcher = difflib.SequenceMatcher(None, original, corrected)
    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            parts.append(html_mod.escape(original[i1:i2]))
        elif tag == 'delete':
            parts.append(
                f'<span style="color:#c0392b;text-decoration:line-through;background:#fde8e8;">'
                f'{html_mod.escape(original[i1:i2])}</span>'
            )
        elif tag == 'insert':
            parts.append(
                f'<span style="color:#27ae60;text-decoration:underline;background:#e8fde8;">'
                f'{html_mod.escape(corrected[j1:j2])}</span>'
            )
        elif tag == 'replace':
            parts.append(
                f'<span style="color:#c0392b;text-decoration:line-through;background:#fde8e8;">'
                f'{html_mod.escape(original[i1:i2])}</span>'
            )
            parts.append(
                f'<span style="color:#27ae60;text-decoration:underline;background:#e8fde8;">'
                f'{html_mod.escape(corrected[j1:j2])}</span>'
            )
    return ''.join(parts)


ZERO_WIDTH_BOUNDARY_CHARS = {"\u200B", "\u200C", "\u200D", "\uFEFF", "\u2060"}


def strip_boundary_whitespace_for_preview(text: str) -> str:
    """
    Strip boundary whitespace for UI preview only.
    Includes all Python-recognized whitespace plus common zero-width marks.
    """
    if not text:
        return text

    start = 0
    end = len(text)

    while start < end and (text[start].isspace() or text[start] in ZERO_WIDTH_BOUNDARY_CHARS):
        start += 1

    while end > start and (text[end - 1].isspace() or text[end - 1] in ZERO_WIDTH_BOUNDARY_CHARS):
        end -= 1

    return text[start:end]

def compute_character_diffs(original: str, corrected: str) -> List[Tuple[str, int, int, str]]:
    """
    Compute character-level differences between original and corrected text.
    Returns list of (operation, start, end, text) tuples:
    - ('delete', start, end, deleted_text)
    - ('insert', position, position, inserted_text)
    - ('replace', start, end, new_text)
    """
    diffs = []
    matcher = difflib.SequenceMatcher(None, original, corrected)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'delete':
            diffs.append(('delete', i1, i2, original[i1:i2]))
        elif tag == 'insert':
            diffs.append(('insert', i1, i1, corrected[j1:j2]))
        elif tag == 'replace':
            diffs.append(('replace', i1, i2, corrected[j1:j2]))
    
    return diffs

def apply_tracked_changes(
    rdoc: RevisionDocument,
    edits: List[Edit],
    author: str
) -> Dict[str, Any]:
    stats = {
        "deletions": 0, 
        "insertions": 0, 
        "errors": 0,
        "failed_edits": []
    }
    
    for edit in edits:
        try:
            if edit.paragraph_index < 0 or edit.paragraph_index >= len(rdoc.paragraphs):
                stats["errors"] += 1
                stats["failed_edits"].append({
                    "paragraph_index": edit.paragraph_index,
                    "reason": "out_of_range",
                    "total_paragraphs": len(rdoc.paragraphs),
                    "edit": edit.model_dump()
                })
                continue
            
            para_element = rdoc.paragraphs[edit.paragraph_index]
            current_text = para_element.text
            
            # Use the actual document text as the original for diffing
            original_text = current_text
            
            rp = RevisionParagraph.from_paragraph(para_element)
            diffs = compute_character_diffs(original_text, edit.corrected_text)
            
            # Process diffs in reverse order to maintain correct positions
            for op, start, end, text in reversed(diffs):
                if op == 'delete':
                    rp.add_tracked_deletion(
                        start=start,
                        end=end,
                        author=author
                    )
                    stats["deletions"] += 1
                    
                elif op == 'insert':
                    # For pure insertions, we need to insert at a position.
                    # replace_tracked_at requires start < end, so we "borrow"
                    # an adjacent character, delete it, and re-insert it
                    # alongside the new text.
                    if start >= len(original_text) or len(original_text) == 0:
                        # Insert at end or into empty paragraph
                        rp.add_tracked_insertion(
                            text=text,
                            author=author
                        )
                    elif start == 0:
                        # Insert at beginning - borrow the first character
                        rp.replace_tracked_at(
                            start=0,
                            end=1,
                            replace_text=text + original_text[0],
                            author=author
                        )
                    else:
                        # Insert in middle - borrow the character at position
                        rp.replace_tracked_at(
                            start=start,
                            end=start + 1,
                            replace_text=text + original_text[start],
                            author=author
                        )
                    stats["insertions"] += 1
                    
                elif op == 'replace':
                    # Use replace_tracked_at which handles both deletion and insertion at position
                    rp.replace_tracked_at(
                        start=start,
                        end=end,
                        replace_text=text,
                        author=author
                    )
                    stats["deletions"] += 1
                    stats["insertions"] += 1
                    
        except Exception as e:
            stats["errors"] += 1
            stats["failed_edits"].append({
                "paragraph_index": edit.paragraph_index,
                "reason": "exception",
                "error_message": str(e),
                "corrected_text": edit.corrected_text,
                "edit_reason": edit.reason
            })
    
    return stats

def enforce_workspace_auth() -> None:
    """Require Streamlit native login and restrict access to the allowed workspace domain."""
    user_email = (getattr(st.user, "email", "") or "").strip().lower()
    if not user_email:
        st.login()
        st.stop()

    if not user_email.endswith(ALLOWED_EMAIL_DOMAIN):
        st.error("❌ 未授權：僅允許 @red-publish.com 帳號存取此應用程式。")
        st.caption(f"目前登入帳號：{user_email or '未知'}")
        if st.button("登出", key="unauthorized_logout", type="primary"):
            st.logout()
        st.stop()

def main():
    enforce_workspace_auth()

    st.title("📝 紅出版 Word AI校對工具")
    st.markdown("上傳 Word 文件，讓 AI 進行校對並以追蹤修訂模式標記修改")
    
    # UI lock flag — disables all interactive elements while proofreading is running
    is_proofreading = st.session_state.get("is_proofreading", False)
    
    # Load prompts
    if 'prompts' not in st.session_state:
        st.session_state.prompts = load_prompts()
    
    prompts = st.session_state.prompts
    prompt_names = list(prompts.keys())
    
    with st.sidebar:
        st.header("⚙️ 設定")
        st.caption(f"已登入：{getattr(st.user, 'email', '未知帳號')}")
        if st.button("登出", key="sidebar_logout"):
            st.logout()
            st.stop()

        provider = st.selectbox(
            "供應商",
            options=LLM_PROVIDERS,
            index=0,
            help="選擇要使用的 LLM 供應商",
            disabled=is_proofreading
        )

        api_key = ""
        google_vertex_ready = False
        provider_models = OPENROUTER_MODELS if provider == "openrouter" else []

        if provider == "openrouter":
            api_key = st.secrets.get("openrouter", {}).get("api_key", "")
            if not api_key:
                st.warning("⚠️ 未在 secrets.toml 找到 OpenRouter API 金鑰（[openrouter].api_key）")

        if provider == "google":
            google_settings = st.secrets.get("google_vertex", {})
            raw_service_account = google_settings.get("service_account_json")
            project_id = google_settings.get("project_id")
            location = google_settings.get("location")

            if not project_id:
                st.error("⚠️ Google Vertex 設定無效：缺少 [google_vertex].project_id")
            elif not location:
                st.error("⚠️ Google Vertex 設定無效：缺少 [google_vertex].location")
            elif raw_service_account:
                try:
                    service_account_info = parse_google_service_account_info(raw_service_account)
                    google_vertex_ready = True

                    if GOOGLE_VERTEX_MODELS:
                        provider_models = GOOGLE_VERTEX_MODELS
                    else:
                        cache_key = (
                            f"google_vertex_models::{project_id}::{location}::"
                            f"{service_account_info.get('client_email', '')}"
                        )
                        if st.session_state.get("google_vertex_models_cache_key") != cache_key:
                            st.session_state.google_vertex_models_cache_key = cache_key
                            st.session_state.google_vertex_dynamic_models = []
                            st.session_state.google_vertex_dynamic_models_error = ""

                        if not st.session_state.get("google_vertex_dynamic_models"):
                            try:
                                google_client_for_models = get_google_vertex_client()
                                st.session_state.google_vertex_dynamic_models = fetch_google_vertex_models(google_client_for_models)
                                st.session_state.google_vertex_dynamic_models_error = ""
                            except Exception as fetch_error:
                                st.session_state.google_vertex_dynamic_models_error = str(fetch_error)

                        provider_models = st.session_state.get("google_vertex_dynamic_models", [])
                        if not provider_models and st.session_state.get("google_vertex_dynamic_models_error"):
                            st.warning(
                                "⚠️ 無法自動取得 Google 模型，"
                                f"錯誤：{st.session_state.google_vertex_dynamic_models_error}"
                            )
                        elif not provider_models:
                            st.warning("⚠️ Vertex AI 未回傳可用 Gemini 模型")
                except ValueError as config_error:
                    st.error(f"⚠️ Google Vertex 設定無效：{config_error}")
            else:
                st.warning("⚠️ 未在 secrets.toml 找到 Google Vertex 設定（[google_vertex].service_account_json）")

        model = ""
        if provider_models:
            model = st.selectbox(
                "模型",
                options=provider_models,
                index=0,
                help="選擇用於校對的 LLM 模型",
                disabled=is_proofreading
            )
        else:
            st.warning("⚠️ 目前沒有可用模型可供選擇")
        
        author_name = st.text_input(
            "作者名稱",
            value="紅出版",
            help="顯示在追蹤修訂中的名稱",
            disabled=is_proofreading
        )
        
        # ============================================================
        # Prompt Management
        # ============================================================
        st.markdown("---")
        st.subheader("📋 校對提示")
        
        # Initialize mode state
        if 'creating_new_prompt' not in st.session_state:
            st.session_state.creating_new_prompt = False
        
        # Mode toggle button
        if not st.session_state.creating_new_prompt:
            if st.button("➕ 新增提示", use_container_width=True, type="secondary", disabled=is_proofreading):
                st.session_state.creating_new_prompt = True
                st.session_state.new_prompt_name_input = ""
                st.session_state.confirm_delete_target = None
                st.rerun()
        else:
            if st.button("⬅️ 返回", use_container_width=True, type="secondary", disabled=is_proofreading):
                st.session_state.creating_new_prompt = False
                st.session_state.confirm_delete_target = None
                st.rerun()
        
        # ============================================================
        # NEW PROMPT MODE
        # ============================================================
        if st.session_state.creating_new_prompt:
            st.info("🆕 新增模式：輸入名稱並編輯內容")
            
            # New prompt name input
            new_prompt_name = st.text_input(
                "新提示名稱",
                value=st.session_state.get('new_prompt_name_input', ''),
                key="new_prompt_name",
                disabled=is_proofreading
            )
            
            # Template selector - use existing prompts as templates
            template_options = ["(空白)"] + list(prompts.keys())
            
            def on_template_change():
                selected = st.session_state.template_selector
                if selected != "(空白)":
                    st.session_state.new_prompt_content = prompts[selected]["content"]
            
            selected_template = st.selectbox(
                "從範本開始",
                options=template_options,
                key="template_selector",
                on_change=on_template_change,
                disabled=is_proofreading
            )
            
            # Reuse the textarea UI for new prompt content
            new_prompt_content = st.text_area(
                "提示內容",
                value="",
                height=250,
                help="編輯新提示的內容",
                key="new_prompt_content",
                disabled=is_proofreading
            )
            
            # Create button
            if st.button("💾 建立提示", use_container_width=True, type="secondary", key="create_new", disabled=is_proofreading):
                if not new_prompt_name or not new_prompt_name.strip():
                    st.error("請輸入提示名稱")
                elif not new_prompt_content or not new_prompt_content.strip():
                    st.error("請輸入提示內容")
                else:
                    success, message = add_prompt(prompts, new_prompt_name, new_prompt_content)
                    if success:
                        st.success(message)
                        st.session_state.prompts = load_prompts()
                        st.session_state.creating_new_prompt = False
                        # Store the newly created prompt name to auto-select it
                        st.session_state.newly_created_prompt = new_prompt_name
                        st.session_state.confirm_delete_target = None
                        st.rerun()
                    else:
                        st.error(message)
            
            # Use empty content for system prompt in new mode
            system_prompt = new_prompt_content
        
        # ============================================================
        # EDIT EXISTING PROMPT MODE
        # ============================================================
        else:
            # If we need to change the selectbox value, do it BEFORE the widget renders
            if 'newly_created_prompt' in st.session_state:
                st.session_state.prompt_selector = st.session_state.newly_created_prompt
                del st.session_state.newly_created_prompt
            
            if 'reset_prompt_selector' in st.session_state:
                st.session_state.prompt_selector = st.session_state.reset_prompt_selector
                del st.session_state.reset_prompt_selector
            
            # Callback when user changes selection - clear delete confirmation
            def on_selector_change():
                st.session_state.confirm_delete_target = None
            
            # Prompt selector - controlled via key "prompt_selector"
            selected_prompt_name = st.selectbox(
                "選擇提示",
                options=prompt_names,
                key="prompt_selector",
                on_change=on_selector_change,
                help="選擇要使用的提示範本",
                disabled=is_proofreading
            )
            
            # Get selected prompt details
            selected_prompt_data = prompts[selected_prompt_name]
            is_protected = selected_prompt_data.get("protected", False)
            original_content = selected_prompt_data["content"]
            
            # Callback to track content changes
            def on_prompt_change():
                st.session_state.prompt_modified = True
            
            # Prompt content textarea (read-only for protected prompts)
            current_prompt_content = st.text_area(
                "提示內容",
                value=original_content,
                height=250,
                help="此提示受保護，無法編輯" if is_protected else "編輯提示內容",
                key=f"prompt_content_{selected_prompt_name}",
                disabled=is_protected or is_proofreading,
                on_change=on_prompt_change
            )
            
            if is_protected:
                st.caption("🔒 此提示受保護，無法儲存修改或刪除")
            
            # Check if content has been modified
            content_modified = current_prompt_content != original_content
            
            # Save button - only enabled when content is modified and not protected
            save_disabled = is_protected or not content_modified or is_proofreading
            if st.button(
                f"💾 儲存",
                disabled=save_disabled,
                use_container_width=True,
                key="save_current",
                type="primary" if content_modified and not is_protected else "secondary"
            ):
                success, message = update_prompt(prompts, selected_prompt_name, current_prompt_content)
                if success:
                    st.success(message)
                    st.session_state.prompts = load_prompts()
                    st.session_state.prompt_modified = False
                    st.rerun()
                else:
                    st.error(message)
            
            # Delete button with confirmation
            delete_disabled = is_protected or is_proofreading
            
            # Initialize confirmation state
            if 'confirm_delete_target' not in st.session_state:
                st.session_state.confirm_delete_target = None
            
            # Check if we're confirming deletion for the currently selected prompt
            is_confirming = st.session_state.confirm_delete_target == selected_prompt_name
            
            if not is_confirming:
                if st.button(
                    f"🗑️ 刪除",
                    disabled=delete_disabled,
                    use_container_width=True,
                    key="delete_current"
                ):
                    st.session_state.confirm_delete_target = selected_prompt_name
                    st.rerun()
            else:
                st.warning(f"⚠️ 確定要刪除 '{selected_prompt_name}' 嗎？")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ 確認刪除", use_container_width=True, type="primary", key="confirm_delete_yes", disabled=is_proofreading):
                        success, message = delete_prompt(prompts, selected_prompt_name)
                        if success:
                            st.success(message)
                            st.session_state.prompts = load_prompts()
                            st.session_state.confirm_delete_target = None
                            # Schedule selector reset for next rerun (can't modify after widget renders)
                            st.session_state.reset_prompt_selector = prompt_names[0]
                            st.rerun()
                        else:
                            st.error(message)
                            st.session_state.confirm_delete_target = None
                
                with col2:
                    if st.button("❌ 取消", use_container_width=True, key="confirm_delete_no", disabled=is_proofreading):
                        st.session_state.confirm_delete_target = None
                        st.rerun()
            
            # Set the system prompt to use (use current edited content)
            system_prompt = current_prompt_content
        
        st.markdown("---")
        
        with st.expander("🧪 測試選項", expanded=False):
            process_percentage = st.slider(
                "處理文件的百分比",
                min_value=1,
                max_value=100,
                value=100,
                step=1,
                help="測試用：僅處理文件的一部分（例如 10% = 前 10% 的段落）",
                disabled=is_proofreading
            )
            st.caption(f"將處理文件的 {process_percentage}%")
        
        st.markdown("---")
        st.markdown("### 關於")
        st.markdown("本應用程式使用 AI 校對 Word 文件，並以追蹤修訂模式加入修正。")        
    
    uploaded_file = st.file_uploader(
        "上傳 Word 文件 (.docx)",
        type=["docx"],
        help="選擇要校對的 .docx 檔案",
        disabled=is_proofreading
    )
    
    if uploaded_file is not None:
        if provider == "openrouter" and not api_key:
            st.warning("⚠️ 請在 secrets.toml 設定 OpenRouter API 金鑰（[openrouter].api_key）")
            return
        if provider == "google" and not google_vertex_ready:
            st.warning("⚠️ 請在 secrets.toml 設定 Google Vertex 憑證（[google_vertex].service_account_json）")
            return
        
        file_bytes = uploaded_file.getvalue()
        current_doc_signature = hashlib.md5(file_bytes).hexdigest()

        # Clear cached proofreading result when user uploads a different file
        if st.session_state.get("proofread_result_doc_signature") != current_doc_signature:
            st.session_state.pop("proofread_result_data", None)
            st.session_state.pop("proofread_stats", None)
            st.session_state.pop("proofread_output_data", None)
            st.session_state.pop("proofread_output_filename", None)
            st.session_state.pop("edit_page", None)

        tmp_input_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_input:
            tmp_input_path = tmp_input.name
            tmp_input.write(file_bytes)
        
        try:
            with st.spinner("📖 讀取文件中..."):
                rdoc = RevisionDocument(tmp_input_path)
                document_text = read_document_paragraphs(rdoc)
            
            # Check for pending tracked changes
            has_changes, change_count = check_for_tracked_changes(rdoc)
            if has_changes:
                st.error(f"❌ 此文件包含 {change_count} 個未處理的追蹤修訂")
                st.warning("⚠️ 請先在 Microsoft Word 中接受或拒絕所有追蹤修訂，然後重新上傳文件。")
                st.info("💡 在 Word 中：審閱 → 接受 → 接受所有修訂（或逐一檢視）")
                return
            
            st.success(f"✅ 已載入文件，共 {len(rdoc.paragraphs)} 個段落")
            
            with st.expander("📄 文件預覽", expanded=False):
                show_full = st.checkbox("顯示完整文件", value=False)
                
                if show_full:
                    paragraphs_list = document_text.split('\n')
                    total_paragraphs = len(paragraphs_list)
                    
                    # Pagination settings
                    paragraphs_per_page = DEFAULT_PARAGRAPHS_PER_PAGE
                    total_pages = (total_paragraphs + paragraphs_per_page - 1) // paragraphs_per_page
                    
                    page = st.number_input(
                        f"頁面 (1-{total_pages})",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        step=1
                    )
                    
                    start_idx = (page - 1) * paragraphs_per_page
                    end_idx = min(start_idx + paragraphs_per_page, total_paragraphs)
                    
                    st.caption(f"顯示第 {start_idx + 1}-{end_idx} 行，共 {total_paragraphs} 行（非空段落）")
                    st.text('\n'.join(paragraphs_list[start_idx:end_idx]))
                else:
                    st.text(document_text[:2000] + ("..." if len(document_text) > 2000 else ""))
                    if len(document_text) > 2000:
                        st.caption(f"顯示前 2000 個字元。勾選「顯示完整文件」以查看更多內容。")
            
            if st.button("🚀 開始校對", type="primary", use_container_width=True, disabled=is_proofreading):
                if not model:
                    st.error("⚠️ 尚未取得可用模型，請確認供應商設定後重試")
                    st.stop()
                st.session_state["is_proofreading"] = True
                st.rerun()
            
            # Run proofreading on the rerun where all widgets are already disabled
            if is_proofreading:
                run_start_time = time.time()
                debug_run_id = None
                try:
                    from debug_logging import create_run as log_create_run, update_run_completed as log_update_run_completed, update_run_failed as log_update_run_failed

                    model_info = None
                    if provider == "openrouter":
                        client = get_openrouter_client(api_key)

                        # Fetch and cache model info for dynamic chunk sizing
                        cache_key = f"model_info_{provider}_{model}"
                        if cache_key not in st.session_state or st.session_state.get(cache_key) is None:
                            with st.spinner("📡 取得模型資訊..."):
                                st.session_state[cache_key] = fetch_model_info(api_key, model)
                        model_info = st.session_state[cache_key]
                    else:
                        client = get_google_vertex_client()

                        # Fetch and cache model info for dynamic chunk sizing
                        cache_key = f"model_info_{provider}_{model}"
                        if cache_key not in st.session_state or st.session_state.get(cache_key) is None:
                            with st.spinner("📡 取得模型資訊..."):
                                google_model_info, google_model_debug = fetch_google_model_info(client, model)
                                st.session_state[cache_key] = google_model_info
                                st.session_state[f"model_info_debug_{provider}_{model}"] = google_model_debug
                        model_info = st.session_state[cache_key]

                    # Compute chunk info for logging before calling proofread_with_llm
                    total_paragraphs_count = len(rdoc.paragraphs)
                    paragraphs_to_process_count = max(1, int(total_paragraphs_count * process_percentage / 100))
                    if model_info and model_info.get("context_length", 0) > 0:
                        log_chunk_size = compute_dynamic_chunk_size(
                            rdoc, system_prompt,
                            model_info["context_length"],
                            model_info.get("max_completion_tokens", 0),
                            process_percentage
                        )
                    else:
                        log_chunk_size = DEFAULT_CHUNK_SIZE
                    log_total_chunks = (paragraphs_to_process_count + log_chunk_size - 1) // log_chunk_size

                    # Build document paragraphs snapshot for logging
                    doc_paragraphs = [
                        {"index": i, "text": rdoc.paragraphs[i].text}
                        for i in range(total_paragraphs_count)
                    ]

                    # Determine the prompt name used
                    log_prompt_name = selected_prompt_name if not st.session_state.get("creating_new_prompt") else "(新建提示)"

                    # Create debug run record
                    try:
                        sb = get_supabase_client()
                        debug_run_id = log_create_run(
                            sb,
                            user_email=st.user.email if hasattr(st, "user") and st.user else "unknown",
                            file_name=uploaded_file.name,
                            file_hash=current_doc_signature,
                            provider=provider,
                            model=model,
                            prompt_name=log_prompt_name,
                            prompt_content=system_prompt,
                            process_percentage=process_percentage,
                            total_paragraphs=total_paragraphs_count,
                            paragraphs_processed=paragraphs_to_process_count,
                            chunk_size=log_chunk_size,
                            total_chunks=log_total_chunks,
                            model_info=model_info,
                            document_paragraphs=doc_paragraphs,
                        )
                    except Exception:
                        sb = None
                        debug_run_id = None
                    
                    with st.spinner(f"🤖 使用 {provider}/{model} 校對中..."):
                        result = proofread_with_llm(
                            client,
                            provider,
                            model,
                            rdoc,
                            system_prompt,
                            max_workers=DEFAULT_MAX_WORKERS,
                            process_percentage=process_percentage,
                            model_info=model_info,
                            run_id=debug_run_id,
                            supabase_client=sb if debug_run_id else None
                        )
                    
                    if result is None:
                        st.error("❌ 無法取得校對結果")
                        # Log run failure
                        if debug_run_id and sb:
                            try:
                                log_update_run_failed(
                                    sb, debug_run_id,
                                    error_message="proofread_with_llm returned None",
                                    duration_seconds=time.time() - run_start_time,
                                )
                            except Exception:
                                pass
                    else:
                        st.session_state["proofread_result_data"] = result.model_dump()
                        st.session_state["proofread_result_doc_signature"] = current_doc_signature
                        st.session_state["edit_page"] = 1

                        # Log run success
                        if debug_run_id and sb:
                            try:
                                log_update_run_completed(
                                    sb, debug_run_id,
                                    total_edits=len(result.edits),
                                    duration_seconds=time.time() - run_start_time,
                                    combined_summary=result.summary,
                                    warnings=[],
                                )
                            except Exception:
                                pass

                        # Snapshot original paragraph texts before apply_tracked_changes modifies rdoc
                        st.session_state["original_paragraph_texts"] = {
                            i: rdoc.paragraphs[i].text for i in range(len(rdoc.paragraphs))
                        }

                        if result.edits:
                            with st.spinner("✏️ 套用追蹤修訂中..."):
                                stats = apply_tracked_changes(rdoc, result.edits, author_name)

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_output:
                                tmp_output_path = tmp_output.name

                            rdoc.save(tmp_output_path)

                            with open(tmp_output_path, "rb") as f:
                                output_data = f.read()

                            os.unlink(tmp_output_path)

                            original_name = uploaded_file.name.rsplit(".", 1)[0]
                            output_filename = f"{original_name}_proofread.docx"

                            st.session_state["proofread_stats"] = stats
                            st.session_state["proofread_output_data"] = output_data
                            st.session_state["proofread_output_filename"] = output_filename
                        else:
                            st.session_state["proofread_stats"] = None
                            st.session_state["proofread_output_data"] = None
                            st.session_state["proofread_output_filename"] = None
                except Exception as exc:
                    # Log unexpected failure
                    if debug_run_id:
                        try:
                            sb = get_supabase_client()
                            log_update_run_failed(
                                sb, debug_run_id,
                                error_message=str(exc),
                                duration_seconds=time.time() - run_start_time,
                            )
                        except Exception:
                            pass
                    raise
                finally:
                    st.session_state["is_proofreading"] = False

            stored_result_data = st.session_state.get("proofread_result_data")
            if stored_result_data and st.session_state.get("proofread_result_doc_signature") == current_doc_signature:
                result = ProofreadingResponse(**stored_result_data)

                if not result.edits:
                    st.info("✨ 無需修正！您的文件看起來很棒。")
                    st.markdown(f"**AI 摘要：** {result.summary}")
                else:
                    st.success(f"✅ 找到 {len(result.edits)} 個建議修正")
                    st.markdown(f"**摘要：** {result.summary}")

                    with st.expander("📝 建議修改", expanded=True):
                        total_edits = len(result.edits)

                        if total_edits > DEFAULT_EDITS_PER_PAGE:
                            edits_per_page = DEFAULT_EDITS_PER_PAGE
                            total_edit_pages = (total_edits + edits_per_page - 1) // edits_per_page

                            # Initialize and clamp page
                            if 'edit_page' not in st.session_state:
                                st.session_state.edit_page = 1
                            st.session_state.edit_page = max(1, min(st.session_state.edit_page, total_edit_pages))
                            
                            page_options = list(range(1, total_edit_pages + 1))

                            # Sync both selectbox keys to current page BEFORE widgets render
                            st.session_state["ep_sel_top"] = st.session_state.edit_page
                            st.session_state["ep_sel_bottom"] = st.session_state.edit_page

                            def _on_select_change(key_suffix):
                                st.session_state.edit_page = st.session_state[f"ep_sel_{key_suffix}"]

                            def _on_prev():
                                st.session_state.edit_page = max(1, st.session_state.edit_page - 1)

                            def _on_next():
                                st.session_state.edit_page = min(total_edit_pages, st.session_state.edit_page + 1)

                            def _render_edit_pagination(key_suffix: str):
                                """Render [Prev] [Dropdown] [Next] pagination row."""
                                col_prev, col_select, col_next = st.columns([1, 2, 1])
                                with col_prev:
                                    st.button("⬅️ 上一頁", key=f"ep_prev_{key_suffix}",
                                              disabled=(st.session_state.edit_page <= 1),
                                              use_container_width=True, on_click=_on_prev)
                                with col_select:
                                    st.selectbox(
                                        "頁面",
                                        options=page_options,
                                        format_func=lambda x: f"第 {x} / {total_edit_pages} 頁",
                                        key=f"ep_sel_{key_suffix}",
                                        label_visibility="collapsed",
                                        on_change=_on_select_change,
                                        args=(key_suffix,)
                                    )
                                with col_next:
                                    st.button("下一頁 ➡️", key=f"ep_next_{key_suffix}",
                                              disabled=(st.session_state.edit_page >= total_edit_pages),
                                              use_container_width=True, on_click=_on_next)

                            # Top pagination
                            _render_edit_pagination("top")

                            start_edit = (st.session_state.edit_page - 1) * edits_per_page
                            end_edit = min(start_edit + edits_per_page, total_edits)
                            edits_to_show = result.edits[start_edit:end_edit]
                            edit_offset = start_edit
                        else:
                            edits_to_show = result.edits
                            edit_offset = 0

                        ignore_boundary_ws = st.checkbox(
                            "忽略段落前後的空白",
                            value=True,
                            key="ignore_boundary_whitespace_main",
                        )

                        for i, edit in enumerate(edits_to_show, edit_offset + 1):
                            st.markdown(f"**修改 {i}** (段落 {edit.paragraph_index})")

                            # Look up original text from snapshot (before apply_tracked_changes modified rdoc)
                            orig_texts = st.session_state.get("original_paragraph_texts", {})
                            if edit.paragraph_index in orig_texts:
                                original_text = orig_texts[edit.paragraph_index]
                            elif 0 <= edit.paragraph_index < len(rdoc.paragraphs):
                                original_text = rdoc.paragraphs[edit.paragraph_index].text
                            else:
                                original_text = "(段落索引超出範圍)"

                            preview_original = original_text
                            preview_corrected = edit.corrected_text
                            if ignore_boundary_ws:
                                preview_original = strip_boundary_whitespace_for_preview(preview_original)
                                preview_corrected = strip_boundary_whitespace_for_preview(preview_corrected)

                            tracked_html = render_tracked_changes_html(preview_original, preview_corrected)
                            st.markdown(
                                f'<div style="padding:0.75em 1em;border:1px solid #ddd;border-radius:6px;'
                                f'line-height:1.8;font-size:1rem;white-space:pre-wrap;">'
                                f'{tracked_html}</div>',
                                unsafe_allow_html=True
                            )

                            st.caption(f"💡 {edit.reason}")
                            st.markdown("---")

                        # Bottom pagination (only if paginated)
                        if total_edits > DEFAULT_EDITS_PER_PAGE:
                            _render_edit_pagination("bottom")

                    stats = st.session_state.get("proofread_stats")
                    if stats:
                        st.info(f"📊 已套用 {stats['deletions']} 個刪除和 {stats['insertions']} 個插入")

                        if stats['errors'] > 0:
                            st.warning(f"⚠️ {stats['errors']} 個修改無法套用")

                            # Debug interface for failed edits
                            with st.expander("🔍 除錯：失敗修改詳情", expanded=False):
                                st.markdown("### 失敗修改分析")
                                st.caption(f"總失敗數：{len(stats['failed_edits'])}")

                                # Group by failure reason
                                range_failures = [f for f in stats['failed_edits'] if f['reason'] == 'out_of_range']
                                exception_failures = [f for f in stats['failed_edits'] if f['reason'] == 'exception']

                                st.markdown(f"- **超出範圍**：{len(range_failures)}")
                                st.markdown(f"- **例外錯誤**：{len(exception_failures)}")

                                # Show out of range failures
                                if range_failures:
                                    st.markdown("### 📍 超出範圍失敗")
                                    for failure in range_failures:
                                        st.markdown(f"- 段落 {failure['paragraph_index']}（文件共有 {failure['total_paragraphs']} 個段落）")

                                # Show exception failures
                                if exception_failures:
                                    st.markdown("### ⚠️ 例外錯誤失敗")
                                    for i, failure in enumerate(exception_failures[:10], 1):
                                        st.markdown(f"**例外 {i}** - 段落 {failure['paragraph_index']}")
                                        st.code(failure['error_message'])
                                        st.caption(f"修正內容：{failure['corrected_text'][:100]}")
                                        st.markdown("---")

                                # Export debug data
                                st.markdown("### 💾 匯出除錯資料")
                                debug_json = json.dumps(stats['failed_edits'], indent=2, ensure_ascii=False)
                                st.download_button(
                                    label="下載失敗修改 JSON",
                                    data=debug_json,
                                    file_name="failed_edits_debug.json",
                                    mime="application/json"
                                )

                    output_data = st.session_state.get("proofread_output_data")
                    output_filename = st.session_state.get("proofread_output_filename")
                    if output_data and output_filename:
                        st.download_button(
                            label="⬇️ 下載校對後的文件",
                            data=output_data,
                            file_name=output_filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            type="primary",
                            use_container_width=True
                        )

                        st.success("✅ 文件已準備好！在 Microsoft Word 中開啟以檢視追蹤修訂。")
        
        except Exception as e:
            st.error(f"❌ 處理文件時發生錯誤：{str(e)}")
        finally:
            if tmp_input_path and os.path.exists(tmp_input_path):
                os.unlink(tmp_input_path)
    else:        
        
        st.markdown("### 使用方法")
        st.markdown("""
        1. **上傳**您的 Word 文件 (.docx)
        2. **設定** AI 模型和校對提示
        3. **檢視**建議的修正
        4. **下載**帶有追蹤修訂的文件
        5. **在 Microsoft Word 中開啟**以接受/拒絕修改
        """)

if __name__ == "__main__":
    main()
