"""
Debug logging helpers for proofreading runs.
Stores run metadata and per-chunk data in Supabase for debugging.
All functions are fire-and-forget: exceptions are caught and printed
so logging never breaks the proofreading pipeline.
"""

from typing import Any, Dict, List, Optional


def create_run(
    supabase: Any,
    *,
    user_email: str,
    file_name: str,
    file_hash: str,
    provider: str,
    model: str,
    prompt_name: str,
    prompt_content: str,
    process_percentage: int,
    total_paragraphs: int,
    paragraphs_processed: int,
    chunk_size: int,
    total_chunks: int,
    model_info: Optional[Dict[str, Any]],
    document_paragraphs: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Insert a new proofreading run row with status='running'.
    Returns the run_id (uuid string) or None on failure.
    """
    try:
        row = {
            "user_email": user_email,
            "file_name": file_name,
            "file_hash": file_hash,
            "provider": provider,
            "model": model,
            "prompt_name": prompt_name,
            "prompt_content": prompt_content,
            "process_percentage": process_percentage,
            "total_paragraphs": total_paragraphs,
            "paragraphs_processed": paragraphs_processed,
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "status": "running",
            "model_info": model_info,
            "document_paragraphs": document_paragraphs,
        }
        response = supabase.table("proofreading_runs").insert(row).execute()
        return response.data[0]["id"] if response.data else None
    except Exception as e:
        print(f"[debug_logging] create_run failed: {e}")
        return None


def insert_chunk(
    supabase: Any,
    *,
    run_id: str,
    chunk_index: int,
    start_paragraph: int,
    end_paragraph: int,
    chunk_text: str,
    user_prompt: str,
    system_prompt: str,
    raw_response: Optional[str],
    parsed_result: Optional[Dict[str, Any]],
    warnings: List[str],
    retries: int,
    status: str,
    duration_seconds: float,
    error_message: Optional[str] = None,
) -> None:
    """Insert a chunk result row. Fire-and-forget."""
    try:
        row = {
            "run_id": run_id,
            "chunk_index": chunk_index,
            "start_paragraph": start_paragraph,
            "end_paragraph": end_paragraph,
            "chunk_text": chunk_text,
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "raw_response": raw_response,
            "parsed_result": parsed_result,
            "warnings": warnings,
            "retries": retries,
            "status": status,
            "duration_seconds": duration_seconds,
            "error_message": error_message,
        }
        supabase.table("proofreading_chunks").insert(row).execute()
    except Exception as e:
        print(f"[debug_logging] insert_chunk failed (run={run_id}, chunk={chunk_index}): {e}")


def update_run_completed(
    supabase: Any,
    run_id: str,
    *,
    total_edits: int,
    duration_seconds: float,
    combined_summary: str,
    warnings: List[str],
) -> None:
    """Mark a run as completed with final stats. Fire-and-forget."""
    try:
        supabase.table("proofreading_runs").update({
            "status": "completed",
            "total_edits": total_edits,
            "duration_seconds": duration_seconds,
            "combined_summary": combined_summary,
            "warnings": warnings,
        }).eq("id", run_id).execute()
    except Exception as e:
        print(f"[debug_logging] update_run_completed failed (run={run_id}): {e}")


def update_run_failed(
    supabase: Any,
    run_id: str,
    *,
    error_message: str,
    duration_seconds: float,
    warnings: Optional[List[str]] = None,
) -> None:
    """Mark a run as failed. Fire-and-forget."""
    try:
        update = {
            "status": "failed",
            "error_message": error_message,
            "duration_seconds": duration_seconds,
        }
        if warnings is not None:
            update["warnings"] = warnings
        supabase.table("proofreading_runs").update(update).eq("id", run_id).execute()
    except Exception as e:
        print(f"[debug_logging] update_run_failed failed (run={run_id}): {e}")
