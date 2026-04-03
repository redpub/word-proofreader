import streamlit as st
import difflib
import html as html_mod
import json
from datetime import datetime
from db_tables import get_table_name


ALLOWED_EMAIL_DOMAIN = "@red-publish.com"


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


@st.cache_resource
def get_supabase_client():
    from supabase import create_client
    url = st.secrets["connections"]["supabase"]["SUPABASE_URL"]
    key = st.secrets["connections"]["supabase"]["SUPABASE_KEY"]
    return create_client(url, key)


def load_runs(limit: int = 50):
    """Load recent proofreading runs."""
    sb = get_supabase_client()
    response = (
        sb.table(get_table_name("proofreading_runs"))
        .select("id, created_at, user_email, file_name, provider, model, prompt_name, "
                "process_percentage, total_paragraphs, total_chunks, total_edits, "
                "duration_seconds, status, chunk_size")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def load_run_detail(run_id: str):
    """Load full run record."""
    sb = get_supabase_client()
    response = (
        sb.table(get_table_name("proofreading_runs"))
        .select("*")
        .eq("id", run_id)
        .single()
        .execute()
    )
    return response.data


def load_chunks(run_id: str):
    """Load all chunks for a run, ordered by chunk_index."""
    sb = get_supabase_client()
    response = (
        sb.table(get_table_name("proofreading_chunks"))
        .select("*")
        .eq("run_id", run_id)
        .order("chunk_index")
        .execute()
    )
    return response.data or []


def format_duration(seconds):
    """Format seconds into readable duration."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def format_timestamp(ts_str):
    """Format ISO timestamp for display."""
    if not ts_str:
        return "—"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts_str


def status_emoji(status):
    if status == "completed":
        return "✅"
    elif status == "failed":
        return "❌"
    elif status == "running":
        return "⏳"
    return "❓"


def render_tracked_changes_html(original: str, corrected: str) -> str:
    """
    Render inline tracked-changes HTML like Word's track changes view.
    Deletions shown as red strikethrough, insertions as green underline.
    """
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
    """Strip leading/trailing whitespace (including zero-width marks) for preview only."""
    if not text:
        return text

    start = 0
    end = len(text)

    while start < end and (text[start].isspace() or text[start] in ZERO_WIDTH_BOUNDARY_CHARS):
        start += 1

    while end > start and (text[end - 1].isspace() or text[end - 1] in ZERO_WIDTH_BOUNDARY_CHARS):
        end -= 1

    return text[start:end]


def main():
    enforce_workspace_auth()

    st.title("🔍 校對除錯紀錄")
    st.caption("查看所有校對執行的詳細紀錄，包含每個區塊的提示和 LLM 回應")

    # ── Run list ──
    if "selected_run_id" not in st.session_state:
        st.session_state.selected_run_id = None

    runs = load_runs()

    if not runs:
        st.info("目前沒有校對紀錄。")
        return

    # Back button when viewing a run
    if st.session_state.selected_run_id:
        if st.button("⬅️ 返回列表"):
            st.session_state.selected_run_id = None
            st.rerun()

        show_run_detail(st.session_state.selected_run_id)
        return

    # ── Runs table ──
    st.subheader(f"最近 {len(runs)} 次校對")

    for run in runs:
        status = status_emoji(run.get("status", ""))
        ts = format_timestamp(run.get("created_at"))
        dur = format_duration(run.get("duration_seconds"))
        edits = run.get("total_edits")
        edits_str = str(edits) if edits is not None else "—"

        col_status, col_time, col_file, col_model, col_edits, col_dur, col_btn = st.columns(
            [0.5, 1.5, 2, 2, 0.8, 0.8, 1]
        )
        with col_status:
            st.write(status)
        with col_time:
            st.caption(ts)
        with col_file:
            st.write(f"**{run.get('file_name', '—')}**")
        with col_model:
            st.caption(f"{run.get('provider', '')}/{run.get('model', '')}")
        with col_edits:
            st.write(edits_str)
        with col_dur:
            st.caption(dur)
        with col_btn:
            if st.button("查看", key=f"view_{run['id']}"):
                st.session_state.selected_run_id = run["id"]
                st.rerun()

        st.markdown("<hr style='margin:2px 0;border:none;border-top:1px solid #eee;'>", unsafe_allow_html=True)


def show_run_detail(run_id: str):
    """Display full detail for a single run."""
    run = load_run_detail(run_id)
    if not run:
        st.error("找不到此紀錄")
        return

    # ── Header ──
    status = run.get("status", "unknown")
    st.subheader(f"{status_emoji(status)} {run.get('file_name', '—')}")

    # ── Metadata grid ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("狀態", status)
        st.metric("使用者", run.get("user_email", "—"))
    with c2:
        st.metric("供應商 / 模型", f"{run.get('provider')} / {run.get('model')}")
        st.metric("提示名稱", run.get("prompt_name", "—"))
    with c3:
        st.metric("總段落", run.get("total_paragraphs"))
        st.metric("已處理段落", run.get("paragraphs_processed"))
    with c4:
        st.metric("區塊大小", run.get("chunk_size"))
        st.metric("總區塊", run.get("total_chunks"))

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("總修改數", run.get("total_edits") if run.get("total_edits") is not None else "—")
    with c6:
        st.metric("耗時", format_duration(run.get("duration_seconds")))
    with c7:
        st.metric("處理百分比", f"{run.get('process_percentage', 100)}%")
    with c8:
        st.metric("時間", format_timestamp(run.get("created_at")))

    # ── Error message ──
    if run.get("error_message"):
        st.error(f"**錯誤：** {run['error_message']}")

    # ── Combined summary ──
    if run.get("combined_summary"):
        with st.expander("📋 摘要", expanded=False):
            st.text(run["combined_summary"])

    # ── System prompt ──
    with st.expander("📝 系統提示 (System Prompt)", expanded=False):
        st.code(run.get("prompt_content", ""), language=None)

    # ── Model info ──
    if run.get("model_info"):
        with st.expander("🔧 模型資訊", expanded=False):
            st.json(run["model_info"])

    # ── Warnings ──
    warnings = run.get("warnings") or []
    if warnings:
        with st.expander(f"⚠️ 警告 ({len(warnings)})", expanded=False):
            for w in warnings:
                st.warning(w)

    # ── Document paragraphs ──
    doc_paras = run.get("document_paragraphs") or []
    if doc_paras:
        with st.expander(f"📄 文件段落 ({len(doc_paras)} 段)", expanded=False):
            # Pagination for large documents
            per_page = 50
            total_pages = max(1, (len(doc_paras) + per_page - 1) // per_page)
            if total_pages > 1:
                page = st.selectbox(
                    "頁面",
                    list(range(1, total_pages + 1)),
                    format_func=lambda x: f"第 {x} / {total_pages} 頁",
                    key="doc_para_page"
                )
            else:
                page = 1
            start = (page - 1) * per_page
            end = min(start + per_page, len(doc_paras))
            for p in doc_paras[start:end]:
                idx = p.get("index", "?")
                text = p.get("text", "")
                if text.strip():
                    st.text(f"[{idx}] {text}")
                else:
                    st.caption(f"[{idx}] (空段落)")

    # ── Chunks ──
    st.markdown("---")
    st.subheader("📦 區塊詳情")

    chunks = load_chunks(run_id)
    if not chunks:
        st.info("此紀錄沒有區塊資料。")
        return

    # Build paragraph index lookup from run's document_paragraphs
    para_lookup = {}
    for p in doc_paras:
        para_lookup[p.get("index")] = p.get("text", "")

    for chunk in chunks:
        cidx = chunk.get("chunk_index", "?")
        cstatus = status_emoji(chunk.get("status", ""))
        cdur = format_duration(chunk.get("duration_seconds"))
        retries = chunk.get("retries", 0)

        header = (
            f"{cstatus} **區塊 {cidx}** "
            f"(段落 {chunk.get('start_paragraph')}–{chunk.get('end_paragraph')}) "
            f"| {cdur} | 重試 {retries}x"
        )

        with st.expander(header, expanded=False):
            if chunk.get("error_message"):
                st.error(f"錯誤：{chunk['error_message']}")

            # Chunk warnings
            cwarnings = chunk.get("warnings") or []
            if cwarnings:
                for w in cwarnings:
                    st.warning(w)

            tab_prompt, tab_response, tab_parsed, tab_preview = st.tabs(
                ["📤 提示", "📥 原始回應", "📊 解析結果", "📝 修改預覽"]
            )

            with tab_prompt:
                st.markdown("**User Prompt：**")
                st.code(chunk.get("user_prompt", ""), language=None)
                st.markdown("**System Prompt：**")
                st.code(chunk.get("system_prompt", ""), language=None)
                st.markdown("**Chunk Text：**")
                st.code(chunk.get("chunk_text", ""), language=None)

            with tab_response:
                raw = chunk.get("raw_response")
                if raw:
                    st.code(raw, language="json")
                else:
                    st.info("無原始回應（可能失敗）")

            with tab_parsed:
                parsed = chunk.get("parsed_result")
                if parsed:
                    st.json(parsed)
                else:
                    st.info("無解析結果")

            with tab_preview:
                parsed = chunk.get("parsed_result")
                edits = parsed.get("edits", []) if parsed else []
                if not edits:
                    st.info("此區塊無修改")
                else:
                    ignore_boundary_ws = st.checkbox(
                        "忽略段落前後的空白",
                        value=True,
                        key=f"ignore_boundary_whitespace_debug_{cidx}",
                    )
                    st.caption(f"共 {len(edits)} 項修改")
                    for edit_i, edit in enumerate(edits):
                        pidx = edit.get("paragraph_index")
                        corrected = edit.get("corrected_text", "")
                        reason = edit.get("reason", "")
                        original = para_lookup.get(pidx, "")

                        st.markdown(f"**段落 {pidx}**")
                        if original:
                            preview_original = original
                            preview_corrected = corrected
                            if ignore_boundary_ws:
                                preview_original = strip_boundary_whitespace_for_preview(preview_original)
                                preview_corrected = strip_boundary_whitespace_for_preview(preview_corrected)

                            tracked_html = render_tracked_changes_html(preview_original, preview_corrected)
                            st.markdown(
                                f'<div style="padding:0.75em 1em;border:1px solid #ddd;'
                                f'border-radius:6px;line-height:1.8;font-size:1rem;'
                                f'white-space:pre-wrap;">{tracked_html}</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.code(corrected, language=None)
                            st.caption("⚠️ 找不到原文，僅顯示修正後文字")
                        if reason:
                            st.caption(f"💡 {reason}")
                        if edit_i < len(edits) - 1:
                            st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #eee;'>",
                                        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
