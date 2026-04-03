import streamlit as st


def is_prod_environment() -> bool:
    """Resolve environment mode from secrets.

    Defaults to production mode when IS_PROD is missing.
    """
    raw_value = st.secrets.get("IS_PROD", True)

    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)):
        return bool(raw_value)

    normalized = str(raw_value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "on"}


def get_table_name(base_name: str) -> str:
    """Return env-scoped Supabase table name.

    - prod: `<base_name>`
    - dev: `dev_<base_name>`
    """
    if is_prod_environment():
        return base_name
    return f"dev_{base_name}"
