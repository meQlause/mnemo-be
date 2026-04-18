from datetime import datetime, timedelta, timezone

def get_jakarta_now() -> datetime:
    """Returns current time in Jakarta (WIB - UTC+7)"""
    return datetime.now(timezone(timedelta(hours=7)))

def get_jakarta_today_str() -> str:
    """Returns current date in Jakarta as YYYY-MM-DD"""
    return get_jakarta_now().strftime("%Y-%m-%d")
