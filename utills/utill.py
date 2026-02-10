from datetime import datetime


#=========================
# JSON / 시간 / 해시 유틸
#=========================


# iso 표준 시간 맞추기
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")