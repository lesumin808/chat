import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict


#=========================
# JSON / 시간 / 해시 유틸
#=========================

#mainfest.json(사람이 만든 선언) -> reindex.py(판단 + 실행 + 기록) -> state.json(시스템이 만든 기록)


# iso 표준 시간 맞추기
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

#데이터(byte)를 str(16진 문자열로 변환)
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest() #이진수(binary) 데이터를 16진수(Hexadecimal) 문자열로 변환


#파일을 16진수 hash로 변환
def sha256_file(path:Path, chunk_size:int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f: # file을 read바이너리로 읽고
            for chunk in iter(lambda: f.read(chunk_size), b""): #chunk_size 1MB
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        print(f"{path} 경로에 파일이 없습니다.")
    except Exception as e:
        print(f"{e} 발생")
    return None

#json 생성
def load_json(path:Path) -> Dict[str, any]:
    try:
        return json.load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"{path} 경로에 파일이 없습니다.")
    except json.JSONDecodeError:
        print(f"{path} 파일의 json 형식이 올바르지  않습니다.")
    except Exception as e:
        print(f"{e} 발생")
    return None


#json 저장
def save_json(path: Path, obj:Dict[str, any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

