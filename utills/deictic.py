from pathlib import Path
import yaml
import re #Regular Expression (정규표현식) 라이브러리
from functools import lru_cache


BASE_DIR = Path(__file__).resolve().parents[1] #상위 디렉토리
#Path(__file__).resolve()        # /home/leesumin/dev/app/main.py
#.parents[0]                    # /home/leesumin/dev/app
#.parents[1]                    # /home/leesumin/dev
DICT_DIR = BASE_DIR / "dictionary" #yaml파일 존재 경로

@lru_cache(maxsize=1)
def load_deictic_patterns():
    """history_aware_retriever 를 활용하기 위해 yaml파일을 불러온다."""
    path = DICT_DIR / "deictic_patterns.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["patterns"]


def get_deictic_regex():
    """지시어를 정규화하고 해당 지시어를 합쳐서 반환한다."""
    patterns = load_deictic_patterns()
    escaped = [re.escape(p) for p in patterns]

    return re.compile("|".join(escaped)) # 반환값은 re.Pattern이라는 정규식 객체
   # key, value를 빠르게 분석할때는 re 정규식객체로 변환한 것이 옳다. (검색 엔진 객체)



