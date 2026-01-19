from functools import lru_cache #함수 결과를 메모리에 캐싱하는 데코레이터 -> 같은 함수가 호출되면 파일을 다시 읽지 않고 캐시 반환 (yaml 한번만 읽기)
from pathlib import Path #파일 경로를 OS 독립적으로 다루는 표준 라이브러리(windows / linux 경로 차이 자동 처리)
import yaml #pyyaml 라이브러리로 파일 파싱 전용


BASE_DIR = Path(__file__).resolve().parents[1] #상위 디렉토리
#Path(__file__).resolve()        # /home/leesumin/dev/app/main.py
#.parents[0]                    # /home/leesumin/dev/app
#.parents[1]                    # /home/leesumin/dev
DICT_DIR = BASE_DIR / "dictionary" #yaml파일 존재 경로


@lru_cache(maxsize=1) # 캐시 데코레이터 : 함수의 실행 결과를 메모리에 저장. 같은 인자로 호출되면 파일을 다시 읽지 않고 바로 반환
def load_yaml(name: str) -> dict: #str 스트링 dict 딕셔너리 => 개발자간의 약속 설명
    path = DICT_DIR / name # pathlib.PAth 문법
    with open(path,"r", encoding="utf-8") as f: #f 는 파일
        #with 는 자원을 안전하게 열고, 자동으로 정리해주는 문법
        return yaml.safe_load(f)
    

#yaml 파일 load 한번만 캐싱 후 적재한다. 반환씨 dict 파일 key value로 반환
@lru_cache(maxsize=1)
def get_dictionary_bundle():
    synonyms = load_yaml("synonyms_v1.yaml")["synonyms"]
    error_patterns = load_yaml("error_patterns_v1.yaml")["error_patterns"]
    intents = load_yaml("intents_v1.yaml")["intents"]
    normalization = load_yaml("normalization_rules_v1.yaml")["normalization_rules"]

    return {
        "synonyms" : synonyms,
        "error_patterns" : error_patterns,
        "intents" : intents,
        "normalization" : normalization
    }

#딕셔너리를 가져와 사용자의 query(질문)을 변환 
def normalize_query(text: str) -> str:
    bundle = get_dictionary_bundle() # 모든 규칙을 불러옴
    t = text.strip().lower() # 기본 정규화

    preserve_tokens = bundle["normalization"].get("preserve_tokens", [])

    #preserve_tokens이 있으면 치환되지 않음
    for token in preserve_tokens:
        if token.lower() in t:
            return t

    for k,v in bundle["normalization"].get("replace", {}).item  #replace키가 있으면 가져오고 아니면 {} 빈값 가져오기
        t = t.replace(k.lower(), v.lower()) #k는 원래 표현, v는 바꿀 결과
