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
    synonyms = load_yaml("synonyms.yaml")["synonyms"]
    error_patterns = load_yaml("error_patterns.yaml")["error_patterns"]
    intents = load_yaml("intents.yaml")["intents"]
    normalization = load_yaml("normalization_rules.yaml")["normalization_rules"]

    return {
        "synonyms" : synonyms,
        "error_patterns" : error_patterns,
        "intents" : intents,
        "normalization" : normalization
    }

#딕셔너리를 가져와 사용자의 query(질문)을 변환 
def normalize_query(text: str, bundle: dict) -> str:
    #bundle = get_dictionary_bundle() # 모든 규칙을 불러옴
    t = text.strip().lower() # 기본 정규화

    preserve_tokens = bundle["normalization"].get("preserve_tokens", [])

    #preserve_tokens이 있으면 치환되지 않음
    for token in preserve_tokens:
        if token.lower() in t:
            return t

    for k,v in bundle["normalization"].get("replace", {}).items():  #replace키가 있으면 가져오고 아니면 {} 빈값 가져오기
        t = t.replace(k.lower(), v.lower()) #k는 원래 표현, v는 바꿀 결과

    # 동의어 매핑 -> 표준키 확장 ( ex. 동의어가 질문에 계속 추가되는 형태 ) 
    expansions = [] # 확장하고자 하는 키워드 + 
    for canonical, variants in bundle["synonyms"].items():
        for v in variants:
            if v.lower() in t:
                expansions.append(canonical)
                break

    # 일반 딕셔너리 일때는 k,v 도메인적 의미가 있을 경우 canonical 같은 의미가 있는 변수를 사용
    # 사용자 질문안에 에러가 메시지가 있으면 에러를 대표하는 => 표준 검색어를 자동으로 추가    
    extracted = []
    for name, cfg in bundle["error_patterns"].items(): #name은 지금은 안쓰지만 의미용 변수 or 운영시 로그 확인용 작성해둠
        for p in cfg.get("patterns", []):
            if p.lower() in t:
                canon = cfg.get("canonical_query", [])
                extracted.extend(canon if canon else [p]) #extend는 리스트로 받아서 배열에 담아 넣어주기
                break

    # 최종 검색어 생성
    final_query = "".join([t] + extracted + expansions) # 문자열 조립 함수

    return final_query

    