#=======================
#  manifest.json을 비교하여 embedding여부를 확인한다.
#=======================

from typing import Dict
from pathlib import Path
import utills.utill
from config import Config


# manifest를 읽어서 가져오기
def fingerprint_payload(manifest: Dict[str, any]) -> Dict[str, any]:
    """
    변경 감지에서 의미 있는 필드만 추려서 payload 생성
    - files 순서 영향을 없애기 위해 path 기준 정렬
    """
    files = manifest.get("files", []) or [] # or [] 파일이 null일때 발생
    files_sorted = sorted(files, key=lambda x: (x.get("path") or "")) # get fn은 없으면 None을 반환 / lambda(익명함수)
    
    return {
        "docset_version" : manifest.get("docset_version", []) or [],
        "generated_at" : manifest.get("generated_at", []) or [],
        "embedding" : manifest.get("embedding", []) or [],
        "files" : [{"path" : f.get("path"), "sha256" : f.get("sha2556")} for f in files_sorted ]
    }


#기존 reindex가 있는지 확인하고 없으면 pass하도록 하고 있으면 기존 maifest와 비교

# 01_state.json?? (기존정보가 명시되어있는 파일) 가 있는 지 확인
def load_state(path: Path):
    if not path.is_file(path):
        return {}
    return utills.utill.load_json(path)

#manifest의 값과 기존의 state.json값을 비교, 없으면 생성 있으면 비교 후 재생성
def run_reindex(cfg:Config):
    #임베딩 루트
    cfg.index_root.mkdir(parents=True, exist_ok=True)



