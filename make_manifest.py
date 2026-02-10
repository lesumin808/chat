#%%
import hashlib
import json
from pathlib import Path
from datetime import datetime

# ----------------------------
# manifest.json 생성 스크립트
# ----------------------------

#file경로
DOC_DIR = Path("./docs") #doc 파일 경로
OUT = DOC_DIR / "manifest.json" #json 파일 경로

# ----------------------------
# 파일을 읽어서 hash값을 계산
# ----------------------------
def file_to_hash(p:Path) -> str:
    hash_obj = hashlib.sha256()

    with p.open("rb") as f: #"rb" : read + binary (바이너리 모드)
        while True:
            chunk = f.read(1024 * 1024) #1mb

            if not chunk:
                break

            hash_obj.update(chunk)

    return hash_obj.hexdigest()
    
#%%    
files = [] #file을 담는 배열
#%%  
for p in sorted(DOC_DIR.rglob("*.md")): #Generator[YieldType, SendType, ReturnType]
    print(type(p))
    files.append({
        "path": str(p.relative_to(DOC_DIR)),
        "sha256": file_to_hash(p)
    })

embedding_model = "intfloat/multilingual-e5-large-instruct"
chunking_option = []

chunking_option.append({
    "chunk_size": 900,
    "chunk_overlap" : 200
})

manifest = {
    "docset_version" : datetime.now().strftime("%Y.%m.%d"), #사람이 보기 쉬운 시간
    "generated_at" : datetime.now().isoformat(), # 정확한 생성 시점 (기계파싱)
    "embedding" : embedding_model,
    "chunking" : chunking_option,
    "files" : files
}

OUT.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("manifest.json created.")
