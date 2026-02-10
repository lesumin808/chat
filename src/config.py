from dataclasses import dataclass
from pathlib import Path

"""
1. reindex 시작
2. lock 존재? → 있으면 종료
3. lock 생성
4. state.status = "running"
5. manifest 비교
6. 변경 없으면 state = "skipped"
7. 변경 있으면 reindex
8. 성공 → state = "success"
9. 실패 → state = "failed"
10. lock 삭제
"""

@dataclass(frozen=True)
class Config:
    doc_dir: Path = Path("../docs")
    manifest_path: Path = Path("../docs/manifest.json")

    # 벡터 db 경로
    index_root: Path = Path("../index")
    index_current_dir: Path = Path("../index/chroma_hugginface")
    index_tmp_dir: Path = Path("../index/chroma_tmp")

    #lock lock lock 
    state_path: Path = Path("")
    lock_path: Path = Path("")
    #reindex.lock, state.json ? lock(잠궈둠)

    verify_file: bool = False

    # swap 시 backup을 남겨둘지? (디버깅, 롤백용..) ?? True면 _backup을 유지함
    keep_backup: bool = False