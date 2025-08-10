from __future__ import annotations

from typing import List
from pathlib import Path

try:
    import sentencepiece as spm  # type: ignore
except ImportError:
    spm = None


class SentencePieceTokenizer:
    """
    SentencePiece 토크나이저를 감싸고, 특수 토큰을 처리하는 래퍼 클래스.
    - <pad>: 0
    - <bos>: 1
    - <eos>: 2
    - <unk>: 3
    SentencePiece가 생성하는 ID와 충돌하지 않도록 ID를 시프트하여 관리합니다.
    """
    def __init__(self, model_path: str):
        if spm is None:
            raise RuntimeError("sentencepiece가 설치되지 않았습니다. 'pip install sentencepiece'로 설치해주세요.")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.num_special_tokens = 4

    def encode(self, text: str, add_special: bool = False) -> List[int]:
        """텍스트를 ID 시퀀스로 인코딩합니다. SPM ID는 특수 토큰 수만큼 시프트됩니다."""
        # spm에서 나오는 id가 0 (unk)부터 시작하므로, 예약된 특수 토큰 id와 겹치지 않게 offset을 더해줌
        ids = [i + self.num_special_tokens for i in self.sp.EncodeAsIds(text)]
        if add_special:
            return [1] + ids + [2]  # <bos>=1, <eos>=2
        return ids

    def decode(self, ids: List[int]) -> str:
        """ID 시퀀스를 텍스트로 디코딩합니다. 특수 토큰은 제외됩니다."""
        # 특수 토큰(0,1,2,3)을 제외하고, spm이 아는 id로 다시 시프트하여 디코딩
        core_ids = [i - self.num_special_tokens for i in ids if i >= self.num_special_tokens]
        return self.sp.DecodeIds(core_ids)

    @property
    def vocab_size(self) -> int:
        """SPM 사전 크기에 특수 토큰 수를 더한 전체 사전 크기를 반환합니다."""
        return int(self.sp.GetPieceSize()) + self.num_special_tokens

    @property
    def pad_id(self) -> int:
        """패딩 토큰의 ID를 반환합니다."""
        return 0

    @property
    def bos_id(self) -> int:
        """문장 시작 토큰의 ID를 반환합니다."""
        return 1

    @property
    def eos_id(self) -> int:
        """문장 종료 토큰의 ID를 반환합니다."""
        return 2

    def save(self, path: str | Path) -> None:
        """SentencePiece 모델은 외부 파일이므로, 이 메서드는 경로 정보만 저장합니다."""
        Path(path).write_text("spm-external", encoding="utf-8")
