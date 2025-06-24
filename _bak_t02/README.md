
# cb_t02 Project 안내

이 프로젝트는 Windows 10 + RTX 4060 기반에서 구동되는 한국어 챗봇 학습 시스템입니다.  
PyTorch 2.3.0 (CUDA 11.8 GPU 전용), torchtext 0.18.0 환경을 요구합니다.

---

## 📦 실행 환경

- OS: Windows 10
- Python 3.10
- torch==2.3.0+cu118
- torchtext==0.18.0
- sentence-transformers 등은 requirements.txt 참고

### GPU 설치

```bash
pip install torch==2.3.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

## ⚠️ 주의사항

- 개발 및 테스트는 **사용자 로컬 기준**에서 수행됩니다.
- 코드 작성, 학습, 경로 구조 등 모든 규정은 **AGENTS.md에 정의**되어 있으며, 이를 준수해야 합니다.
- 본 README.md는 문서용이며, 자동화 툴(Codex 등)은 AGENTS.md만 기준으로 동작합니다.
- **README.md는 사용자 외 절대 수정하지 말 것**
- **UI.html 파일은 디자인·스타일 측면에서 함부로 수정 금지 (예외 시 지시 필요)**

---

## 📑 지침 문서

→ [AGENTS.md를 반드시 참조하여 작업할 것]

### 사전 토큰화
```bash
python tools/pretokenize.py data/raw/ --out cache/
```

### 전체 데이터 학습
```bash
python run.py --config configs/current.json --mode full
```

### CLI 학습 모드 예시
```bash
python run.py --mode train --config configs/current.json
```
