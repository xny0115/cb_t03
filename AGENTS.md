
# AGENTS.md

Codex는 아래 지침을 무조건 준수하여 작업해야 합니다.  
이 지침은 코드 작성, 테스트, 저장소 운영 전반에 적용되며, **모든 내부 판단보다 우선**합니다.

---
## 🚀 프로젝트 목표
- 외부 사전학습 모델 연동이 아닌 **직접 구현한 트랜스포머 신경망**을 개발·학습·서빙하는 것이 최우선 목표다.
- `HFModel`, `DummyModel` 모듈은 흐름 검증용 임시 코드일 뿐 실제 서비스 모델이 아니다.
- 이 원칙은 README와 Changelog 등 모든 문서에 반복 기록한다.


## ✅ 작업 환경 및 제한

- Codex는 자신의 테스트 환경(torch 미설치 등)을 **절대 기준으로 삼지 말 것**.
- 모든 작업은 **사용자 로컬 기준(Windows 10 + RTX 4060, CUDA 11.8)** 으로 간주하고 수행할 것.
- torch는 반드시 GPU 전용 버전 2.3.0만 사용하며, torchtext는 0.18.0만 허용.
- CPU 버전은 **테스트에서도 절대 사용 금지**. 단, 내부 컨테이너 테스트용은 예외로 허용.

---

## ✅ 코드 작성 및 구조 규칙

1. 코드 파일은 250줄 이하로 유지. 초과 시 책임 단위로 파일 분할 (※ ui.html 및 데이터셋 예외)
2. 함수는 150줄 이하로 유지. 초과 시 리팩토링 또는 분할 필수
3. 모든 학습/임베딩/모델 파일은 `/models/`, 데이터셋은 `/datas/` 경로에서만 불러올 것
4. 폴더 및 파일 구조는 역할 기반으로 모듈화하고, **cross-import, 순환참조 금지**
5. 실행/테스트/설정은 각기 분리. 설정·상수는 config 파일에서만 관리
6. 테스트 코드는 메인 코드와 분리할 것
7. 예외처리, 로깅, 디버깅, 테스트 코드 항상 포함할 것
8. PEP8 준수, 타입 힌트, docstring 작성 필수
9. 코드, 변수, 함수명은 영어만 사용 / 주석은 한글만 사용
10. 작업 전 구조·경로·지침 검토 후 위반 시 **전체 코드 재출력**
11. 본 프로젝트는 한국어 llm을 개발하는게 목적이므로 한국어 기준의 문장을 생성해야함. (영어 또는 기타 외국어 아님)
12. 수정 사항 적용후 Changelog.md 파일에 아주 상세한 문제점 부터 패치 내역까지의 총과정을 기록. (한국어로만 기록.)
13. 지시내용은 미처리 및 누락없이 충실히 이행하시오.
14. 본프로젝트는 이전 프로젝트의 리펙터링 과정임. 이전 프로젝트는 _bak_t02에 있음.
   - _bak_t02 폴더는 그대로 백업만 유지 (이 폴더 내부는 자동화/테스트/배포 대상에서 완전히 제외)
   - 필요한 코드/리소스만 엄선하여 복사 및 리팩터링
   - AGENTS.md 최신 기준대로 설계/작성/작업
   - 폴더 및 파일명, 경로, 내부 구조 모두 ASCII 규칙 유지
15. 중요
   - UI(특히 ui.html)는 “디자인 및 레이아웃 변경 금지”가 명확히 지시되었다.
   - 입력 필드, 컴포넌트, 배치, CSS 등 모든 디자인 요소는 기존 구조를 100% 유지해야 한다.
   - 기능 추가, 필드 삽입, 로직 보강이 필요할 경우 반드시 “디자인을 손대지 않는 범위 내”에서만 코드/스크립트 수정이 허용된다.
   - “디자인 그대로 유지” 원칙 위반 시, 작업 중단 및 전체 롤백(혹은 코드 재출력)한다.
   -  주관적 해석, “작은 변경이니 괜찮겠지” 등 변명 일절 불가.

---

## ✅ 데이터셋(Instruction) JSON 구조 기준

예시 1
    {
    "instruction": "텍스트를 영어로 번역하세요.",
    "input": "안녕하세요. 만나서 반갑습니다.",
    "output": "Hello. Nice to meet you."
    }

예시 2

    {
    "instruction": "로렘 입숨(Lorem Ipsum)이 무엇인지 설명하세요.",
    "input": "",
    "output": "로렘 입숨은 출판 및 그래픽 디자인 분야에서 사용되는 더미 텍스트입니다..."
    }

### 3단계 학습 데이터 구조

```
datas/
    pretrain/              # 사전학습용 txt
    finetune/              # 1차 파인튜닝용 jsonl
    additional_finetune/   # 추가 파인튜닝용 jsonl
```

각 폴더에는 `sample_pretrain.txt`, `sample_finetune.jsonl`, `sample_additional_finetune.jsonl` 예시 파일을 포함한다.
버튼 클릭 시 `pretrain`, `finetune`, `additional_finetune` 값을 백엔드로 전달해 해당 데이터만 로드한다.
모든 스테이지의 데이터는 폴더 내 여러 파일을 메모리에서만 통합하며
원본 파일은 절대 수정하거나 삭제하지 않는다.


## ✅ 학습 로직 강제 기준

- torch는 `import torch`로 직접 로딩하고, 실패 시 로직을 우회하지 말 것
- JSON 기반 우회 저장 금지. 학습 모델은 반드시 `.pth`로 `torch.save()` 저장
- 모델 저장 후 `.exists()` 및 `파일 크기 ≥ 1MB` 검증 필수
- 학습 로그는 `Training complete`, `Model saved to ...` 등 명시적으로 출력할 것
- 삭제 버튼은 실제 `.pth` 파일 삭제 동작과 연결되어야 함
- 삭제 시 `models` 폴더 내 존재하는 모든 모델 파일을 일괄 제거한다.

---

## 📌 기타

- 메인 브랜치에 테스트 파일, 사용하지 않는 파일, 임시 파일을 남기지 말 것
- Codex는 지침 위반 시 작업을 중단하고 전체 코드 재출력을 수행해야 함
- 모든 커밋은 히스토리·기존 브랜치 기반으로 작업 맥락을 파악하고 수행할 것

