**작업 지침 요약**
- 본 프로젝트는 GPU(CUDA) 환경을 전제로 하며, 주 로직은 항상 GPU 기반 코드를 유지합니다.
- CPU 모드는 tests/ 디렉터리에서의 내부 테스트 또는 환경 제약으로 GPU 사용이 불가한 경우에만 허용됩니다.
- 환경 차이로 인해 자율 수정이 필요한 경우, 변경 사유와 영향 범위를 결과 보고와 Changelog에 반드시 기록해야 합니다.
- SentencePiece(SPM) 산출물은 항상 `models/` 디렉터리에 저장됩니다.
- 학습 시작 전 SPM을 자동 생성·정합성 검사하고 모델+vocab 결합 SHA256을 기록합니다.
- 서빙 시 체크포인트 메타와 SPM의 `piece_size`와 SHA256이 다르면 즉시 중단합니다.

**중요: 테스트 환경 안내**  
- 본 프로젝트의 **학습·서빙 로직은 GPU(CUDA) 전제**로 설계되었습니다.  
- 다만 코덱스 컨테이너 등 **GPU 미탑재 환경**에서는 개발·검증을 위해 **tests/** 디렉터리에서 **CPU 모드**로 테스트를 허용합니다.  
- CPU 테스트 방법: `ALLOW_CPU_TRAINING=1` 환경 변수 설정 후 tests/ 스크립트를 실행하십시오.  
- CPU 테스트는 **내부 검증 전용**이며, 본 프로젝트 메인 코드에는 **GPU 로직만 반영**합니다.  
왜 필요한가? → GPU 없는 환경에서 내부 테스트가 중단되는 문제를 방지하고, 로직 안전성 검증을 지속하기 위함입니다.

---
> **프로젝트 지향점(중요)**  
> 본 저장소는 ‘일반 챗봇’이 아니라 **한국어 지향 LLM 학습·서빙 스택**입니다.  
> 로직 판단은 항상 **LLM 기준(토크나이저/컨텍스트 길이/마스크/체크포인트/평가 지표)**으로 수행합니다.  
> 단순 규칙기반/faq 챗봇 가정은 금지합니다.
---

# cb_t03 Project 안내

이 프로젝트는 Windows 10 + RTX 4060 기반에서 구동되는 한국어 챗봇 학습 시스템입니다.  
PyTorch 2.3.0 (CUDA 11.8 GPU 전용), torchtext 0.18.0 환경을 요구합니다.
본 프로젝트는 외부 사전학습 모델이 아닌 **직접 구현한 트랜스포머 신경망**을 학습하고 서빙하는 것을 목표로 합니다.
HFModel과 DummyModel 모듈은 테스트용임을 명확히 구분합니다.

---

## 준비
python train_spm.py --input "datas/pretrain/**/*.txt"

## 📦 실행 환경

- OS: Windows 10
- Python 3.10
- torch==2.3.0+cu118
- torchtext==0.18.0
- sentence-transformers 등은 requirements.txt 참고

## ⚠️ 주의사항

- 개발 및 테스트는 **사용자 로컬 기준**에서 수행됩니다.
- 코드 작성, 학습, 경로 구조 등 모든 규정은 **AGENTS.md에 정의**되어 있으며, 이를 준수해야 합니다.
- 본 README.md는 문서용이며, 자동화 툴(Codex 등)은 AGENTS.md만 기준으로 동작합니다.
- **README.md는 사용자 외 절대 수정하지 말 것**
- **UI.html 파일은 디자인·스타일 측면에서 함부로 수정 금지 (예외 시 지시 필요)**
- 학습 모드는 CUDA 환경에서만 동작하며 GPU가 없으면 즉시 오류가 발생합니다.
- STOP 센티넬은 에폭 종료 직후에만 감지되어 학습이 중단됩니다.
- 모든 체크포인트 저장 경로는 `models/`로 통일되어 STOP 파일 경로와 일치합니다.
- 토크나이저는 학습 중 교체할 수 없으며, 교체 시 프리플라이트 검사에서 즉시 중단됩니다.

## 설정 적용 원칙

- 모든 학습·추론 설정은 `trainconfig.ini` 한 곳에서 관리합니다.
- `lock_ui = yes`일 때 UI에서 입력한 값은 무시됩니다.
- 설정 병합 순서는 `DEFAULT` → `[train]` → `[pretrain|finetune]`입니다.
- `[pretrain]`과 `[finetune]`은 `[train]`과 다른 값이 있을 때만 키를 작성합니다.
- `min_lr`는 항상 `0.00001` 이상으로 클램프되며 `resume` 기본값은 `no`입니다.
- 학습률 값은 소수 형태로 표기하며(예: 0.0002), 코드 기본값 2e-4와 동일합니다.

[신규 LLM 프로젝트 개발 지시서 - 2025.06.24 기준]

1. 기존 프로젝트는 더 이상 유지/복구/수정하지 않는다.  
   - 기존 전체 코드는 "_bak_t02" 폴더에 백업되어 있으니,  
     해당 폴더는 오직 백업·참고용으로만 사용한다.  
   - 자동화, 테스트, 배포 등에서는 "_bak_t02" 폴더를 완전히 제외한다.

2. 신규 프로젝트는 완전히 새 환경에서 시작한다.  
   - 파일/폴더명 및 경로는 반드시 영문, 숫자, 언더바(ASCII)만 사용한다.
   - 모든 코드와 데이터는 Instruction(지시문) 방식으로 전환하여 설계/구현한다.  
     (기존 QA 구조는 사용하지 않음)

3. 재사용 원칙:  
   - 기존 프로젝트에서 **ui.html, 디버거, 폴더 구조, 그 외 필요 핵심 코드**는  
     신 프로젝트에 맞게 선택적으로 이관/수정/재사용 가능하다.
   - 단, 모든 코드/데이터/구조는 Instruction 방식에 맞게 반드시 수정/리팩터링해서 사용할 것.

4. 신규 개발 시 AGENTS.md(어젠트 명세) 및 Changelog.md(작업 이력)  
   - **항상 최신화하며,**  
   - 모든 변경/추가/재사용 내용은 즉시 기록한다.

5. 프로젝트의 목표와 핵심 차별점  
   - 이전 프로젝트는 QA 기반, **신규 프로젝트는 Instruction 기반** LLM 개발이다.  
   - 그 외 폴더구조, 일부 UI, 디버깅 등 실무적으로 유용한 부분만 “선별 재사용”하여 효율성/일관성 확보

6. 기타
   - 기존 프로젝트가 심각하게 손상되어 있으므로, 신규 프로젝트 환경에서는
     “최소 단위 복사, 불필요한 파일 미포함, 클린코드/ASCII 경로/명확한 구조” 원칙을 절대 준수한다.

## 📂 데이터셋 구조 변경

인스트럭션 기반 데이터셋만 지원한다. 각 항목은 `instruction`, `input`, `output` 필드로 구성되며,
학습 시에는 `instruction` 과 `input`을 공백으로 이어 붙인 문자열을 모델 입력으로 사용한다.
`output` 값만 정답으로 처리한다.

---

## 📑 지침 문서

→ [AGENTS.md를 반드시 참조하여 작업할 것]

### GPU 스모크 실행(10분)
1. CUDA 확인
   ```bash
   python - <<'PY'
   import torch; print('cuda_available=', torch.cuda.is_available()); 
   print('device=', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
   PY
   ```
2. 실행
   - `python run.py` 실행 후 UI에서 학습 1~2 스텝만 수행.
   - 로그에서 `[GPU] cuda_available=` / `[ENV] DISABLE_SDP_KERNEL=` / `[GPU] torch=` 문구를 확인.
3. ETA(대략) 계산 팁
   - 실측 tokens/sec = (스텝당 토큰 수 × 스텝/초)
   - 총 ETA ≈ (총 토큰 수) ÷ (tokens/sec × GPU 수)
### 운영 체크리스트(요약)
- 실행 전: CUDA 가용성, 디바이스명 확인(`[GPU] cuda_available=` 로그).
- ENV: `DISABLE_*` 값은 **'1'**만 비활성(기타 값은 무시) — 경고 로그 확인.
- 학습 시작/종료: `[TRAIN-START]` / `[TRAIN-END]` 로그와 체크포인트 경로 확인.
- 구성: `[CFG] epochs/batch/lr/...` 로그로 설정 확인.
- 장애 시: 콘솔 로그 마지막 200줄과 스택트레이스 원문만 수집·보고.

### 재개(resume) 절차
- 학습 중단 후 재개하려면 `trainconfig.ini`의 대상 섹션(`[pretrain]` 또는 `[finetune]`)에서 `resume = yes`로 지정하고, 최근 체크포인트가 `models/`에 존재해야 합니다.
- `python run.py` 실행 후 UI에서 재개 모드를 선택하면 됩니다.
- 로그에 `[TRAIN-START]` / `[CFG]` / `[TRAIN-END]`가 순서대로 출력되면 정상 재개입니다.
- 실패 시 콘솔 로그 마지막 200줄과 스택트레이스 원문을 보고하십시오.

### 서빙 스모크 절차
- `python run.py` 실행 → UI에서 질문 1회 입력.
- 로그에서 `[SERVE] model_loaded` / `[GEN] max_new_tokens=` 문구가 출력되는지 확인.
- 응답이 반환되면 정상. 실패 시 콘솔 로그 마지막 200줄과 스택트레이스 원문을 수집·보고.
- (선택) `trainconfig.ini`의 `resume` 설정과 관계없이 서빙은 최신 체크포인트를 자동 로드합니다.


### MVP 릴리스 체크리스트
- 학습 스모크(50~100 step): 로그에 `[GPU]`/`[ENV]`/`[CFG]`/`[DATA]`/`[TRAIN-START]`→`[TRAIN-END]` 순서로 출력됨.
- 재개(20 step): `trainconfig.ini`에서 `resume = yes` 유지, 체크포인트 존재 확인 후 재개 성공.
- 서빙 1회: `[SERVE] model_loaded`와 `[GEN] max_new_tokens=` 로그가 출력되고 응답이 반환됨.
- 실패 시 보고: 콘솔 로그 **마지막 200줄** + **스택트레이스 원문** + GPU/torch 버전.
- ENV 규칙: `DISABLE_*` 값은 '1'만 비활성(기타 값은 무시, 경고 로그 발생).

환경 변수 `LOG_EVERY_STEPS`로 스텝 로그 간격을 조절합니다(기본 10). 예: `LOG_EVERY_STEPS=50`

### Runtime/ENV
- `DISABLE_SDP_KERNEL=1` : `torch.backends.cuda.sdp_kernel` 호출을 건너뛰어 구형 GPU에서 오류를 방지합니다. (기본: 호출)
  - 예) `DISABLE_SDP_KERNEL=1 python run.py`
- `DISABLE_CUDNN_BENCHMARK=1` : `torch.backends.cudnn.benchmark=False`로 설정하여 입력 크기 변동 시 불안정을 줄입니다. (기본: True)
  - 예) `DISABLE_CUDNN_BENCHMARK=1 python run.py`
