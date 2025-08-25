# 변경 이력
## v1.78: STOP 기반 안전 중지 및 INI 1회 적용
- UI 버튼을 "모델 삭제"에서 "학습 중지"로 고정하고 성공/실패 메시지를 "중지 요청 완료"/"중지 요청 실패"로 통일했습니다.
- ChatbotService.delete_model이 모델 파일 삭제 대신 STOP 센티넬 파일을 생성해 에폭 종료 시 안전하게 학습을 중지합니다.
- 학습 루프가 STOP 파일을 감지하면 체크포인트 이후 즉시 중단하며 최종 모델 저장을 생략합니다.
- start_training이 INI를 학습 직전 1회만 병합하고 `[CFG-TRAIN]` 로그와 `min_lr` 클램프(최소 1e-5)를 명시합니다.
- 호환성 변화 없음.
## v1.77: 학습 중지 센티넬 및 INI 클램프 적용
- UI 버튼을 "모델 삭제"에서 "학습 중지"로 변경하고 성공 메시지를 "중지 요청 완료"로 수정했습니다.
- ChatbotService.delete_model이 모델 파일 삭제 대신 STOP 센티넬 파일을 생성하여 에폭 종료 후 안전하게 중지하도록 변경했습니다.
- 학습 루프가 STOP 파일을 감지하면 에폭 완료 후 즉시 중단하며 최종 모델 저장을 생략합니다.
- start_training에 `[CFG-TRAIN]` 로그를 강화하고 `min_lr`를 1e-5 이상으로 클램프했습니다.
- 호환성 변화 없음.
## v1.76: SPM 결합 해시 고정·테스트/문서/릴리스 정리
- 모델+vocab 결합 SHA256을 고정값으로 확정하고 가드를 최종 확정했습니다.
- 생성→재사용→불일치 가드까지 전수 테스트를 통과했습니다.
- README와 Changelog를 정리하고 `v1.76` 태그를 생성했습니다.
## v1.75: SPM 결합 해시 및 가드 안정화
- SPM 모델과 vocab을 결합한 SHA256 해시를 프리플라이트와 서빙 가드에 도입했습니다.
- 체크포인트에 결합 해시를 기록하고 로그에는 앞 8자만 출력하도록 수정했습니다.
- SPM vocab 파일이 없을 경우 즉시 중단하도록 오류 처리를 강화했습니다.
## v1.74: SPM 경로 통합 및 프리플라이트 가드
- SentencePiece 모델 경로를 `tokenizer/`에서 `models/`로 일원화했습니다.
- 학습 시작 직전에 SPM 존재 확인, 필요 시 자동 생성 및 정합성 검사를 수행합니다.
- 서빙 시 체크포인트 메타와 SPM의 `piece_size`/`sha256`을 비교해 불일치 시 중단하도록 가드를 추가했습니다.
## v1.73: INI 인라인 주석 및 모드 오버레이
- `trainconfig.ini`에 모든 항목을 인라인 주석 형식으로 통일하고 `epochs` 키를 도입했습니다.
- `_read_ini`가 `inline_comment_prefixes`를 사용하고 `pretrain`/`finetune` 섹션과 `epochs`를 후방 호환으로 파싱합니다.
- `start_training`이 모드별 섹션을 오버레이하여 `[train]` 기본값 위에 선택적으로 덮어씁니다.
- `DEFAULT_CONFIG`에 `grad_clip`, `min_lr` 기본값을 추가하여 누락을 방지했습니다.
- README 상단에 SentencePiece 준비 명령어를 명시했습니다.
## v1.72: INI 주석 확장 및 구성 검증 강화
- `trainconfig.ini`에 모든 사용 키와 한국어 주석을 추가하고 기본값으로 초기화했습니다.
- `service.py`의 구성 로그 태그를 `[CFG-TRAIN]`, `[CFG-GEN]`으로 통일했습니다.
- `start_training` 진입 전 INI 덮어쓴 값에 대한 범위 검사를 추가했습니다.
- `training.simple`에는 "검증만 수행, 로직 변경 없음" 주석을 삽입했습니다.
## v1.71: INI 전용 추론 파라미터와 명시적 이어학습 분기
- service.py의 `_resolve_generate`가 요청 값을 무시하고 `trainconfig.ini`만 참조하도록 수정했습니다.
- `start_training`이 `--mode resume`일 때만 `resume`을 활성화하도록 강제했습니다.
- HFModel 추론 시 INI에서 읽은 생성 파라미터를 그대로 전달합니다.
- `trainconfig.ini`의 `min_lr` 기본값을 `1e-5`로 조정했습니다.
## v1.70: Prototype 모드 – INI 전용 제어 및 학습 안정화.
- service.py에서 `_resolve_generate`와 `_apply_train_ini`를 도입해 UI 수치를 무시하고 `trainconfig.ini` 값만으로 추론·학습 파라미터를 확정.
- `resume` 기본값을 False로 고정하고 [CFG] 로그를 강화, run.py에 `resume` 모드 CLI 분기 추가.
- `training.simple`에 학습률 하한(min_lr) 적용해 스케줄러가 0 이하로 내려가지 않도록 보호.
## v1.69: trainconfig.ini 기반 추론/학습 파라미터 오버라이드 추가.
- service.py에 INI 로더와 `_resolve_params`를 도입해 [CFG] 로그와 함께 생성 파라미터를 적용.
- start_training이 INI의 train 섹션 값을 config에 병합하여 UI 없이도 수치 조정 가능.
- 루트에 `trainconfig.ini` 샘플 파일을 추가.
## v1.68: 레거시 체크포인트 vocab 추론 로드 및 DummyModel 로깅 가드 추가.
- 구형 ckpt에 vocab 메타가 없을 때 가중치 행렬에서 단어 수를 추론하여 트랜스포머 로드.
- DummyModel 로드 시 parameters() 부재로 발생하던 로깅 예외를 _param_count 가드로 방지.
## v1.67: resume 시 에폭 보정(추가 학습 보장) 및 스텝 로그 간격화(LOG_EVERY_STEPS, 기본 10).
## v1.66: README에 'MVP 릴리스 체크리스트' 추가(코드 변경 없음).
## v1.65: 서빙 로드/생성 요약 로그 및 서빙 스모크 절차 문서화(기능 변경 없음).
## v1.64: service.py에 [DATA] 크기 로그 추가, README에 재개 절차 문서화(기능 변경 없음).
## v1.63: service.py에 [CFG] 요약 로그, README에 운영 체크리스트 추가(기능 변경 없음).
## v1.61: simple.py에 GPU 텔레메트리 로그 추가, README에 스모크·ETA 가이드 추가(기능 변경 없음).
## v1.59: ENV 오입력 경고 로그(2줄) 추가 — 기능/성능 변경 없음.
## v1.58: simple.py에 GPU/ENV 상태 로그(2줄) 추가 — 기능/성능 변경 없음.
## v1.57
- `training.simple`에 sdp_kernel·cudnn 환경 변수 가드를 추가했습니다. Codex 내부 테스트는 GPU 미지원으로 `ALLOW_CPU_TRAINING=1` 설정 후 `tests/`만 CPU 모드로 실행합니다.
## v1.56
- `training.simple`: `zero_grad(set_to_none=True)` 호출을 예외 처리 래핑하고, 패딩만 있는 배치·더미 옵티마이저를 안전하게 처리하도록 보강했습니다.
- `service.infer`: 트랜스포머 모델에만 토크나이저를 필수로 요구하도록 검증 조건을 보강했습니다.
- `service.auto_tune`: `get_dataset_info` 호출 시 추가 데이터 디렉터리를 인자로 전달하도록 수정했습니다.
## v1.55
- README와 AGENTS 문서 상단에 GPU 전제, CPU 모드 사용 조건, 변경 사유 기록 의무를 명시했습니다.
## v1.54
- GPU 전제 로직과 CPU 전용 테스트 환경을 README, 주요 테스트 스크립트, `src/training/simple.py` 주석에 명시했습니다.
## v1.53
- README 상단에 LLM 지향 프로젝트 배너를 추가하고 학습 루프 및 서비스 모듈에 LLM 기준 주석을 삽입했습니다.

## v1.52
- CUDA 필수 정책을 강화하여 GPU가 없을 경우 `ALLOW_CPU_TRAINING=1` 환경 변수를 설정하지 않으면 즉시 오류를 발생하도록 수정했습니다.
- 작은 텐서로 드라이런을 수행해 옵티마이저와 스케줄러 스텝이 정상 동작함을 확인했습니다.
## v1.51
- DataLoader 기본값이 하드코딩되어 설정이 무시되던 문제를 수정했습니다. 이제 `num_workers`, `pin_memory`, `drop_last`가 cfg 값에 따라 동작합니다.
- CUDA 미사용 환경에서 학습이 중단되던 로직을 경고 후 CPU 학습으로 전환하고, AMP가 CUDA에서만 활성되도록 조건을 보강했습니다.
- 서비스 레이어 역시 CPU 사용 시 경고만 출력하도록 수정했습니다.
- SentencePiece 모델 미존재 시 모호하던 예외 메시지에 생성 명령어 가이드를 추가했습니다.
## v1.50
- **이어학습 로직 수정**: 사전학습(`pretrain`) 모델의 이어학습 시, 데이터를 일반 텍스트가 아닌 명령어(Instruction) 형식으로 잘못 처리하여 `AttributeError`가 발생하던 버그를 수정했습니다. 이제 이어학습 시 모델의 종류에 따라 올바른 데이터 로더를 사용합니다.
- **디버그 로그 정리**: 사전학습 시 불필요하게 `collate_fn`의 실행 시간을 측정하는 디버그 로그가 과도하게 출력되던 문제를 해결했습니다.
## v1.49
- **학습 프로세스 오류 수정**:
  - `_train_epoch` 함수가 `loss`와 `duration`을 올바르게 반환하지 않아 발생하던 `TypeError`를 수정하여 학습이 정상적으로 진행되도록 했습니다.
  - (이전 수정에 포함) `train_spm.py` 스크립트에서 파일 경로에 Non-ASCII 문자가 포함될 때 발생하는 인코딩 오류 및 공백이 포함된 파일명을 처리하지 못하는 문제를 해결했습니다.
  - (이전 수정에 포함) 학습 시작 시 체크포인트 저장 디렉토리가 없어 `RuntimeError`가 발생하던 문제를 해결했습니다.
## v1.48
- **성능 대폭 개선 및 최적화 (GPT 지침 기반)**
  - `torch.autograd.set_detect_anomaly(True)`를 비활성화하여 치명적인 성능 저하 요인 제거.
  - SDPA(Flash Attention), TF32, `cudnn.benchmark` 등 PyTorch 성능 최적화 옵션을 활성화하여 GPU 연산 효율 극대화.
  - `CosineAnnealingLR` 스케줄러의 `step()` 호출을 배치 단위에서 에폭 단위로 수정하여 올바르게 동작하도록 변경.
  - `DataLoader`의 `num_workers`, `prefetch_factor` 등을 최적화하고, 데이터 전송 시 `non_blocking=True`를 사용하여 CPU-GPU 전송 병목 완화.
  - 학습 루프 내 불필요한 모델 상태 복제 로직을 완전히 제거하여 에폭 당 시간을 크게 단축.
- **이어학습(Resume) 기능 완전 개편**
  - 모델 가중치뿐만 아니라 Optimizer, Scheduler, Scaler, Epoch 상태까지 모두 포함하는 완전한 체크포인트 시스템 구현 (`last_*.ckpt`).
  - `checkpoint.py` 유틸리티를 추가하여 저장/복원 로직을 모듈화.
  - 이어학습 시 `last_*.ckpt` 체크포인트가 있으면 모든 상태를 복원하고, 없으면 최종 모델(`*.pth`)의 가중치만 불러와 학습을 재개하는 Fallback 로직 추가.
- **UI 및 서비스 안정성 강화**
  - UI 설정 탭에서 동작하지 않던 CPU/GPU 리소스 모니터링 기능(`get_status`)을 복원.
  - 학습 시작 시 사용 중인 GPU 모델을 명시적으로 로깅하여 사용자에게 투명성 제공.
## v1.47
- 이어학습(Resume) 모드를 추가하여 중단된 지점부터 학습을 재개하는 기능을 구현.
- 기존 '추가 파인튜닝' 기능을 이어학습으로 대체하고 UI 버튼을 '이어학습'으로 변경.
- 토크나이저를 기존 문자(Char) 방식에서 SentencePiece(BPE) 서브워드 방식으로 전면 교체.
  - `SentencePieceTokenizer` 래퍼 클래스를 `tokenizer.py`에 구현하고 ID 충돌 방지를 위한 시프팅 로직 적용.
  - `simple.py`, `service.py` 등 관련 모듈의 토크나이저 로직을 모두 `SentencePieceTokenizer`에 맞게 수정.
- `requirements.txt`에 `sentencepiece` 의존성을 추가하고 중복 항목을 정리.
- `config.py`에 이어학습 및 SPM 모델 경로 관련 설정을 추가.
- 사용자가 직접 SPM 모델을 생성할 수 있도록 `train_spm.py` 스크립트 추가.
## v1.46
- `cudnn.benchmark`를 전역에서 활성화해 반복 커널을 고정.
- DataLoader 기본 `num_workers`를 4로 조정하고 `persistent_workers`를 사용.
- 검증 로더는 `drop_last=False`로 유지하도록 수정.
- epoch 비율 DEBUG 로그를 제거해 I/O를 최소화.
- `config/default.yaml` 파일을 추가해 기본값을 명시.
## v1.45
- AMP 사용이 안정화되어 기본 설정에서 True로 전환.
- 기본 batch_size를 48로 확대하고 DataLoader가 마지막 미만 배치를 버리도록 수정.
- 시간 대비 라인 비율 로그를 DEBUG 레벨로 낮춰 불필요한 I/O를 줄임.
- 전반적인 학습 속도가 약 30% 향상됨을 확인.
## v1.44
- scaled_dot_product_attention 사용 시 Bool 마스크로 인해 전체 행이 ∞ 처리되어 NaN 손실이 발생하던 문제를 해결.
- Bool 마스크를 float(-1e4)로 변환하고 전부 가려진 행을 0으로 되돌려 안전한 Softmax를 보장.
## v1.43
- AMP 토글 상태가 에폭 간 유지되도록 학습 루프를 수정하고 FP32에서 NaN이 3회 이상 반복되면 학습을 중단하도록 변경.
- Auto-Tune이 혼합 정밀도를 사용하지 않고 학습률을 0.0002로 권장하도록 조정.
- 기본 설정의 learning_rate 값을 0.0002로 낮춤.
## v1.42
- run.py에서 `pretrain` 모드를 CLI 옵션으로 추가해 GUI와 동일하게 사용 가능하게 수정.
- 선택한 모드에 따라 `start_training("pretrain")`를 호출하도록 분기 처리.
## v1.41
- `src/training/simple.py`에서 로거 변수를 함수 안에서 재정의해 `UnboundLocalError`가 발생하던 문제를 수정.
- 모듈 상단에 전역 `logger`를 선언해 모든 함수에서 동일한 로거를 사용하도록 통일.

## v1.40
- float16 환경에서 NaN이 발생하는 문제를 수정하기 위해 모든 attention mask를 BoolTensor로 전달하도록 변경.
- 학습 중 손실이 비정상적일 경우 AMP를 자동으로 비활성화해 fp32로 재시작하도록 수정.
- `use_mixed_precision` 기본값을 False로 유지해 안전성을 확보.
## v1.39
- PAD로만 구성된 배치를 건너뛰어 NaN loss를 예방하도록 학습 루프를 수정.
- `cross_entropy` 계산 시 PAD 토큰을 무시하며 비정상 손실을 검증.
- 디버깅 편의를 위해 `torch.autograd.set_detect_anomaly(True)` 호출 추가.
## v1.38
- 사전학습 데이터 로드 시 2글자 미만 문장을 필터링하고 건너뛴 수를 기록.
- 배치의 타겟 길이가 2 미만이면 학습을 건너뛰어 NaN 손실을 예방.
## v1.37
- Transformer encoder에서 잘못된 src_mask 차원 계산을 제거하고
  padding mask만 사용하도록 수정.

## v1.36
- GPU 미사용 환경에서 GPU 마크 테스트를 자동 건너뛰도록 `conftest.py` 추가.
- 에폭 실행 속도 변동이 5% 미만이면 INFO로, 이상이면 WARNING으로 기록.
- 학습 완료 시 모델과 함께 `tokenizer.json`을 저장하도록 `train` 함수 개선.
## v1.35
- CUDA가 필수임을 명시하고 사용 불가 시 즉시 오류를 발생하도록 학습 초기화 코드를 수정.
- 배치 텐서와 스냅샷을 GPU와 동일한 장치로 강제하여 장치 불일치 문제를 방지.
- flash-attention 관련 UserWarning을 Transformer 모듈에서 전역적으로 차단.
- 모델 파라미터 스냅샷을 CPU 기준으로 통일해 변화율 계산을 안정화.
## v1.34
- 사전학습 루프가 3개 배치에서 멈추는 디버그 코드를 제거.
- 평균 손실 계산을 실제 처리 배치 수로 보정.
- 배치 소비 검증용 단위 테스트 추가.
## v1.33
- 학습 과정 검증을 위해 파라미터 변화율, 시간 대비 라인 비율을 기록하고
  모델 저장 전후 해시·용량 비교 및 사전학습 추론 검증 루틴을 추가.
  이전 로그와 동일할 경우 경고를 출력하도록 서비스 로직을 보강.
## v1.32
- 데이터셋 라인/중복/문자 수 등 통계와 encode 평균 시간을 디버그 로그로 출력.
- DataLoader 반복자 생성 시간과 collate_fn 실행 시간을 측정하도록 보강.
- 배치별 텐서 이동, forward, backward 시간을 기록해 초기 2개 배치 흐름을 세분화.
## v1.31
- 사전학습 모드에서 출력 문장을 입력으로 사용하도록 encode 로직 분기.
- 토큰 평균 길이와 인코딩 예시를 디버그 메시지로 출력하도록 보강.
## v1.30
- Flash Attention 경고 억제 시점을 학습 루프 시작 직전으로 이동.
- GPU 메모리 사용량과 데이터 평균 길이를 디버그 출력하도록 보강.
## v1.29
- Auto-Tune 스캔 로직이 `pretrain`, `finetune`, `additional_finetune` 폴더만 탐색하도록 조정.
- JSONL 처리 시 `instruction` 필드가 없는 라인은 샘플로 인정하지 않고 토큰 계산에서 제외.
- `clean_subtitles.py`와 문서의 경로를 `datas/pretrain/`으로 변경.
## v1.28
- Auto-Tuner가 모든 데이터셋 폴더를 순회해 샘플 수와 토큰 수를 계산하도록 수정.
- Flash Attention 경고 억제 코드를 transformer.py 최상단으로 이동.
- 디버그 로그에 탐색한 파일 수와 건너뛴 파일 목록을 출력하도록 보강.
## v1.27
- Auto-Tune 실행 시 디버그 메시지로 데이터셋 크기와 토큰 수 출력.
- 배치별 실행 시간을 측정해 3회만 표시하도록 학습 루프 수정.
- Flash Attention 경고 억제를 run.py 최상단으로 이동.
## v1.26
- Windows 환경에서 Flash Attention 경고를 무시하도록 조치.
- 모델 및 입력 텐서의 디바이스 상태를 디버그 로그로 출력.
- 에폭별 실행 시간을 로그에 기록.
## v1.25
- Auto-Tune 실행 시 데이터셋 크기와 토큰 수를 함께 출력하도록 로그 보강.
- 학습 함수에 토크나이저 로드, 데이터로더 구축, 에폭 실행 시간을 디버그 로그로 추가.
## v1.24
- 오토튠 후 페이지가 새로고침되면 설정 탭이 자동으로 열리도록 로컬 스토리지 연동 로직 추가.
## v1.23
- 로그를 단순 텍스트 형식으로 변경하고 학습 시작 시 설정값을 모두 출력하도록 개선.
## v1.22
- Auto-Tune 완료 후 페이지를 새로고침하여 설정이 바로 반영되도록 수정.

## v1.21
- Auto-Tune 버튼 결과가 각 입력 필드에 자동 반영되도록 UI 로직 수정.

## v1.20
- Auto-Tune이 dropout_ratio 값을 포함해 제안하며 UI 입력폼에 바로 적용되도록 개선.

## v1.19
- DataLoader가 배치 크기, num_workers, pin_memory 설정을
  Auto-Tune 결과와 사용자 설정 그대로 반영하도록 수정.

## v1.18
- Auto-Tune 결과에 필수 하이퍼파라미터가 모두 포함되도록 검증 로직을 강화.
- UI 적용 시 누락값을 즉시 알리고 오류를 발생시키도록 수정.

## v1.17
- 필수 입력이 비어 있으면 학습 버튼을 자동으로 비활성화하도록 UI 로직을 강화.
- 설정 로드를 마친 뒤 버튼 상태를 즉시 갱신해 Dead UI를 방지.

## v1.16
- 설정값 검증 모듈을 추가해 None이나 빈 값이 있으면 학습을 중단하도록 개선했다.
- Auto-Tune과 UI가 결과 값의 유효성을 확인하도록 수정했다.

## v1.15
- 릴리즈 마일스톤 요약 문서 추가.

## v1.14
- torch.compile 호출을 제거하고 DataLoader 기본값을 고정했다.
- Auto-Tune이 학습을 시작하지 않고 설정만 저장하도록 수정했다.
- UI에 Auto-Tune 결과가 즉시 반영되고 Start 버튼으로만 학습을 실행하도록 변경했다.

## v1.13
- DataLoader 설정을 수정해 학습 속도를 향상시켰다.
- Auto-Tune 결과가 UI에 바로 표시되도록 서비스와 스크립트를 수정했다.

## v1.12
- AutoTuner가 모든 데이터셋 크기를 계산해 더 정확한 설정을 추천하도록 개선.
- VRAM 조건을 만족하면 mixed precision 사용을 권장하도록 수정.
- UI에서 Auto-Tune 버튼 클릭 시 반환 데이터 처리 오류를 수정.

## v1.11
- 데이터 로더 기본값을 CPU 코어 수에 맞게 수정하고 배치 사이즈 확장.
- 학습 시작 시 사용 디바이스와 에폭별 진행률을 포함한 로그 강화.
- Changelog에서 날짜 표기를 제거.

## v1.10
- 학습 함수와 서비스 로직이 사용 중인 디바이스를 로그로 출력하도록 수정.
- CUDA가 없을 경우 경고를 남기고 CPU 사용을 알리도록 개선.
- AutoTuner가 VRAM/RAM 정보와 데이터 크기를 로그로 기록.

## v1.9
- 사전학습용 자막 정제 실행파일 `clean_subtitles.py` 추가.
- 학습 로직이 GPU 설정(`num_workers`, `pin_memory`, AMP)을 반영하도록 수정.

## v1.8
- 학습 루프가 에폭별 손실과 소요 시간을 로그로 출력하도록 개선.

## v1.7
- 자막 파일을 자동 정제하여 `datas/pretrain/`에 저장하는 로직 추가.
- pretrain 학습 시작 시 정제기가 실행되도록 ChatbotService 수정.
- 정제기 사용법 문서 `docs/subtitle_cleaner.md` 작성.

## v1.6
- delete_model 기능이 models 폴더의 모든 모델 파일을 일괄 삭제하도록 수정.
- 데이터셋 자동 병합 문서를 보강하여 원본 파일 무결성 원칙 명시.

## v1.5
- 폴더 내 전체 데이터셋 자동 통합 로딩 기능 구현.
- 데이터 중복 제거 및 경고 로그 처리 추가.
- 테스트 코드와 문서 업데이트.

## v1.4
- 학습 데이터를 pretrain/finetune/additional_finetune 폴더로 분리.
- start_training API가 학습 유형 인자를 받아 분기하도록 수정.
- 예시 데이터 파일 및 AGENTS.md 갱신.

## v1.3
- 오토 튠(HPO) 기능을 추가하여 데이터 크기와 하드웨어에 맞는 설정을 자동 제안.

## v1.2
- 학습 완료 및 모델 저장 로그를 이전 방식과 동일하게 출력하도록 수정.

## v1.1
- numpy와 Python 기반 배열 연산 모듈 및 테스트 코드 추가.

## v1.0
- UI 설정값을 학습 파이프라인과 서빙에 적용.
- Encoder-Decoder 구조의 커스텀 트랜스포머 구현.
- 자체 트랜스포머 학습 파이프라인 구현 및 관련 문서 업데이트.
- HFModel과 DummyModel은 테스트용임을 명시.
- 예외 입력 처리 로직 추가 및 테스트 코드 작성.
- HuggingFace 모델 연동 프로토타입 구현.
- UI 입력창을 단일 필드로 통합하고 서버 연동 로직 수정.
- 데이터셋 구조를 instruction + input 조합 방식으로 정리.
- DummyModel과 ChatbotService가 해당 구조만 사용하도록 수정.
- Instruction 방식 샘플 데이터 추가.
- 데이터 로더를 instruction/input/output 구조로 신규 작성.
- DummyModel 및 ChatbotService 구현.
- UI 입력 필드를 instruction 구조에 맞게 수정.
- run.py 모듈 경로 업데이트.

## 문서 작성 규칙
- 버전(vX.X)만 표기하고 최신 기록을 가장 위에 둔다.
