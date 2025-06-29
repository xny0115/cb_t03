# 변경 이력
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
