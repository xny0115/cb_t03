# 변경 이력

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
- 자막 파일을 자동 정제하여 `datas/01_pretrain/`에 저장하는 로직 추가.
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
