# Changelog

* 반드시 한국어로 작성할것.

## v0.2.0
- pywebview UI를 FastAPI 백엔드로 교체.
- 설정 시스템과 조기 중단 기능 추가.
- 시작 스크립트와 95% 커버리지의 테스트 제공.

## v0.3.0
- WebBackend를 새 ChatbotService API와 통합함.
- 최소 UI(ut.html)와 로그 뷰어 추가.
- 로깅 및 학습 흐름 개선.
- 테스트 커버리지를 90%까지 향상.

## v0.4.0
- 이전 ut.html 제거 후 ui.html 재사용.
- torch 선택적 지원으로 학습 단순화.
- 로그 파일명 패턴 검사와 테스트 개선.

## v0.5.0
- Komoran 시작 메시지와 GPU 경고 억제.
- 경고 필터링 테스트 추가.

## v0.6.0
- 기본적으로 DataLoader 병렬 작업자 비활성화.
- `get_komoran`으로 Komoran JVM 단일화 보장.
- soynlp 토크나이저의 `FutureWarning` 필터링.
- README에 GPU 설치 명령 문서화.

## v0.7.0
- `_encode_to_tensor` 사용 시 커스텀 토크나이저 추론 수정.

## v0.8.0
- 사전 토큰화 스크립트와 샤딩 데이터셋 로더 추가.
- 전체 데이터셋 모드를 위한 설정 및 README 업데이트.

## v0.9.0
- `generate_response`가 모델 시그니처를 확인해 안전하게 호출되도록 수정.

## v0.9.1
- inference 단계에서 generate 함수 인자 이름 오류 수정.

## v0.9.2
- run.py에서 ChatbotService 임포트를 main 내부로 이동해 torch 자동 설치 로직이
  항상 동작하도록 수정.
- 학습 파이프라인 CLI와 경로 환경 변수(DATA_DIR, CONFIG_PATH, MODEL_PATH) 지원.

## v0.9.3
- 커스텀 토크나이저에서 decode 오류 발생 시 호환 로직 추가.
- generate_utils 개선으로 빈 문자열 응답 문제 해결.


## v0.9.4
- 빈 응답 처리 로직 강화로 대화 말풍선 공백 오류 수정.

- 데이터셋 로딩 시 토큰 누락 샘플을 자동 보완

## v0.9.5
- generate_response 함수의 잘못된 인자명(src)을 input_ids로 수정해 모델 응답이 출력되지 않던 오류 해결.

## v0.9.6
- eos_token_id 미지원 토크나이저 사용 시 AttributeError가 발생하지 않도록 처리.

## v0.9.7

- 커스텀 모델 generate 호출 시 필수 인자인 src를 사용하도록 복원.

## v0.9.8
- generate_response 함수 디버깅을 위해 eos_token_id 제거 및 출력 텐서 로그 추가.

## v0.9.9
- 윈도우 콘솔 환경에서 인코딩 오류가 발생하지 않도록 출력 로그의 이모지를 제거하고
  텐서를 리스트로 변환해 출력.

## v0.9.10
- 디코딩 실패 원인 파악을 위해 _decode_tokens와 generate_response에 디버그 로그
  및 예외 처리를 추가.

## v0.9.11
- decode() 결과가 빈 문자열인 경우 수동 매핑 테이블을 사용해 복원하도록
  `generate_utils` 개선.


## v0.9.12
- 토크나이저 vocab 사이즈 로그 출력 및 manual_vocab 토큰 매핑 확장.
- 디코드 실패 시 fallback 동작을 검증하는 테스트 추가.
