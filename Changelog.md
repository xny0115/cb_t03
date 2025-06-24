# 변경 이력

## v1.3 - 2025-06-25
- 오토 튠(HPO) 기능을 추가하여 데이터 크기와 하드웨어에 맞는 설정을 자동 제안.

## v1.2 - 2025-06-24
- 학습 완료 및 모델 저장 로그를 이전 방식과 동일하게 출력하도록 수정.

## v1.1 - 2025-06-24
- numpy와 Python 기반 배열 연산 모듈 및 테스트 코드 추가.

## v1.0 - 2025-06-25
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
- 버전(vX.X)과 작성일(YYYY-MM-DD)을 함께 표기하고 최신 기록을 가장 위에 둔다.
- 미래일이나 임의 날짜 사용 금지.
