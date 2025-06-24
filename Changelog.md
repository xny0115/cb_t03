# 변경 이력

## 2025-06-25
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
- 작성일(YYYY-MM-DD) 표기, 최신 기록이 가장 위에 위치.
- 미래일이나 임의 날짜 사용 금지.
