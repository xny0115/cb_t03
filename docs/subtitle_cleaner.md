# Subtitle Cleaner

프로젝트 루트에 존재하는 `.smi`, `.srt` 자막 파일을 사전학습용 텍스트로 변환한다.
사전학습을 시작할 때 자동으로 실행되어 자막에서 한국어 대사만 추출한다.

## 사용법
1. 자막 파일을 루트 디렉터리에 배치한다.
2. `ChatbotService.start_training("pretrain")` 호출 시 정제기가 실행된다.
3. 추출된 문장은 `datas/pretrain/` 폴더의 동일한 이름의 `.txt` 파일로 저장된다.

모든 태그와 타임코드는 제거되며 한 줄에 하나의 대사만 남는다.
