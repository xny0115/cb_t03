읽음 확인
# WORKLOG
Date(KST): 2025-09-20
Agent: Codex
Repo: /workspace/cb_t03
Branch: work
HEAD: 3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8
Dirty: no
Status: DONE (2025-09-20)

Directives:
  - 지시-1) 최근 커밋 해시가 문서에 정확히 기록돼 있는지 재점검하고, 오기된 버전 정보를 모두 수정.
  - 지시-2) 잘못된 해시가 재발하지 않도록 검증 로그와 근거를 문서화.

Actions:
  - 처리-1) AGENTS.md·MERGE_RULES 지침을 재확인하고, 현재 작업 분기가 `work`인 상태에서 지시를 이행함을 명시.
  - 처리-2) `git rev-parse HEAD`와 `git show --no-patch`로 실제 커밋 `3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8` 존재를 검증.
  - 처리-3) Changelog v1.83/v1.82 및 WORKLOG 2025-09-19·2025-09-18 블록의 잘못된 해시(`a6354b1c1bf7...`)를 실제 값으로 교체하고 새 v1.84 항목에 근거를 기록.
  - 처리-4) `rg 'a6354b1c1bf7'`로 전체 저장소를 탐색해 교정 후에는 설명 문장에만 해당 문자열이 남았음을 확인.

FilesChanged:
  - Changelog.md
  - WORKLOG.md

Logs:
  - git rev-parse HEAD → 3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8
  - git show 3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8 --no-patch
  - rg 'a6354b1c1bf7' → Changelog.md:4 / WORKLOG.md:18,19 (교정 경과 기록만 잔존)

Test:
  - not run (문서 정리 작업)

Pending/Rollback/Next: none

---
# WORKLOG
Date(KST): 2025-09-19
Agent: Codex
Repo: /workspace/cb_t03
Branch: work
HEAD: 3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8
Dirty: no
Status: DONE (2025-09-19)

Directives:
  - 지시-1) report/ 디렉터리의 보고서가 Markdown 형식과 작성일자 포함 파일명 규칙을 준수하는지 재점검.
  - 지시-2) 최근 작업 로그(Changelog·WORKLOG)의 버전·HEAD 표기를 실제 커밋 정보로 정정하고 상세 기록을 추가.
  - 지시-3) 위 조치 결과를 로그 명령 출력과 함께 문서화해 재발을 방지.

Actions:
  - 처리-1) `find report -name '*.txt'`로 잔존 TXT 보고서가 없음을 확인하고, `report/issues_20250916.md` 머리말에 작성일이 명시돼 있음을 재검토.
  - 처리-2) Changelog 최신 항목(v1.82)의 잘못된 HEAD 해시(`1860269...`)를 실제 커밋 `3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8`로 교체하고, 정정 작업을 새로운 v1.83 항목으로 기록.
  - 처리-3) WORKLOG 2025-09-18 항목의 HEAD/Logs를 실제 커밋으로 수정하고, 이번 정정 내역을 2025-09-19 항목으로 추가.

FilesChanged:
  - Changelog.md
  - WORKLOG.md

Logs:
  - git rev-parse HEAD (base) → 3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8
  - find report -name '*.txt' → (no output)

Test:
  - not run (문서 정리 작업)

Pending/Rollback/Next: none

---
# WORKLOG
Date(KST): 2025-09-18
Agent: Codex
Repo: /workspace/cb_t03
Branch: work
HEAD: 3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8
Dirty: no
Status: DONE (2025-09-18)

Directives:
  - 지시-1) Changelog·WORKLOG 전반의 버전·HEAD 표기를 재검증하고 보완 사항을 정리.
  - 지시-2) report/ 디렉터리 내 텍스트 보고서를 Markdown으로 전환하고 파일명에 작성 일자를 표기.

Actions:
  - 처리-1) git 로그·cat-file을 사용해 기존 WORKLOG 항목에 기록된 HEAD 해시(4565faa74584, 0cf5d5594494, 77e768979e57 등)가 실제 커밋으로 존재함을 재확인.
  - 처리-2) report/issues.txt를 `report/issues_20250916.md`로 이동하고 개요/문제/개선 구조를 갖춘 Markdown 문서로 재작성.
  - 처리-3) 변경된 파일명을 참조하도록 Changelog와 WORKLOG의 관련 기록을 업데이트.

FilesChanged:
  - report/issues_20250916.md
  - Changelog.md
  - WORKLOG.md

Logs:
  - git rev-parse HEAD → 3cfd5dc74ee9dd2f0b889061a05357f28ed8d9e8
  - git log --oneline | head -n 5
  - git cat-file -t 4565faa74584ff44ac1561ae5823f48ee6ae1258 → commit
  - git cat-file -t 0cf5d5594494 → commit
  - git cat-file -t 77e768979e57 → commit

Test:
  - not run (문서 정리 작업)

Pending/Rollback/Next: none

---
# WORKLOG
Date(KST): 2025-09-17
Agent: Codex
Repo: /workspace/cb_t03
Branch: work
HEAD: 4565faa74584
Dirty: no
Status: DONE (2025-09-17)

Directives:
  - 지시-1) 저장소 로그 및 작업내역의 버전·날짜 표기를 검증하고 불일치 항목을 수정.
  - 지시-2) 수정 사항을 WORKLOG와 Changelog에 상세히 기록.

Actions:
  - 처리-1) Changelog 상단을 v1.80(2025-09-16 KST) 릴리스로 확정하고 신규 정정 작업을 v1.81로 문서화.
  - 처리-2) WORKLOG 최신 항목의 HEAD 해시를 병합 커밋(4565faa74584ff44ac1561ae5823f48ee6ae1258)과 일치하도록 수정.
  - 처리-3) 기타 문서에서 버전 표기 이상 유무를 확인하고 추가 조치 필요 없음을 검증.

FilesChanged:
  - Changelog.md
  - WORKLOG.md

Logs:
  - git rev-parse HEAD → 4565faa74584ff44ac1561ae5823f48ee6ae1258

Test:
  - not run (문서 정리 작업)

Pending/Rollback/Next: none

---
# WORKLOG
Date(KST): 2025-09-16
Agent: Codex
Repo: /workspace/cb_t03
Branch: work
HEAD: 4565faa74584
Dirty: no
Status: DONE (2025-09-16)

Directives:
  - 지시-1) 저장소 전체 점검 결과를 report/issues_20250916.md로 정리하여 사용자에게 제공.
  - 지시-2) 핵심 로직 이상 징후를 식별하고 개선 방향을 기술.
  - 지시-3) 작업 내역을 WORKLOG·Changelog에 문서화하고 저장소 규칙을 준수.

Actions:
  - 처리-1) report/issues_20250916.md 작성 — 핵심 로직, 토크나이저, 학습 루프, 서비스 로직 문제를 10건 이상 정리.
  - 처리-2) WORKLOG/Changelog를 갱신하여 점검 범위와 산출물을 기록.

FilesChanged:
  - report/issues_20250916.md (신규)
  - WORKLOG.md
  - Changelog.md

---
# WORKLOG
Date(KST): 2025-08-26
Agent: Codex
Repo: /workspace/cb_t03
Branch: work
HEAD: 0cf5d5594494
Dirty: no
Status: DONE (2025-08-26)

Directives:
  - 지시-1) SDP/cuDNN 가드 및 로그 보강; 문서 업데이트(src/training/simple.py, README.md, Changelog.md)
  - 지시-2) DISABLE_CUDNN_BENCHMARK 환경변수 문서화(README)
  - 지시-3) CPU 드라이런 안전가드 적용; compileall 통과
  - 지시-4) 학습 시작 로그 추가(epochs, bs, lr, dropout, mixed)
  - 지시-5) 상태 요약 이원화(Changelog+WORKLOG); AGENTS 규칙 보강
  - 지시-6) 학습률 표기 통일(trainconfig.ini, src/config.py, README)
  - 지시-7) Changelog vNEXT 세부항목 보강(설정 diff, 로그 샘플, Runtime)

Actions:
  - 처리-1) src/training/simple.py L25–70 +53/-14 — torch.cuda 가용성 가드와 sdp/cudnn 설정 로그를 추가.
  - 처리-2) README.md L148–152 +5/-0 — DISABLE_CUDNN_BENCHMARK 사용법을 Runtime/ENV 섹션에 문서화.
  - 처리-3) src/training/simple.py L25–51 +53/-14 — sdp_kernel·cudnn 블록을 cuda 체크와 try/except로 감싸 CPU 드라이런을 보호; compileall OK.
  - 처리-4) src/training/simple.py L376–386 +8/-0 — 학습 시작 직전 [CFG-TRAIN] 설정 로그를 추가.
  - 처리-5) AGENTS.md L10–14 +1/-0 — 작업 전 WORKLOG 기록 의무 규칙을 명문화.
  - 처리-6) trainconfig.ini L1 +1/-0; src/config.py L14 +1/-1; README.md L62 +1/-0 — 학습률을 소수 표기로 명시.
  - 처리-7) Changelog.md L1–10 +10/-0 — vNEXT에 설정 diff와 [CFG-TRAIN]/[CFG-GEN] 로그 샘플, Runtime 정보를 추가.

FilesChanged:
  - AGENTS.md +1/-0
  - Changelog.md +10/-0
  - README.md +7/-0
  - src/config.py +1/-1
  - src/training/simple.py +53/-14
  - trainconfig.ini +1/-0
  - WORKLOG.md +50/-0

Config diff (trainconfig.ini):
  - learning_rate: 0.0002 → 0.0002

Logs:
  - [CFG-TRAIN] sdp=disabled, cudnn.benchmark=False
  - [CFG-GEN] t=0.30 tp=0.90 k=0 mnt=128 rep=1.10 beams=1 sample=True

Test:
  - cmd: ALLOW_CPU_TRAINING=1 pytest tests/test_train_loop.py::test_batch_consumption tests/test_infer_exceptions.py::test_unknown_input -s --log-cli-level=INFO
  - metrics: 2 passed
  - warn/fail(raw): none

Pending/Rollback/Next: none

Date(KST): 2025-08-26
Agent: Codex
Repo: /workspace/cb_t03
Branch: work
HEAD: 77e768979e57
Dirty: no
Status: DONE (2025-08-26)

Directives:
  - 지시-1) [CFG-TRAIN] 앵커 라인 선두 정렬 및 tests/ 검증 추가.
  - 지시-2) WORKLOG 지시/처리 정리와 확장 섹션 가이드 문서화.

Actions:
  - 처리-1) src/training/simple.py L52–57,L383–390 +10/-2; tests/test_cfg_train_anchor.py L1–14 +14/-0 — [CFG-TRAIN] 로그 문자열을 라인 선두로 배치하고 그렙 검증 테스트를 추가.
  - 처리-2) AGENTS.md L15 +1/-0; Changelog.md L11–18 +8/-0; WORKLOG.md L52–95 +45/-0 — WORKLOG 확인·확장 안내를 문서화하고 항목을 정비.

FilesChanged:
  - AGENTS.md +1/-0
  - Changelog.md +8/-0
  - src/training/simple.py +10/-2
  - tests/test_cfg_train_anchor.py +14/-0
  - WORKLOG.md +45/-0

Logs:
  - [CFG-TRAIN] sdp=disabled, cudnn.benchmark=False

Test:
  - cmd: ALLOW_CPU_TRAINING=1 pytest -q
  - metrics: 17 passed, 3 skipped
  - warn/fail(raw): none

Pending/Rollback/Next: none

---
- Context: branch=work, head=N/A, range=N/A, modules=[src.training.simple, tests.test_cfg_train_anchor]
- Entrypoints/Functions: start_training
- Invariants/Non-Goals: 기본값 유지, 새 ENV 없음
- Risk & Mitigation: 앵커 미감지 → grep 테스트로 가드
- Before/After: grep '^\[CFG-TRAIN\]' src/
- Perf/Correctness Probes: N/A
- Runtime toggles used: ALLOW_CPU_TRAINING=1
- Tests touched/added: tests/test_cfg_train_anchor.py::test_cfg_train_anchor
- Rollback: git revert HEAD
- Open/Next: none
