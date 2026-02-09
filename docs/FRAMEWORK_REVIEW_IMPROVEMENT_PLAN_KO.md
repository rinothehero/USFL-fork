# Framework 전체 리뷰 상세 개선 문서 (Korean)

작성일: 2026-02-09  
대상 저장소: `USFL-fork`  
범위: `sfl_framework-fork-feature-training-tracker`, `GAS_implementation`, `multisfl_implementation`, `experiment_core`, `deploy`

## 1. 문서 목적

이 문서는 프레임워크 전체 코드 리뷰에서 발견한 핵심 이슈를 **문제 단위로 상세화**한 개선 설계 문서다.  
각 이슈마다 아래를 포함한다.

1. 증상
2. 근본 원인
3. 실제 영향
4. 재현 방법
5. 개선 방향
6. 검증 기준(완료 조건)

## 2. 우선순위 요약

| ID | 우선순위 | 제목 | 영향 범위 |
|---|---|---|---|
| F-001 | P0 | GAS GPU 강제 고정(`CUDA_VISIBLE_DEVICES=0`) | 분산 실행/멀티 GPU |
| F-002 | P1 | `scala` 검증 함수 중복으로 검증 누락 | 설정 유효성 |
| F-003 | P1 | `sfl-u` CLI 허용 vs 실행 라우팅 미지원 | 런타임 안정성 |
| F-004 | P1 | 모델 지원 매트릭스 불일치 | 실행 재현성 |
| F-005 | P2 | `parse_known_args` fallback으로 오타 은닉 | 실험 신뢰성 |
| F-006 | P2 | 서버 config 객체 직접 mutate | 동시성/상태오염 |
| F-007 | P2 | `experiment_core/test_adapters.py` 구버전 계약 의존 | 회귀 검증 |
| F-008 | P2 | busy-wait + 매 라운드 전체 JSON 재기록 | 성능/확장성 |
| F-009 | P3 | emulation GPU 배정 `% 4` 하드코딩 | 이식성 |
| F-010 | P3 | adapter raw result 탐색 시 상대 경로 취약 | 실험 자동화 안정성 |

---

## 3. 상세 이슈

### F-001 (P0): GAS가 GPU를 강제로 0번으로 고정

### 문제 설명
`GAS_main.py`가 실행 초기에 `CUDA_VISIBLE_DEVICES=0`을 강제한다.  
이 때문에 외부 오케스트레이터(`experiment_core`, `deploy.sh`)가 지정한 GPU 정책이 무시된다.

### 근거 코드
- `GAS_implementation/GAS_main.py:47`
- `experiment_core/batch_runner.py:127`

### 증상
1. 여러 GAS 실험을 병렬로 돌려도 실제 계산이 동일 물리 GPU로 몰림
2. GPU OOM/성능 급락
3. 배치 스펙의 `gpu` 값과 실제 사용 GPU가 불일치

### 근본 원인
런처 계층과 프레임워크 계층의 책임 분리가 깨져 있음.

1. 런처(`batch_runner`)가 `CUDA_VISIBLE_DEVICES`를 설정
2. GAS 코드가 다시 하드코딩으로 덮어씀

### 영향
1. 분산 실행 안정성 저하
2. 실험 결과 재현성 저하
3. 서버 자원 활용 불균형

### 재현 방법
1. 서로 다른 GPU를 배정한 GAS 2개 실험을 동시 실행
2. 런타임 로그에서 GPU 사용 확인
3. 두 실험 모두 동일 GPU(0) 사용 확인

### 개선안
1. `GAS_main.py`에서 하드코딩 제거
2. `CUDA_VISIBLE_DEVICES`는 런처에서만 책임지도록 고정
3. 시작 로그에 실제 디바이스/visible devices 출력

### 검증 기준
1. 배치 스펙의 GPU 매핑과 실제 디바이스 사용이 일치
2. 병렬 실행 시 GPU 충돌이 재현되지 않음
3. 단일/다중 서버 배포 모두 동일 동작

---

### F-002 (P1): `scala` 검증 함수 중복 정의로 검증 누락

### 문제 설명
`server/main.py`에 `_validate_scala_parameters`가 2번 선언되어, 뒤의 선언이 앞 선언을 덮어쓴다.

### 근거 코드
- `sfl_framework-fork-feature-training-tracker/server/main.py:238`
- `sfl_framework-fork-feature-training-tracker/server/main.py:252`
- `sfl_framework-fork-feature-training-tracker/server/main.py:305`

### 증상
1. `scala` 실행 시 split 관련 필수 조건 검증이 일부 누락
2. 잘못된 설정이 실행 중간에 터지는 형태로 지연 실패

### 근본 원인
중복 함수 정의가 정적 검사 없이 병합됨.

### 영향
1. 사전 검증 단계 신뢰성 저하
2. 장시간 실험 도중 실패 가능성 증가

### 재현 방법
1. `scala`에 부적절한 split 설정 전달
2. 원래라면 시작 전 실패해야 하나, 일부 케이스가 통과

### 개선안
1. `_validate_scala_parameters` 단일 함수로 통합
2. split strategy/split ratio/criterion 검증을 한곳에서 수행
3. validator 테스트 추가

### 검증 기준
1. 유효하지 않은 `scala` 설정이 시작 전에 100% 실패
2. 정상 설정만 통과

---

### F-003 (P1): `sfl-u`는 인자에서 허용되지만 stage organizer 라우팅이 없음

### 문제 설명
CLI/검증 계층은 `sfl-u`를 인정하나, 실행 라우터(`get_stage_organizer`)에 구현이 없다.

### 근거 코드
- `sfl_framework-fork-feature-training-tracker/server/server_args.py:310`
- `sfl_framework-fork-feature-training-tracker/server/main.py:102`
- `sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/stage_organizer.py:162`

### 증상
1. 인자 파싱과 초기 검증은 통과
2. 런타임에서 `Invalid method` 예외

### 근본 원인
“지원 메서드 목록”이 계층별로 분산되어 있고 단일 소스가 없음.

### 영향
1. 사용자 혼란 (지원되는 줄 알았는데 실행 실패)
2. 자동 실험 배치에서 후반 실패

### 재현 방법
1. `-M sfl-u`로 실행
2. stage organizer 라우팅 시점에서 예외 확인

### 개선안
옵션 A 또는 B 중 하나를 명확히 선택해야 한다.

1. **옵션 A(비지원 고정):** parser/validator에서 `sfl-u` 제거
2. **옵션 B(지원 복원):** `SFL-U StageOrganizer` 구현 및 라우팅 추가

### 검증 기준
1. 문서/CLI/라우팅/테스트에서 `sfl-u` 상태가 일관됨
2. 사용자 관점에서 “보이는데 안 되는” 상태가 사라짐

---

### F-004 (P1): 모델 지원 목록 불일치 (`resnet18_flex`, `resnet18_cifar`)

### 문제 설명
Parser/모델 팩토리는 허용하지만, workload 검증이 일부 모델을 거절한다.

### 근거 코드
- `sfl_framework-fork-feature-training-tracker/server/server_args.py:287`
- `sfl_framework-fork-feature-training-tracker/server/main.py:47`
- `sfl_framework-fork-feature-training-tracker/server/modules/model/model.py:22`

### 증상
1. 유효해 보이는 모델 설정이 validation 단계에서 거절
2. simulation/emulation 동작 체계가 달라 보이는 인상

### 근본 원인
지원 모델 목록이 다중 위치에 하드코딩.

### 영향
1. 실험 재현성 저하
2. 모델 전환 시 예기치 않은 실패

### 재현 방법
1. `resnet18_flex` 설정으로 실행
2. 경로에 따라 통과/실패 불일치 관찰

### 개선안
1. 지원 모델 레지스트리 단일화
2. parser/validator/factory가 같은 registry를 참조

### 검증 기준
1. 모든 계층에서 동일한 model capability 적용
2. 모델별 허용/비허용 이유가 명확히 출력

---

### F-005 (P2): `parse_known_args` fallback이 설정 오류를 숨김

### 문제 설명
서버/클라이언트 인자 파싱에서 broad `except` 후 `parse_known_args`로 fallback한다.

### 근거 코드
- `sfl_framework-fork-feature-training-tracker/server/server_args.py:1138`
- `sfl_framework-fork-feature-training-tracker/client/client_args.py:240`

### 증상
1. 오타 인자/잘못된 플래그가 조용히 무시될 수 있음
2. “설정 적용됐다”는 착시 발생

### 근본 원인
운영 안정성 대신 유연성 위주의 fallback이 기본값으로 들어감.

### 영향
1. 실험 결과 해석 오류
2. 재현 불가능한 실험 증가

### 재현 방법
1. 의도적으로 존재하지 않는 옵션 전달
2. 즉시 실패 대신 진행 여부 확인

### 개선안
1. 기본은 strict parsing (`parse_args`)
2. fallback은 `--allow-unknown-args` 같은 명시 플래그일 때만 허용
3. 허용 시에도 unknown 목록을 경고로 반드시 출력

### 검증 기준
1. 잘못된 인자는 기본적으로 즉시 실패
2. unknown 허용 모드에서 경고 로그 필수

---

### F-006 (P2): 서버 config 객체를 요청 경로에서 직접 수정

### 문제 설명
요청 핸들러가 `self.config.__dict__`를 직접 수정하여 client별 `mask_ids`를 삽입한다.

### 근거 코드
- `sfl_framework-fork-feature-training-tracker/server/modules/ws/handler/fl_handler.py:22`

### 증상
1. 동시 요청 시 상태 오염 가능성
2. 다른 클라이언트 요청에 이전 값이 섞일 위험

### 근본 원인
공유 객체를 per-request payload로 재사용함.

### 영향
1. 데이터 마스킹/분배 관련 이상 동작 가능
2. 디버깅 난이도 증가

### 재현 방법
1. 다수 클라이언트가 동시에 server config 요청
2. 요청/응답 페이로드 비교로 오염 여부 확인

### 개선안
1. `configs = dict(self.config.__dict__)`로 복사본 사용
2. 요청별 필드는 복사본에만 삽입

### 검증 기준
1. 동시 요청에서도 응답 페이로드가 client별로 독립
2. 서버 전역 config는 불변 유지

---

### F-007 (P2): `experiment_core/test_adapters.py`가 현재 구현과 불일치

### 문제 설명
테스트가 구버전 어댑터 인터페이스를 가정한다. 현재는 SFL이 `sfl_runner.py`를 경유한다.

### 근거 코드
- `experiment_core/test_adapters.py:84`
- `experiment_core/test_adapters.py:117`
- `experiment_core/adapters/sfl_adapter.py:38`

### 확인된 실제 증상
`python -m experiment_core.test_adapters` 실행 시 초기 테스트에서 실패.

### 근본 원인
adapter 구조 변경 후 테스트 계약 업데이트 누락.

### 영향
1. 회귀 검증 기능 상실
2. 어댑터 변경 시 릴리즈 안정성 저하

### 개선안
1. 테스트를 “현재 계약” 기준으로 재작성
2. build_command 결과뿐 아니라 spec->workload 매핑 검증 강화
3. CI에 mandatory 체크로 편입

### 검증 기준
1. 로컬/CI에서 테스트 100% 통과
2. adapter 변경 시 테스트가 즉시 회귀를 탐지

---

### F-008 (P2): busy-wait와 매 라운드 full JSON rewrite로 확장성 저하

### 문제 설명
in-round 대기 루프가 초고빈도 polling을 수행하고, 서버는 매 라운드 결과 JSON 전체를 다시 기록한다.

### 근거 코드
- `sfl_framework-fork-feature-training-tracker/server/modules/trainer/stage/in_round/in_round.py:67`
- `sfl_framework-fork-feature-training-tracker/server/modules/trainer/trainer.py:93`
- `sfl_framework-fork-feature-training-tracker/server/modules/global_dict/global_dict.py:136`

### 증상
1. CPU 낭비 증가
2. 라운드 수가 커질수록 I/O 비용 증가
3. 큰 metric 객체일 때 저장 지연

### 근본 원인
1. 이벤트 신호 기반이 아닌 polling 기반 동기화
2. append 방식이 아닌 full dump 방식 저장

### 영향
1. 대규모 실험 처리량 저하
2. tail latency 증가

### 개선안
1. `asyncio.Condition/Event` 기반 대기 구조로 전환
2. metric 저장을 append 로그 또는 checkpoint 간격 저장으로 변경
3. 필요 시 `orjson` + line-delimited JSON 도입

### 검증 기준
1. 동일 워크로드에서 CPU 사용률 유의미 감소
2. 라운드당 저장 시간 감소

---

### F-009 (P3): emulation GPU 배정 `% 4` 하드코딩

### 문제 설명
클라이언트 프로세스 GPU 배정을 `i % 4`로 고정한다.

### 근거 코드
- `sfl_framework-fork-feature-training-tracker/emulation.py:53`

### 증상
1. 4 GPU가 아닌 환경에서 비효율/오동작 가능
2. 외부 스케줄러 정책과 불일치

### 개선안
1. 실제 visible GPU 개수를 기반으로 라운드로빈
2. 또는 런처에서 전달된 디바이스 맵 사용

### 검증 기준
1. 1/2/4/8 GPU 환경 모두 정상 할당
2. 의도된 디바이스 정책과 실제 실행 결과 일치

---

### F-010 (P3): adapter raw result 탐색의 상대 경로 취약

### 문제 설명
`find_latest_raw_result`에서 `execution.cwd`가 상대 경로일 경우 repo-root 기준 resolve가 누락된 경로가 사용될 수 있다.

### 근거 코드
- `experiment_core/adapters/sfl_adapter.py:71`
- `experiment_core/adapters/gas_adapter.py:148`
- `experiment_core/adapters/multisfl_adapter.py:126`

### 증상
1. 환경에 따라 결과 파일 탐색 실패
2. 간헐적 “Could not locate raw result file” 오류

### 개선안
1. run 경로와 동일하게 `repo_root` 기준 resolve 적용
2. raw_result_glob 탐색 대상 경로를 로그로 출력

### 검증 기준
1. 상대/절대 `cwd` 모두 동일 동작
2. CI에서 결과 탐색 flakiness 제거

---

## 4. 구조적 개선 제안 (Cross-cutting)

### 4.1 단일 Capability Registry 도입

아래를 코드 한 곳에서 관리하고 전 계층이 참조하도록 통합:

1. 지원 method 목록
2. method별 허용 model/dataset/split 전략
3. 선택 가능한 selector/aggregator 조합

기대 효과:
1. parser/validator/router/factory 불일치 제거
2. 신규 방법 추가 시 누락 위험 감소

### 4.2 Adapter Contract Test 표준화

`experiment_core`에 아래 테스트를 표준 계약으로 고정:

1. spec 입력 -> command/env 변환 일치
2. framework별 raw result 탐색 규칙
3. normalization schema 유효성

### 4.3 관측성 강화

1. 모든 런타임이 시작 시 실제 device/cuda visibility 출력
2. drift/g/alignment 계산 스킵 조건 로그 명시
3. 결과 파일 경로 탐색 과정 로그 추가

---

## 5. 실행 순서 권장

### Phase 1 (즉시, 안정성)
1. F-001 (GAS GPU 강제값 제거)
2. F-002 (scala 검증 함수 통합)
3. F-003/F-004 (지원 매트릭스 정합성 수정)

### Phase 2 (신뢰성)
1. F-005 (strict parsing)
2. F-006 (config 불변성)
3. F-007 (테스트 갱신 + CI 연동)

### Phase 3 (성능/운영성)
1. F-008 (polling/I-O 개선)
2. F-009/F-010 (이식성/탐색 안정성)

---

## 6. 완료 정의 (Definition of Done)

다음 조건을 모두 만족하면 본 문서 이슈를 “해결”로 본다.

1. P0/P1 이슈가 코드와 테스트에서 모두 해소됨
2. `experiment_core` 어댑터 테스트가 현재 계약 기준으로 통과
3. 최소 1회 실험 배치 실행에서 GPU/결과 경로/로그 정합성 확인
4. 관련 문서(`docs/`)에 최신 동작 기준 반영

