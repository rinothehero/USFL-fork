# Experiment A: `B_c_vs_sfl_iid_mu` Metric

이 문서는 라운드별 클라이언트 합의 업데이트 방향을 SFL-IID 기준과 비교하는 메트릭을 설명한다.

메트릭 정의:

- `mu_c^t(method) = Σ_i w_i * Δ_{c,i}^t`
- `mu_c^t(iid) =` SFL-IID run에서 저장한 동일 라운드의 합의 업데이트
- `B_c_vs_sfl_iid_mu^t = 1 - cos(mu_c^t(method), mu_c^t(iid))`

해석:

- `0`에 가까울수록 SFL-IID 방향과 정렬
- `1` 근처는 직교
- `> 1`은 반대 방향

주의:

- 이 값은 "같은 라운드 번호 기준 IID 경로 정렬"이다.
- "같은 모델 상태에서의 counterfactual 비교"와는 다르다.

## 1) SFL-IID reference 생성 (pass-1)

SFL-IID run에 아래 옵션을 켠다.

- SFL framework (`simulation.py` / spec):
  - `--expa-iid-mu-save-dir <DIR>`
- GAS:
  - `GAS_EXPA_IID_MU_SAVE_DIR=<DIR>`
- MultiSFL:
  - `--expa_iid_mu_save_dir <DIR>`

저장 포맷:

- `<DIR>/round_0001.pt`, `<DIR>/round_0002.pt`, ...
- 각 파일은 `{ "round": t, "mu": tensor(float16) }`

## 2) 비교 대상 run (pass-2)

비교할 각 기법 run에서 reference path를 로드한다.

- SFL framework:
  - `--expa-iid-mu-load-path <DIR_OR_FILE>`
- GAS:
  - `GAS_EXPA_IID_MU_LOAD_PATH=<DIR_OR_FILE>`
- MultiSFL:
  - `--expa_iid_mu_load_path <DIR_OR_FILE>`

## 3) 결과 키

raw `drift_history` / normalized `experiment_a_history`에 아래 키가 기록된다.

- `expA_B_c_vs_sfl_iid_mu`
- `expA_cos_c_vs_sfl_iid_mu`
- `expA_sfl_iid_mu_round_available`

round별 상세(`experiment_a`)에는 저장 성공 시 아래 키가 추가될 수 있다.

- `sfl_iid_mu_saved_path`

## 4) experiment_core common.json

통합 spec/배치 실행에서는 공통 설정으로 아래 키를 사용한다.

- `expa_iid_mu_save_dir`
- `expa_iid_mu_load_path`
