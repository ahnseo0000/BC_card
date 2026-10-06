# 상권 생존 지도 — 설명 가능한 상권 생존 분석

BC카드 소비데이터 공모전 출품작. **소비는 있는데 왜 못 버티는가**를 점포 단위 폐업 기록(생존분석)과 BC카드 소비 구성으로 설명하고, 결과를 지도 웹사이트로 제공한다.

- **웹사이트**: https://ahnseo0000.github.io/BC_card/ (이 저장소의 `docs/index.html`, GitHub Pages)
- **분석 기록(방법·검증·정정 이력 전부)**: [CHANGELOG.md](CHANGELOG.md) — ⑰~㉚이 이 README가 다루는 최근 작업
- **데이터 소스·업종 매핑 정리**: [DATA_MAPPING.md](DATA_MAPPING.md)
- **프로젝트 계획(확정 사항·진행 현황)**: [BC카드_공모전_프로젝트_계획.md](BC카드_공모전_프로젝트_계획.md)

> 이 프로젝트의 모든 결과는 **통계적 연관**이며 인과나 정책 효과가 아니다.

---

## 한눈에

| 항목 | 내용 |
|---|---|
| 분석 대상 | 2026-01-01에 영업 중이던 점포 635,536개(한식계열·일식회집·중국음식·서양음식·스낵·제과점·편의점 7개 업종)를 180일 추적, 폐업 25,329건(3.99%) |
| 결과 변수 | LOCALDATA 인허가 4종의 폐업일자 (점포 단위) |
| 설명 변수 | 점포(영업연수·프랜차이즈·입지·규모 등), 지역 폐업 이력(시군구·이웃 시군구), BC카드 시군구×업종 성별·연령 구성·소비 규모(2026-01~06) |
| 최종 생존 모형 | Cox 비례위험 모형 **M3L** — 선형이라 요인별 기여를 정확히 분해할 수 있다 |
| 검증 | 그룹(시군구×업종) 5-fold, 시군구 5-fold, 전향(시간 밖) 검증(6/30 영업 점포를 9/16까지 78일 추적) |
| 부가 분석 | 상권 유형화(5유형), 단기 소비 추이 예측, 연령 구성 변화(고객 분석) |

### 핵심 결과

1. **지역이 업종보다 크다** — 그룹 간 폐업률 차이를 더미변수로 설명한 정도(조정 R²)가 시군구만 0.44, 업종만 0.12로 약 4배 차이(CHANGELOG ⑳).
2. **점포 단위 판별은 영업연수·프랜차이즈·점포 특성이, 지역 간 순위는 자기·이웃 지역의 최근 폐업 흐름이 만든다** — 블록을 하나씩 빼고 재검증한 결과(㉓).
3. **BC카드 성별·연령 구성은 어떤 유형의 지역인지 알려 주는 맥락 정보다** — 같은 시군구 안에서는 폐업 위험을 가르지 못했다(⑱). 소비 규모·객단가·성장률 등 추가 지표도 검증했지만 예측 순위를 더 개선하지 못했다(⑲㉑).
4. **잔차에 남은 공간 구조는 이웃 시군구의 최근 폐업 흐름 하나로 거의 설명된다** — GNN 없이 Moran's I가 0.157→0.023으로 사라졌다(㉑㉒).
5. **전향 검증으로 시간이 지나도 맞는지 확인했다** — 6월 30일 영업 점포를 9월 16일까지 78일 추적, 점포 단위 C-index 0.627, 위험 최상위 10% 점포의 실제 폐업률이 최하위 10%의 약 4.3배(㉘).
6. **단기 소비 추이는 점포·고객 구성 정보로 예측이 개선된다** — 지속성(4월 값 그대로) 대비 점포·고객 구성을 더한 모형이 MAPE를 8.8%→6.8%로 낮췄다(㉙). 다만 BC 집계 단위는 표본이 커서 지속성만으로도 이미 매우 정확하다.
7. **고객 연령 구성의 변화(상반기→하반기)는 위험과 약하게만 연관된다**(순위상관 −0.12, ㉚) — 강한 근거는 아니다.

자세한 수치·신뢰구간·방법·한계는 사이트 **'모형 근거' 탭**과 CHANGELOG ⑰~㉚에 전부 있다.

---

## 저장소 구성

| 구분 | 파일 | 역할 |
|---|---|---|
| 전처리 | `preprocess_bc.py`, `preprocess_localdata.py`, `join_datasets.py`, `build_region_master.py`, `franchise_brands.csv` | ABP·LOCALDATA 정제, 업종 매핑, 프랜차이즈 목록(수작업), 조인 → `data/final_joined.csv` |
| 탐색 | `eda.py`, `eda_candidates.py`, `km_screening.py` | 변수 후보 스크리닝, Kaplan-Meier |
| 모형(팀원) | `cox_rsf.py`, `cohort_common.py`, `rsf_eval.py`, `shap_rsf_approx.py` | 코호트 생성·Cox/RSF 평가·RSF 근사 SHAP. 다른 스크립트가 `cox_rsf.py`의 코호트 생성부를 그대로 불러 쓴다 |
| 생존모형 설명·검증 | `explain_common.py`, `explain_cox.py`, `explain_final.py`, `confound_check.py`, `bc_scale_check.py`, `risk_decomp.py`, `moran_residuals.py`, `forward_validation.py`, `group_types.py` | Cox 분해, 교란 점검, 5-fold 비교, 위험 원천 분해, 공간 자기상관, 전향 검증, 상권 유형화 |
| 부가 분석 | `demand_forecast.py`, `customer_analysis.py` | 단기 소비 추이 예측(점포·고객 구성 반영), 연령 구성 변화 분석 |
| 사이트 | `prep_site_data.py`, `build_report.py`, `build_site.py`, `site_surv.py`, `build_site_offline.py`, `site_parts/`, `vendor/` | 사이트용 데이터, 리포트·사이트 HTML 생성(`docs/`가 배포본) |
| 기타 | `spatial_features.py`, `sojin_density.py` | 소진공 상가 기반 경쟁밀도(선택, 원자료 필요, 최종 모형에는 미사용) |

### 산출물 번호 체계 (`output/`, git에는 올리지 않음)

| 접두 | 스크립트 | 내용 |
|---|---|---|
| `4_`, `4e_` | `km_screening.py` | Kaplan-Meier |
| `5_`, `5b_` | `sojin_density.py`, `spatial_features.py` | 경쟁밀도(참고용) |
| `6_` | `explain_cox.py` | 초기 Cox 분해(참고용, 최종은 `11_`) |
| `7_` | `confound_check.py` | BC 연령 구성 교란 점검 |
| `8_` | `bc_scale_check.py` | 그룹·시군구 5-fold 비교, 그룹 특성, 미학습 예측(`_cache/oof_group.npz`) |
| `9_` | `risk_decomp.py` | 위험 원천 분해, 소비-폐업 관계 |
| `10_` | `moran_residuals.py` | 잔차 공간 자기상관(Moran's I) |
| `11_` | `explain_final.py` | **최종 모형(M3L) 요인 분해 + 블록 제거 예측 중요도** |
| `12_` | `group_types.py` | 상권 유형화(5유형) |
| `13_` | `forward_validation.py` | 전향(시간 밖) 검증 |
| `14_` | `demand_forecast.py` | 단기 소비 추이 예측 |
| `15_` | `customer_analysis.py` | 연령 구성 변화(고객 분석) |
| `site_data.json` | `prep_site_data.py` | 사이트용 통합 데이터(위 산출물 전부를 묶음) |

---

## 재현 방법

### 1. 환경
- 주 파이프라인: Python 3.13 + [`requirements-py313.txt`](requirements-py313.txt)
- RSF(`cox_rsf.py`의 RSF 절, `rsf_eval.py`): scikit-survival이 필요해 [`requirements.txt`](requirements.txt)(Python 3.9)의 별도 환경에서 실행. 그 외 스크립트는 scikit-survival 없이 돈다.
- 메모리: `data/final_joined.csv`(약 1.3GB)를 다루므로 여유가 필요하다(열을 지정하지 않고 통째로 읽으면 메모리 오류가 날 수 있다).
- Windows에서 사용자 폴더 경로에 한글이 있으면 joblib 병렬 실행이 실패할 수 있다 → `JOBLIB_TEMP_FOLDER=C:\joblib_tmp` (RSF 실행 시).
- 지도 좌표 변환(`prep_site_data.py`)은 `pyproj`가 필요하다(EPSG:5174 → WGS84).

### 2. 원자료 (저장소에 없음 — `data/`에 직접 둔다)
`ABP_CONTEST_DATA.csv`(대회 제공), 식품_일반음식점.csv, 식품_제과점영업.csv, 식품_휴게음식점.csv, 생활_대규모점포.csv (LOCALDATA, data.go.kr). 소진공 상가 자료는 선택(경쟁밀도 스크립트용, 최종 모형에는 쓰지 않음).

### 3. 실행 순서
```
python preprocess_bc.py           # data/bc_clean.csv
python preprocess_localdata.py    # data/localdata_wide.csv, localdata_clean.csv  (franchise_brands.csv 사용)
python join_datasets.py           # data/final_joined.csv, bc_group_month.csv

python bc_scale_check.py          # output/8_*  (그룹·시군구 5-fold 비교, 그룹 특성, 미학습 예측 저장) — 첫 실행은 수십 분, 이후 캐시(output/_cache)로 빠름
python explain_final.py           # output/11_* (M3L 요인 분해, 블록 제거 예측 중요도)
python risk_decomp.py             # output/9_*  (위험 원천 분해, 소비-폐업 분석)
python moran_residuals.py         # output/10_* (잔차 공간 자기상관)
python group_types.py             # output/12_* (상권 유형)
python forward_validation.py      # output/13_* (전향 검증)
python demand_forecast.py         # output/14_* (단기 소비 추이 예측)
python customer_analysis.py       # output/15_* (고객 연령 구성 변화)

python prep_site_data.py          # output/site_data.json  (+ site_parts/site_data.json 사본 갱신)
python build_site.py              # docs/index.html, site/index.html, report/index.html
```
- 각 스크립트는 `cox_rsf.py`의 코호트 생성부를 그대로 실행해 같은 코호트(635,536점포)를 만든다. 크기가 다르면 명확한 오류로 멈춘다.
- `bc_scale_check.py`는 다른 여러 스크립트(`risk_decomp.py`, `moran_residuals.py` 등)가 쓰는 `output/8_group_features.csv`와 미학습 예측 캐시를 만들므로 **가장 먼저 돌려야 한다.**
- `demand_forecast.py`·`customer_analysis.py`는 `output/8_group_features.csv`와 `output/11_final_group_blocks.csv`를 읽으므로 `bc_scale_check.py`·`explain_final.py` 이후에 돌린다.
- `output/`이 없는 환경에서는 `python build_site_offline.py`가 `site_parts/`의 보관본으로 `docs/index.html`을 다시 만든다(수치는 마지막으로 `prep_site_data.py`를 돌렸을 때 값).
- 선택 스크립트: `confound_check.py`(소진공 경쟁밀도 파일 필요), `explain_cox.py`(초기 SHAP 분해, 최종은 `explain_final.py`).
- 결과 CSV와 `data/`, `output/`은 용량 때문에 git에 올리지 않는다(`.gitignore`).

### 4. 배포
`docs/index.html`이 GitHub Pages(Branch `main`, 폴더 `/docs`)로 서비스된다. 배경 지도 타일은 외부(Esri → OpenStreetMap 순 대체)에서 불러오므로 화면을 보려면 인터넷이 필요하다. 지도 라이브러리(Leaflet)는 `vendor/`를 HTML에 넣어 외부 CDN에 의존하지 않는다.

---

## 사이트 안내

- **지도 탐색**: 시군구 버블 지도(색 = 상대 폐업 위험 / 실제 폐업률 / 상권 유형), 업종 필터, 검색("동탄 서양음식", "한식 위험한 곳" 등 정해진 형식), 지역·업종 선택 시 요인별 영향("왜"), BC 소비 구성, **상권 유형 카드**, **내 점포 진단 계산기**(개업 연수·프랜차이즈 여부를 넣으면 상대 위험을 즉시 계산)
- **상권 분석**: 시군구 프로필, 두 지역 비교, 위험·안전 순위, 상권 유형 표
- **모형 근거**: 요인 중요도, 검증 결과(5-fold), **전향 검증 카드**, 시사점, 데이터와 한계
- 화면의 설명 문장은 규칙으로 만들며 LLM을 쓰지 않는다. 숫자는 모두 분석 산출물에서 온다.
- `demand_forecast.py`·`customer_analysis.py`(단기 소비 추이 예측, 고객 연령 구성 변화)는 아직 사이트에 반영하지 않았다 — 결과는 CHANGELOG ㉙·㉚과 `output/14_*`·`15_*`에 있다.

---

## 알려진 한계 / 남은 일

- RSF(`cox_rsf.py`/`rsf_eval.py`)에 최종 모형(M3L)의 자기·이웃 폐업 이력 변수가 아직 반영되지 않았다 — 비선형 비교는 M3 기준.
- BC 원자료의 연령(AGE_CD 1~6)·성별 코드 공식 정의가 저장소에 없어 오름차순 가정(다른 스크립트와 동일 가정)으로 쓴다.
- 인천 서구·중구 입지(중심지 거리) 이상치 원인 미확인, 프랜차이즈 목록(수작업)의 민감도 분석 미실시.
- 소비 추이·고객 분석(`demand_forecast.py`·`customer_analysis.py`)은 2026년 1~6월 6개월뿐이라 계절성을 다루지 못한다 — "단기 추이"로만 해석한다.
- 전체 목록은 [CHANGELOG.md](CHANGELOG.md) 맨 아래 "아직 정리 안 된 것"과 [프로젝트 계획](BC카드_공모전_프로젝트_계획.md)의 "남은 일" 참고.

---

## 협업 메모
- 공용 저장소는 `ahnseo0000/BC_card`. 푸시 전에 `git pull --no-rebase origin main`으로 팀원 변경을 먼저 합친다(CHANGELOG.md를 동시에 고치면 충돌한다 — 양쪽 섹션을 모두 남기고 해결).
- 큰 파일(CSV)은 올리지 않는다. 산출물이 필요하면 위 실행 순서로 각자 만든다.
