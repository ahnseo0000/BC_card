# 상권 생존 지도 — 설명 가능한 상권 생존 분석

BC카드 소비데이터 공모전 출품작. **소비는 있는데 왜 못 버티는가**를 점포 단위 폐업 기록(생존분석)과 BC카드 소비 구성으로 설명하고, 결과를 지도 웹사이트로 제공한다.

- **웹사이트**: https://ahnseo0000.github.io/BC_card/ (이 저장소의 `docs/index.html`, GitHub Pages)
- **분석 기록(방법·검증·정정 이력 전부)**: [CHANGELOG.md](CHANGELOG.md)
- **데이터 소스·업종 매핑 정리**: [DATA_MAPPING.md](DATA_MAPPING.md)
- 프로젝트 계획: [BC카드_공모전_프로젝트_계획.md](BC카드_공모전_프로젝트_계획.md)

> 이 프로젝트의 모든 결과는 **통계적 연관**이며 인과나 정책 효과가 아니다.

## 한눈에

| 항목 | 내용 |
|---|---|
| 분석 대상 | 2026-01-01에 영업 중이던 점포 635,536개(한식계열·일식회집·중국음식·서양음식·스낵·제과점·편의점 7개 업종)를 180일 추적, 폐업 25,329건(3.99%) |
| 결과 변수 | LOCALDATA 인허가 4종의 폐업일자 (점포 단위) |
| 설명 변수 | 점포(영업연수·프랜차이즈·입지·규모 등), 지역 폐업 이력(시군구·이웃 시군구), BC카드 시군구×업종 성별·연령 구성(2026-01~06) |
| 최종 모형 | Cox 비례위험 모형 **M3L** — 선형이라 요인별 기여를 정확히 분해할 수 있다 |
| 검증 | 시군구×업종 그룹 5-fold, 시군구 5-fold, 전향(시간 밖) 검증(6/30 영업 점포를 9/16까지 78일 추적) |

**주요 결과 요약**
- 그룹 간 폐업률 차이는 업종보다 **지역(시군구)** 이 훨씬 크게 가른다.
- 점포 단위 판별에는 영업연수·프랜차이즈·점포 특성이, 지역 간 순위에는 자기·이웃 지역의 최근 폐업 흐름이 기여한다.
- BC카드 성별·연령 구성은 어떤 유형의 지역인지 알려 주는 맥락 정보로 활용한다. 소비 규모·객단가 등 추가 지표는 검증했으나 예측 순위의 추가 개선은 확인되지 않았다(CHANGELOG ⑲, 사이트 '모형 근거' 탭).
- 전향 검증에서 점포 단위 C-index 0.627, 예측 위험 최상위 10% 점포의 실제 폐업률은 평균의 2.0배(최하위 10%는 0.47배).
- 자세한 수치·신뢰구간·한계는 사이트 '모형 근거' 탭과 CHANGELOG ⑱~㉘.

## 저장소 구성

| 구분 | 파일 | 역할 |
|---|---|---|
| 전처리 | `preprocess_bc.py`, `preprocess_localdata.py`, `join_datasets.py`, `build_region_master.py`, `franchise_brands.csv` | ABP·LOCALDATA 정제, 업종 매핑, 프랜차이즈 목록(수작업), 조인 → `data/final_joined.csv` |
| 탐색 | `eda.py`, `eda_candidates.py`, `km_screening.py` | 변수 후보 스크리닝, Kaplan-Meier |
| 모형(팀원) | `cox_rsf.py`, `cohort_common.py`, `rsf_eval.py`, `shap_rsf_approx.py` | 코호트 생성·Cox/RSF 평가·RSF 근사 SHAP. 다른 스크립트가 `cox_rsf.py`의 코호트 생성부를 그대로 불러 쓴다 |
| 설명·검증 | `explain_common.py`, `explain_cox.py`, `explain_final.py`, `confound_check.py`, `bc_scale_check.py`, `risk_decomp.py`, `moran_residuals.py`, `forward_validation.py`, `group_types.py` | Cox 분해, 교란 점검, 5-fold 비교, 위험 원천 분해, 공간 자기상관, 전향 검증, 상권 유형화 |
| 사이트 | `prep_site_data.py`, `build_report.py`, `build_site.py`, `site_surv.py`, `build_site_offline.py`, `site_parts/`, `vendor/` | 사이트용 데이터, 리포트·사이트 HTML 생성(`docs/`가 배포본) |
| 기타 | `spatial_features.py`, `sojin_density.py` | 소진공 상가 기반 경쟁밀도(선택, 원자료 필요) |

## 재현 방법

### 1. 환경
- 주 파이프라인: Python 3.13 + [`requirements-py313.txt`](requirements-py313.txt)
- RSF(`cox_rsf.py`의 RSF 절, `rsf_eval.py`): scikit-survival이 필요해 [`requirements.txt`](requirements.txt)(Python 3.9)의 별도 환경에서 실행. 그 외 스크립트는 scikit-survival 없이 돈다.
- 메모리: `data/final_joined.csv`(약 1.3GB)를 다루므로 여유가 필요하다(열을 지정하지 않고 통째로 읽으면 메모리 오류가 날 수 있다).
- Windows에서 사용자 폴더 경로에 한글이 있으면 joblib 병렬 실행이 실패할 수 있다 → `JOBLIB_TEMP_FOLDER=C:\joblib_tmp` (RSF 실행 시).

### 2. 원자료 (저장소에 없음 — `data/`에 직접 둔다)
`ABP_CONTEST_DATA.csv`(대회 제공), 식품_일반음식점.csv, 식품_제과점영업.csv, 식품_휴게음식점.csv, 생활_대규모점포.csv (LOCALDATA, data.go.kr). 소진공 상가 자료는 선택(경쟁밀도 스크립트용, 최종 모형에는 쓰지 않음).

### 3. 실행 순서
```
python preprocess_bc.py           # data/bc_clean.csv
python preprocess_localdata.py    # data/localdata_wide.csv, localdata_clean.csv  (franchise_brands.csv 사용)
python join_datasets.py           # data/final_joined.csv, bc_group_month.csv
python bc_scale_check.py          # output/8_* (그룹·시군구 5-fold 비교, 그룹 특성, 미학습 예측 저장) — 첫 실행은 수십 분, 이후 캐시(output/_cache)로 빠름
python explain_final.py           # output/11_* (M3L 요인 분해, 블록 제거 예측 중요도)
python risk_decomp.py             # output/9_*  (위험 원천 분해, 소비-폐업 분석)
python moran_residuals.py         # output/10_* (잔차 공간 자기상관)
python group_types.py             # output/12_* (상권 유형)
python forward_validation.py      # output/13_* (전향 검증)
python prep_site_data.py          # output/site_data.json (+ site_parts/site_data.json)
python build_site.py              # docs/index.html, site/index.html, report/index.html
```
- 각 스크립트는 `cox_rsf.py`의 코호트 생성부를 실행해 같은 코호트(635,536점포)를 만든다. 크기가 다르면 명확한 오류로 멈춘다.
- `output/`이 없는 환경에서는 `python build_site_offline.py`가 `site_parts/`의 보관본으로 `docs/index.html`을 다시 만든다.
- 선택 스크립트: `confound_check.py`(소진공 경쟁밀도 파일 필요), `explain_cox.py`(초기 SHAP 분해).
- 결과 CSV와 `data/`, `output/`은 용량 때문에 git에 올리지 않는다(`.gitignore`).

### 4. 배포
`docs/index.html`이 GitHub Pages(Branch `main`, 폴더 `/docs`)로 서비스된다. 배경 지도 타일은 외부(Esri → OpenStreetMap 순 대체)에서 불러오므로 화면을 보려면 인터넷이 필요하다. 지도 라이브러리(Leaflet)는 `vendor/`를 HTML에 넣어 외부 CDN에 의존하지 않는다.

## 사이트 안내

- **지도 탐색**: 시군구 버블 지도(색 = 상대 폐업 위험 / 실제 폐업률 / 상권 유형), 업종 필터, 검색("동탄 서양음식", "한식 위험한 곳" 등 정해진 형식), 지역·업종 선택 시 요인별 영향("왜"), BC 소비 구성, 내 점포 진단 계산기
- **상권 분석**: 시군구 프로필, 두 지역 비교, 위험·안전 순위, 상권 유형 표
- **모형 근거**: 요인 중요도, 검증 결과, 전향 검증, 시사점, 데이터와 한계
- 화면의 설명 문장은 규칙으로 만들며 LLM을 쓰지 않는다. 숫자는 모두 분석 산출물에서 온다.

## 협업 메모
- 공용 저장소는 `ahnseo0000/BC_card`. 푸시 전에 `git pull --no-rebase origin main`으로 팀원 변경을 먼저 합친다(CHANGELOG.md를 동시에 고치면 충돌한다 — 양쪽 섹션을 모두 남기고 해결).
- 큰 파일(CSV)은 올리지 않는다. 산출물이 필요하면 위 실행 순서로 각자 만든다.
