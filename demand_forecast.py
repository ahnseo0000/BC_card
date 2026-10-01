# -*- coding: utf-8 -*-
"""
단기 소비(매출) 추이 예측 — 시군구×업종 그룹의 BC카드 소비액을 1~4월 정보로 5~6월을 예측하고 실제와 비교한다.

[왜 "수요예측"이 아니라 "단기 추이 예측"인가]
  BC 원자료는 2026년 1~6월 6개월뿐이다. 계절성·연 단위 추세를 다루려면 최소 1~2년이 필요하므로, 이 데이터로 할 수 있는 것은
  "최근 수준 + 짧은 추세 + 점포 구성으로 다음 2개월 평균 소비를 맞히는가"까지다. 이 스크립트는 그 범위를 벗어나지 않는다.

[세 가지 방법 비교, 전부 그룹 간 5-fold로 미학습 그룹에 검증]
  persist  : 4월 값을 그대로 5~6월 예측으로 사용(전통적 naive 베이스라인)
  trend    : 1~4월 로그소비의 선형 추세를 5~6월로 연장(점포 구성 정보 없음)
  model    : persist + trend + 점포 구성(프랜차이즈 비중·평균 영업연수·다중이용 비중·그룹 규모) + 업종 더미로 Ridge 회귀
             -> "고객/매장 분석(점포 구성)이 수요 예측에 보탬이 되는가"를 직접 검정한다(모형이 persist보다 나은지 여부로 판정).
  평가: 로그 스케일 R², MAPE(원 단위), 그룹 순위 Spearman. 전부 학습에 쓰지 않은 그룹에 대한 평가(그룹 5-fold, out-of-fold).

[참고] 예측이 맞은 그룹일수록 폐업 위험도 낮은지(수요 추세와 생존의 연결)도 참고로 본다 — 별도 분석이며 인과가 아니다.
산출물: output/14_demand_forecast_group.csv, output/14_demand_forecast_summary.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from explain_common import load_cohort

K = 5
ns, log = load_cohort()
JOIN_KEY, bc = ns["JOIN_KEY"], ns["bc"]
months = sorted(bc["STRD_YYMM"].unique())
assert months == [202601, 202602, 202603, 202604, 202605, 202606], months
TRAIN_M, TEST_M = months[:4], months[4:]

amt = bc.groupby(JOIN_KEY + ["STRD_YYMM"])["amt"].sum().unstack("STRD_YYMM").reindex(columns=months)
n0 = len(amt)
amt = amt[amt.notna().all(axis=1) & (amt > 0).all(axis=1)]           # 6개월 내내 거래가 있던 그룹만(마스킹으로 결측인 소규모 조합 제외)
print(f"대상 그룹 {len(amt):,} / 전체 {n0:,} (6개월 연속 거래 있는 그룹만)", flush=True)
la = np.log(amt)
y = la[TEST_M].mean(axis=1).rename("y_log")                           # 예측 대상: 5~6월 평균 로그소비

# ---- 베이스라인 1: persistence(4월 값) ----
persist = la[TRAIN_M[-1]].rename("persist")

# ---- 베이스라인 2: 1~4월 추세 연장 ----
x = np.arange(len(TRAIN_M))
fit = la[TRAIN_M].apply(lambda row: np.polyfit(x, row.to_numpy(), 1), axis=1, result_type="expand")
fit.columns = ["slope", "intercept"]
test_x = np.arange(len(TRAIN_M), len(months)).mean()
trend = (fit["intercept"] + fit["slope"] * test_x).rename("trend")

# ---- 점포/고객 구성 특성(매장 분석 결과를 수요 예측 입력으로) ----
feat = pd.read_csv("output/8_group_features.csv", encoding="utf-8-sig", usecols=JOIN_KEY + ["ctx_fr", "ctx_age", "ctx_multi", "log_n_grp"]).set_index(JOIN_KEY)
X = pd.concat([persist, fit["slope"], feat.reindex(amt.index)], axis=1).join(y)
X = X.dropna()
print(f"특성 결합 후 {len(X):,}개 그룹", flush=True)
biz = X.index.get_level_values("bc_업종")
dummy = pd.get_dummies(biz, prefix="biz", drop_first=True).set_axis(X.index)
Xall = pd.concat([X.drop(columns="y_log"), dummy], axis=1)
yall = X["y_log"]

fold = np.zeros(len(X), dtype=int)
for k, (_, te) in enumerate(StratifiedKFold(K, shuffle=True, random_state=42).split(Xall, biz)):
    fold[te] = k
oof_model = np.full(len(X), np.nan)
for k in range(K):
    tr, te = fold != k, fold == k
    sc = StandardScaler().fit(Xall[tr])
    reg = Ridge(alpha=3.0).fit(sc.transform(Xall[tr]), yall[tr])
    oof_model[te] = reg.predict(sc.transform(Xall[te]))
oof_model = pd.Series(oof_model, index=X.index, name="model")

preds = {"persist(4월 그대로)": persist.reindex(X.index), "trend(1~4월 추세 연장)": trend.reindex(X.index), "model(+점포·고객 구성, out-of-fold)": oof_model}
rows = []
for name, p in preds.items():
    err = np.exp(p) - np.exp(X["y_log"])
    mape = float((err.abs() / np.exp(X["y_log"])).mean() * 100)
    ss_res = float(((X["y_log"] - p) ** 2).sum())
    ss_tot = float(((X["y_log"] - X["y_log"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    rho = float(spearmanr(p, X["y_log"]).statistic)
    rows.append({"방법": name, "로그스케일 R²": round(r2, 3), "MAPE(%)": round(mape, 1), "그룹 순위 Spearman": round(rho, 3)})
summ = pd.DataFrame(rows)
summ.to_csv("output/14_demand_forecast_summary.csv", index=False, encoding="utf-8-sig")
print("\n=== 5~6월 평균 소비 예측 vs 실제 (그룹 5-fold, 미학습 그룹 기준) ===")
print(summ.to_string(index=False), flush=True)

out = pd.DataFrame({"실제(5~6월 평균, 백만원)": np.exp(X["y_log"]) / 1e6, "persist 예측": np.exp(persist.reindex(X.index)) / 1e6,
                    "trend 예측": np.exp(trend.reindex(X.index)) / 1e6, "model 예측": np.exp(oof_model) / 1e6}).reset_index()
out.round(2).to_csv("output/14_demand_forecast_group.csv", index=False, encoding="utf-8-sig")

# ---- 참고: 수요 추세와 생존(폐업 위험)의 관계 ----
try:
    grp = pd.read_csv("output/11_final_group_blocks.csv", encoding="utf-8-sig").set_index(JOIN_KEY)
    merged = fit["slope"].to_frame("소비추세(1~4월 slope)").join(grp[["상대위험_배수", "obs_rate"]], how="inner")
    rho2 = spearmanr(merged["소비추세(1~4월 slope)"], merged["상대위험_배수"]).statistic
    print(f"\n[참고] 소비 추세(기울기)와 M3L 위험 배수의 순위상관: {rho2:.3f} (음수면 '소비가 느는 그룹일수록 위험이 낮다'는 방향, 참고용이며 인과 아님)", flush=True)
except FileNotFoundError:
    pass
print("\n완료", flush=True)
