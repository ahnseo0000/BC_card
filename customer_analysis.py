# -*- coding: utf-8 -*-
"""
간단한 고객 분석 — 시군구×업종 그룹의 BC카드 고객 연령 구성이 상반기 안에서 어떻게 움직였는지 보고, 그게 위험·업종과 관련 있는지 본다.
지금까지 사이트는 "이 그룹의 연령 구성이 이렇다"는 서술(1개 시점 막대그래프)만 보여줬다. 여기서는 **1~3월 대비 4~6월의 구성 변화**를 측정해
"구성이 바뀌는 중인 그룹"을 찾고 3분위로 나눠 업종·위험과 연관을 본다(연관이며 인과가 아니다).

[연령 코드 방향에 대한 주의] AGE_CD(1~6) 공식 명세는 저장소에 없다. 코드가 오름차순 연령대라는 가정(다른 스크립트의 age_mean_dev와 동일 가정)
하에 "평균 연령 코드"를 하나의 지표로 쓴다. 코드 정의가 확인되면 라벨만 바꾸면 된다(README 미결 항목 참고).
산출물: output/15_customer_trend_group.csv, output/15_customer_trend_summary.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from explain_common import load_cohort

ns, log = load_cohort()
JOIN_KEY, bc = ns["JOIN_KEY"], ns["bc"]
H1, H2 = [202601, 202602, 202603], [202604, 202605, 202606]
ac = bc[bc["AGE_CD"] != "x"].copy()
ac["code"] = ac["AGE_CD"].astype(int)


def mean_code(months):
    d = ac[ac["STRD_YYMM"].isin(months)]
    t = d.groupby(JOIN_KEY + ["code"])["amt"].sum().unstack("code", fill_value=0)
    sh = t.div(t.sum(axis=1), axis=0)
    return (sh * sh.columns.to_numpy()).sum(axis=1), t.sum(axis=1)


mc1, n1 = mean_code(H1)
mc2, n2 = mean_code(H2)
g = pd.DataFrame({"연령코드평균_상반(1~3월)": mc1, "연령코드평균_하반(4~6월)": mc2, "n1": n1, "n2": n2}).dropna()
g = g[(g["n1"] >= 300) & (g["n2"] >= 300)]                 # 표본 작은 그룹은 구성비가 불안정해 변화량 해석이 어렵다
g["변화"] = g["연령코드평균_하반(4~6월)"] - g["연령코드평균_상반(1~3월)"]
print(f"대상 그룹 {len(g):,}개(상·하반기 모두 표본 300 이상)", flush=True)

g["구분"] = pd.qcut(g["변화"], 3, labels=["연령 코드 하락(젊은 층 비중 증가 쪽)", "변화 적음", "연령 코드 상승(고령 비중 증가 쪽)"])
biz = g.index.get_level_values("bc_업종")
g["업종"] = biz

risk = pd.read_csv("output/11_final_group_blocks.csv", encoding="utf-8-sig").set_index(JOIN_KEY)[["상대위험_배수", "obs_rate", "n"]]
gm = g.join(risk, how="left")

print("\n=== 업종별 평균 연령 코드 변화(상반기 → 하반기) ===")
by_biz = gm.groupby("업종").agg(평균변화=("변화", "mean"), 그룹수=("변화", "size")).sort_values("평균변화")
print(by_biz.round(4).to_string(), flush=True)

print("\n=== 연령 구성 변화 구간별 위험 배수·실제 폐업률(가중평균) ===")
miss = gm["상대위험_배수"].isna().sum()
if miss:
    print(f"  (참고: 최종 모형 그룹 표와 매칭 안 된 {miss}개 그룹은 이 요약에서 제외)")
gm_r = gm.dropna(subset=["상대위험_배수", "obs_rate"])
by_seg = gm_r.groupby("구분", observed=True).apply(
    lambda d: pd.Series({"그룹수": len(d), "평균_상대위험_배수": np.average(d["상대위험_배수"], weights=d["n"].fillna(1)),
                         "평균_실제폐업률(%)": np.average(d["obs_rate"], weights=d["n"].fillna(1)) * 100}), include_groups=False)
print(by_seg.round(3).to_string(), flush=True)

rho = spearmanr(gm_r["변화"], gm_r["상대위험_배수"]).statistic
print(f"\n연령 코드 변화량과 위험 배수의 순위상관: {rho:.3f} (참고용, 인과 아님)", flush=True)

movers = gm.reindex(gm["변화"].abs().sort_values(ascending=False).index).head(10)
print("\n=== 상반기 대비 연령 구성이 가장 크게 바뀐 그룹 10 ===")
print(movers[["업종", "연령코드평균_상반(1~3월)", "연령코드평균_하반(4~6월)", "변화", "상대위험_배수"]].round(3).to_string(), flush=True)

gm.reset_index().round(4).to_csv("output/15_customer_trend_group.csv", index=False, encoding="utf-8-sig")
by_seg.round(4).to_csv("output/15_customer_trend_summary.csv", encoding="utf-8-sig")
print("\n완료", flush=True)
