# -*- coding: utf-8 -*-
"""
상권 유형화: M3L 요인 분해(output/11_final_group_blocks.csv)의 '요인 프로필'이 비슷한 시군구×업종 그룹을 묶는다.

- 입력 특성: 그룹의 요인별 log 배수 4개 — 영업연수, 프랜차이즈, 사업장 특성, 자기·이웃 폐업 이력.
  (BC 연령·성별은 지역 유형 신호라 유형이 지역 유형의 복사본이 되므로 넣지 않는다. 업종 기본 위험은 업종 자체이므로 뺀다.)
- 표준화하지 않는다: 모두 같은 단위(log 배수)라 크기의 의미를 유지한다.
- 군집 학습은 점포 100개 이상인 그룹만 쓰고(잡음 방지), 나머지 그룹은 가장 가까운 중심에 배정한다.
- k는 4~7 중 실루엣과 해석 가능성으로 고른다(--k로 고정 가능). 유형 이름은 중심의 지배 요인에서 규칙으로 만든다.
산출물: output/12_group_types.csv (그룹별 유형), output/12_type_summary.csv (유형별 요약)
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=None)
args = ap.parse_args()

g = pd.read_csv("output/11_final_group_blocks.csv", encoding="utf-8-sig")
FEATS = {"영업연수": "x_영업연수", "프랜차이즈": "x_프랜차이즈", "사업장 특성": "x_사업장 확장(다중이용·크기·좌표결측·전화)", "폐업 이력": "x_자기·이웃 시군구 폐업 이력"}
X = np.log(g[list(FEATS.values())].to_numpy())
fit = (g["n"] >= 100).to_numpy()

scores = {}
for k in range(4, 8):
    km = KMeans(k, n_init=20, random_state=42).fit(X[fit])
    scores[k] = silhouette_score(X[fit], km.labels_)
print("실루엣:", {k: round(v, 3) for k, v in scores.items()})
K = args.k or max(scores, key=scores.get)
km = KMeans(K, n_init=50, random_state=42).fit(X[fit])
g["type"] = km.predict(X)
cent = pd.DataFrame(np.exp(km.cluster_centers_), columns=list(FEATS))      # 중심의 요인 배수
print(f"\nk={K}, 요인 배수 중심(exp):")
print(cent.round(2).to_string())

# 유형 이름: 중심에서 1.0(평균)과 가장 크게 다른 요인부터 규칙으로 만든다(최대 3개)
PHRASE = {"영업연수": ("신생 점포 많음", "오랜 점포 많음"), "프랜차이즈": ("개인 점포 중심", "프랜차이즈 많음"),
          "폐업 이력": ("폐업 잦은 지역", "폐업 드문 지역"), "사업장 특성": ("대형·다중이용 점포", "소형 점포")}


def name_of(c):
    dev = sorted(((abs(np.log(v)), k, v) for k, v in c.items() if abs(np.log(v)) >= np.log(1.05)), reverse=True)[:3]
    parts = [PHRASE[k][0] if v > 1 else PHRASE[k][1] for _, k, v in dev]
    return " · ".join(parts) if parts else "평균형"


g["mult"] = g["상대위험_배수"]
rows = []
for t in range(K):
    m = g[g["type"] == t]
    c = cent.loc[t]
    w = m["n"]
    rows.append({"type": t, "이름": name_of(c), "그룹 수": len(m), "점포 수": int(w.sum()), "점포 비중(%)": round(100 * w.sum() / g["n"].sum(), 1),
                 "평균 위험 배수": round(float(np.average(m["mult"], weights=w)), 2), "실제 폐업률(%)": round(100 * m["events"].sum() / w.sum(), 2),
                 "폐업 비중(%)": round(100 * m["events"].sum() / g["events"].sum(), 1),
                 **{f"{k} 배수": round(float(c[k]), 2) for k in FEATS}, "대표 업종": ", ".join(m.groupby("bc_업종")["n"].sum().sort_values(ascending=False).index[:3])})
summ = pd.DataFrame(rows).sort_values("평균 위험 배수", ascending=False).reset_index(drop=True)
summ["rank"] = np.arange(len(summ))            # 0 = 위험이 가장 높은 유형(화면에서 A, B, C ...)
print("\n=== 유형 요약(위험 순) ===")
print(summ.to_string(index=False))
big = g[g["n"] >= 300]
for t in summ["type"]:
    m = big[big["type"] == t].sort_values("mult", ascending=False)
    print(f"\n[{t}] {summ.loc[summ['type'] == t, '이름'].iloc[0]} — 대표 그룹")
    print("  " + " / ".join(f"{r.SIDO_NM[:2]} {r.CCG_NM} {r.bc_업종}(×{r.mult:.2f})" for r in m.head(4).itertuples()))
g["rank"] = g["type"].map(dict(zip(summ["type"], summ["rank"])))
g[["SIDO_NM", "CCG_NM", "bc_업종", "type", "rank"]].to_csv("output/12_group_types.csv", index=False, encoding="utf-8-sig")
summ.to_csv("output/12_type_summary.csv", index=False, encoding="utf-8-sig")
