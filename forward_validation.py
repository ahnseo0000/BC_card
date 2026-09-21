# -*- coding: utf-8 -*-
"""
전향(시간 밖) 검증: 1~6월 코호트로 학습한 모형이 **이후 실제로 일어난 폐업**을 맞히는가.

  학습: 2026-01-01 영업 중 코호트(추적 ~06-30). 그룹 5-fold — 각 fold의 학습 그룹으로 적합한 모형이 그 fold의 **학습에 없던 그룹**의 점포를 예측한다
        (bc_scale_check.py와 같은 분할·같은 적합 캐시를 재사용).
  평가: 2026-06-30에 영업 중인 점포를 2026-09-16까지 78일 추적(LOCALDATA 폐업 기록의 신뢰 구간 끝, cox_rsf.py의 DATA_END).
        특성은 6월 30일 시점으로 다시 계산한다(영업연수, 그룹·시군구·이웃 직전 1년 폐업 이력). BC는 1~6월 집계라 6월 30일에 이미 알려진 정보다 -> 누수 없음.
  => 시간(1~6월 -> 7~9월)과 그룹(학습 그룹 -> 미학습 그룹)이 모두 학습과 분리된 검증.

비교 기준(참고): 지속성 = 그룹의 1~6월 실제 폐업률을 그대로 7~9월 예측으로 쓰기, 직전 1년 폐업률만 쓰기.
지표: 점포 단위 C-index, 그룹 평균 위험점수 vs 실제 폐업률 Spearman(전체·같은 시군구 내, 시군구 cluster bootstrap), 위험 십분위별 실제 폐업률·포착 곡선.
산출물: output/13_forward_summary.csv, 13_forward_increment.csv, 13_forward_decile.csv
"""
import hashlib

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from explain_common import CACHE_DIR, load_cohort, model_cols

B, MIN_N, K = 300, 100, 5
ns, log = load_cohort()
full = ns["full"].reset_index(drop=True)
df = ns["df"]
JOIN_KEY, REGION_KEY, TOP = ns["JOIN_KEY"], ns["REGION_KEY"], ns["TOP"]
LAG3 = ["reg_hist", "nb_reg_hist", "nb_grp_hist"]
TEST_LM, TEST_END = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-09-16")


def alive(d, t):
    return (d["인허가일자"] <= t) & (d["폐업일자"].isna() | (d["폐업일자"] > t))


# ---------- 학습 프레임(1/1 코호트)에 이웃 이력 붙이기: 8_group_features.csv(bc_scale_check.py 산출물) ----------
feat = pd.read_csv("output/8_group_features.csv", encoding="utf-8-sig", usecols=JOIN_KEY + LAG3)
n0 = len(full)
full = full.merge(feat, on=JOIN_KEY, how="left", validate="m:1")
assert len(full) == n0 and not full[LAG3].isna().any().any()

# ---------- 평가 코호트(6/30 영업 중, 9/16까지) ----------
cfg = dict(landmark=TEST_LM, end=TEST_END, months=[202601, 202602, 202603, 202604, 202605, 202606])
cfg["horizon"] = (cfg["end"] - cfg["landmark"]).days
test = ns["build_landmark"]("F", cfg)                       # cox_rsf.py와 같은 정의: 영업연수·BC 공변량·그룹 직전 폐업률을 6/30 시점으로
for b in ns["BIZ_LIST"][1:]:
    test[f"biz_{b}"] = (test["bc_업종"] == b).astype(int)
test = test.dropna(subset=ns["COV_GROUP"] + ns["STORE_CORE"])
groups, regions = ns["groups"], ns["regions"]
n1 = len(test)
test = test.merge(groups, on=JOIN_KEY, how="inner").merge(regions, on=REGION_KEY, how="inner")
print(f"평가 코호트: 6/30 영업 중 {n1:,}점포 -> 학습에 있던 그룹만 {len(test):,}점포 (폐업 {int(test['event'].sum()):,}건, 78일 {test['event'].mean() * 100:.2f}%)", flush=True)
print(f"참고: 학습 코호트 180일 폐업률 {full['event'].mean() * 100:.2f}%", flush=True)

# 6/30 시점의 자기·이웃 폐업 이력(bc_scale_check.py와 같은 정의, 기준일만 6/30)
xy = pd.read_csv("data/final_joined.csv", encoding="utf-8-sig", usecols=REGION_KEY + ["좌표정보(X)", "좌표정보(Y)", "좌표의심"], low_memory=False)
xy = xy[xy["좌표정보(X)"].notna() & xy["좌표정보(Y)"].notna() & (xy["좌표의심"].fillna(0) == 0)]
cent = xy.groupby(REGION_KEY)[["좌표정보(X)", "좌표정보(Y)"]].median().reset_index()
del xy
yr = pd.Timedelta(days=365)
n_ago = df[alive(df, TEST_LM - yr)].groupby(REGION_KEY).size().rename("n_ago")
cl = df[(df["폐업일자"] > TEST_LM - yr) & (df["폐업일자"] <= TEST_LM)].groupby(REGION_KEY).size().rename("cl")
rh = pd.concat([n_ago, cl], axis=1).fillna(0)
rh["reg_hist"] = rh["cl"] / rh["n_ago"].clip(lower=1)
rh = rh.reset_index().merge(cent, on=REGION_KEY, how="inner")
pts = rh[["좌표정보(X)", "좌표정보(Y)"]].to_numpy(float)
_, nn = cKDTree(pts).query(pts, k=6)
pairs = pd.DataFrame({"i": np.repeat(np.arange(len(rh)), 5), "j": nn[:, 1:].ravel()})
pairs = pairs.join(rh[REGION_KEY].reset_index(drop=True), on="i").join(
    rh[REGION_KEY + ["n_ago", "reg_hist"]].reset_index(drop=True).rename(columns={"SIDO_NM": "nS", "CCG_NM": "nC", "n_ago": "nn_ago", "reg_hist": "nb_h"}), on="j")
nb_reg = pairs.groupby(REGION_KEY).apply(lambda d: np.average(d["nb_h"], weights=d["nn_ago"].clip(lower=1)), include_groups=False).rename("nb_reg_hist").reset_index()
gsz = test.groupby(JOIN_KEY).size().rename("nb_w").reset_index()
gt = test.drop_duplicates(JOIN_KEY)[JOIN_KEY + ["g_closure_rate_1y"]].merge(gsz, on=JOIN_KEY).rename(columns={"SIDO_NM": "nS", "CCG_NM": "nC", "g_closure_rate_1y": "nb_g"})
nb_grp = pairs.merge(gt, on=["nS", "nC"], how="inner").groupby(JOIN_KEY).apply(
    lambda d: np.average(d["nb_g"], weights=d["nb_w"].clip(lower=1)), include_groups=False).rename("nb_grp_hist").reset_index()
n2 = len(test)
test = (test.merge(rh[REGION_KEY + ["reg_hist"]], on=REGION_KEY, how="left", validate="m:1").merge(nb_reg, on=REGION_KEY, how="left", validate="m:1")
        .merge(nb_grp, on=JOIN_KEY, how="left", validate="m:1"))
assert len(test) == n2
test["nb_grp_hist"] = test["nb_grp_hist"].fillna(test["nb_reg_hist"])
test = test.dropna(subset=LAG3).reset_index(drop=True)


# ---------- 모형: fold별 학습 계수(캐시 재사용) ----------
def fit_params(name, cs, train):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ident = ("p", name, len(train), tuple(cs), tuple())        # bc_scale_check.py의 fit_params와 같은 키
    path = CACHE_DIR / ("par_" + hashlib.md5(repr(ident).encode("utf-8")).hexdigest()[:12] + ".joblib")
    if path.exists():
        return joblib.load(path)
    keep = list(dict.fromkeys(list(cs) + ["duration", "event"]))
    m = CoxPHFitter().fit(train[keep], duration_col="duration", event_col="event")
    out = {"params": m.params_.copy(), "se": m.standard_errors_.copy()}
    joblib.dump(out, path)
    return out


M2c, M2hc, M3c = model_cols(ns, ns["BASE"]), model_cols(ns, ns["REF"]), model_cols(ns, TOP)
SPEC = {"M2 사업장": ("M2", M2c), "M2h + 그룹 폐업률": ("M2h", M2hc), "M3 + BC 성별·연령": ("M3", M3c), "M3L + 자기·이웃 이력": ("M3L", M3c + LAG3)}
fg = np.zeros(len(groups), dtype=int)
for k, (_, te_i) in enumerate(StratifiedKFold(K, shuffle=True, random_state=42).split(groups, groups["bc_업종"])):
    fg[te_i] = k
fold_test = fg[test["group_id"].to_numpy()]                       # 평가 점포가 속한 그룹의 fold
fold_train = fg[full["group_id"].to_numpy()]
lp = {}
for label, (key, cs) in SPEC.items():
    out = np.full(len(test), np.nan)
    for k in range(K):
        tr = full[fold_train != k]
        b = fit_params(f"g{k}|{key}", cs, tr)["params"].reindex(cs).to_numpy()
        m = fold_test == k
        out[m] = (test.loc[m, cs].to_numpy(float) - tr[cs].to_numpy(float).mean(axis=0)) @ b     # 학습 평균으로 중심화
    assert not np.isnan(out).any()
    lp[label] = out
    print(f"  {label}: 예측 완료", flush=True)

# ---------- 평가 ----------
gid = test["group_id"].to_numpy()
G = int(groups["group_id"].max()) + 1
n_g = np.bincount(gid, minlength=G).astype(float)
ev_g = np.bincount(gid, weights=test["event"].to_numpy(float), minlength=G)
rate_g = ev_g / np.maximum(n_g, 1)
reg_of_g = np.zeros(G, dtype=int)
reg_of_g[gid] = test["region_id"].to_numpy()
keep = n_g >= MIN_N
cnt_reg = np.bincount(reg_of_g[keep], minlength=reg_of_g.max() + 1)
keep2 = keep & (cnt_reg[reg_of_g] >= 2)
R = int(reg_of_g.max()) + 1


def glp(x):
    return np.bincount(gid, weights=x, minlength=G) / np.maximum(n_g, 1)


def dev(x, mask):
    w = np.where(mask, n_g, 0.0)
    m = np.bincount(reg_of_g, weights=w * x, minlength=R) / np.maximum(np.bincount(reg_of_g, weights=w, minlength=R), 1e-9)
    return x - m[reg_of_g]


# 참고 기준: 학습 코호트의 그룹 1~6월 실제 폐업률, 6/30 시점 직전 1년 폐업률
gtr = full.groupby("group_id").agg(n=("event", "size"), ev=("event", "sum"))
persist = np.zeros(G)
persist[gtr.index.to_numpy()] = (gtr["ev"] / gtr["n"]).to_numpy()
last1y = np.zeros(G)
last1y[gid] = test["g_closure_rate_1y"].to_numpy()
base_preds = {"참고: 1~6월 실제 폐업률(지속성)": persist, "참고: 직전 1년 폐업률만": last1y}

rho = lambda a, b: float(spearmanr(a, b).statistic)
rng = np.random.default_rng(42)


def boots(mask):
    idx = np.where(mask)[0]
    by = {r: idx[reg_of_g[idx] == r] for r in np.unique(reg_of_g[idx])}
    ks = list(by)
    return [np.concatenate([by[k] for k in rng.choice(ks, size=len(ks), replace=True)]) for _ in range(B)]


bt_all, bt_in = boots(keep), boots(keep2)
rate_dev = dev(rate_g, keep2)
ev, du = test["event"].to_numpy(), test["duration"].to_numpy()
preds_g = {**{k: glp(v) for k, v in lp.items()}, **base_preds}
rows = []
for name, x in preds_g.items():
    xd = dev(x, keep2)
    v_all = [rho(x[b], rate_g[b]) for b in bt_all]
    v_in = [rho(xd[b], rate_dev[b]) for b in bt_in]
    rows.append({"모형": name, "점포 C-index": round(concordance_index(du, -lp[name], ev), 4) if name in lp else np.nan,
                 "Spearman(전체)": round(rho(x[keep], rate_g[keep]), 3), "95% CI(전체)": f"[{np.percentile(v_all, 2.5):.3f}, {np.percentile(v_all, 97.5):.3f}]",
                 "Spearman(시군구 내)": round(rho(xd[keep2], rate_dev[keep2]), 3), "95% CI(시군구 내)": f"[{np.percentile(v_in, 2.5):.3f}, {np.percentile(v_in, 97.5):.3f}]"})
summ = pd.DataFrame(rows)
summ.to_csv("output/13_forward_summary.csv", index=False, encoding="utf-8-sig")
print(f"\n=== 전향 검증: 6/30 영업 중 점포를 9/16까지(78일) 추적, 평가 그룹 {int(keep.sum())}개 / 시군구 내 {int(keep2.sum())}개 ===")
print(summ.to_string(index=False), flush=True)

pairs_inc = [("M2h + 그룹 폐업률", "M2 사업장", "직전 1년 폐업률(그룹)"), ("M3 + BC 성별·연령", "M2h + 그룹 폐업률", "BC 성별·연령"),
             ("M3L + 자기·이웃 이력", "M3 + BC 성별·연령", "자기·이웃 시군구 폐업 이력"), ("M3L + 자기·이웃 이력", "참고: 1~6월 실제 폐업률(지속성)", "지속성 기준 대비")]
inc = []
for a, c, what in pairs_inc:
    for kind, mask, bts, dv in (("전체", keep, bt_all, False), ("시군구 내", keep2, bt_in, True)):
        xa, xc = (dev(preds_g[a], keep2), dev(preds_g[c], keep2)) if dv else (preds_g[a], preds_g[c])
        y = rate_dev if dv else rate_g
        d0 = rho(xa[mask], y[mask]) - rho(xc[mask], y[mask])
        v = [rho(xa[b], y[b]) - rho(xc[b], y[b]) for b in bts]
        lo, hi = np.percentile(v, [2.5, 97.5])
        inc.append({"블록": what, "비교": f"{a} − {c}", "기준": kind, "ΔSpearman": round(d0, 3), "95% CI": f"[{lo:.3f}, {hi:.3f}]", "0 제외": "예" if (lo > 0 or hi < 0) else "아니오"})
inc = pd.DataFrame(inc)
inc.to_csv("output/13_forward_increment.csv", index=False, encoding="utf-8-sig")
print("\n=== 모형 간 증분(전향) ===")
print(inc.to_string(index=False), flush=True)

# 위험 십분위(점포 단위, 최종 모형): 예측이 높은 점포일수록 실제로 더 폐업했는가 + 포착 곡선
x = lp["M3L + 자기·이웃 이력"]
q = pd.qcut(x, 10, labels=False, duplicates="drop")
dec = pd.DataFrame({"십분위(10=위험 최고)": q + 1, "event": ev}).groupby("십분위(10=위험 최고)").agg(점포=("event", "size"), 폐업=("event", "sum"), 실제_폐업률_pct=("event", lambda s: s.mean() * 100)).reset_index()
dec["평균 대비"] = (dec["실제_폐업률_pct"] / (ev.mean() * 100)).round(2)
order = np.argsort(-x)
cap = {f"상위 {p}% 점포의 폐업 포착률(%)": round(100 * ev[order[: int(len(x) * p / 100)]].sum() / ev.sum(), 1) for p in (10, 20, 30, 50)}
dec.round(3).to_csv("output/13_forward_decile.csv", index=False, encoding="utf-8-sig")
print("\n=== 위험 십분위별 실제 78일 폐업률 ===")
print(dec.round(3).to_string(index=False))
print("포착 곡선:", cap, flush=True)
print("\n완료", flush=True)
