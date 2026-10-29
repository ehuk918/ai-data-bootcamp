# %% [markdown]
# # 위키백과 여행지 페이지뷰 분석
# - 데이터: 7개 여행지 × 언어(ko/en), 2015-07 ~ 2026-09 월별 조회수
# - VSCode에서 각 `# %%` 셀을 Run Cell 로 하나씩 실행하거나, 터미널에서 `python analysis.py` 로 전체 실행

# %% 0. 라이브러리 & 한글 폰트
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"      # macOS 한글 폰트
plt.rcParams["axes.unicode_minus"] = False       # 마이너스 기호 깨짐 방지

BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
OUT = BASE / "output"
OUT.mkdir(exist_ok=True)

# %% 1. 데이터 불러오기 & 기본 확인
df = pd.read_csv(BASE / "wiki_pageviews_monthly.csv", encoding="utf-8-sig")  # BOM 제거
df["month"] = pd.to_datetime(df["month"])
df["year"] = df["month"].dt.year
df["mon"] = df["month"].dt.month

print(df.head())
print(df.info())
print("결측치:\n", df.isna().sum())
print("기간:", df["month"].min().date(), "~", df["month"].max().date())
print("여행지:", df["place"].unique())

# %% 2. 여행지 × 언어별 총 조회수
total = df.pivot_table(index="place", columns="lang", values="views", aggfunc="sum")
total["합계"] = total.sum(axis=1)
total = total.sort_values("합계", ascending=False)
print(total)

total[["ko", "en"]].plot(kind="barh", figsize=(9, 5), title="여행지별 총 조회수 (2015-07 ~ 2026-09)")
plt.xlabel("조회수")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(OUT / "01_total_views.png", dpi=150)
plt.show()

# %% 3. 월별 추이 (한국어 / 영어 따로)
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, lang in zip(axes, ["ko", "en"]):
    wide = df[df["lang"] == lang].pivot(index="month", columns="place", values="views")
    wide.plot(ax=ax)
    ax.set_title(f"월별 조회수 추이 ({lang})")
    ax.set_ylabel("조회수")
    ax.axvspan(pd.Timestamp("2020-03"), pd.Timestamp("2022-06"), color="gray", alpha=0.15, label="코로나 기간")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(OUT / "02_monthly_trend.png", dpi=150)
plt.show()

# %% 4. 연도별 합계 & 전년 대비 증감률 (1~12월이 모두 있는 연도만)
full_years = df.groupby("year")["mon"].nunique()
full_years = full_years[full_years == 12].index

yearly = (df[df["year"].isin(full_years)]
          .pivot_table(index="year", columns=["lang", "place"], values="views", aggfunc="sum"))
print("한국어 연도별 조회수\n", yearly["ko"])
print("한국어 전년 대비 증감률(%)\n", (yearly["ko"].pct_change() * 100).round(1))

yearly["ko"].plot(marker="o", figsize=(10, 5), title="연도별 조회수 (ko)")
plt.ylabel("조회수")
plt.tight_layout()
plt.savefig(OUT / "03_yearly_ko.png", dpi=150)
plt.show()

# %% 5. 계절성: 몇 월에 가장 많이 찾아볼까? (한국어)
# 여행지마다 규모가 다르므로, 여행지별 평균을 1로 맞춘 지수로 비교
ko = df[df["lang"] == "ko"]
season = ko.pivot_table(index="place", columns="mon", values="views", aggfunc="mean")
season_idx = season.div(season.mean(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(11, 5))
im = ax.imshow(season_idx, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(12), [f"{m}월" for m in range(1, 13)])
ax.set_yticks(range(len(season_idx)), season_idx.index)
for i in range(season_idx.shape[0]):
    for j in range(season_idx.shape[1]):
        ax.text(j, i, f"{season_idx.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, label="평균 대비 배수")
ax.set_title("월별 관심도 지수 (ko, 1.0 = 평균)")
plt.tight_layout()
plt.savefig(OUT / "04_seasonality_ko.png", dpi=150)
plt.show()

print("여행지별 최고 관심 월:\n", season_idx.idxmax(axis=1))

# %% 6. 한국인 관심 비중: ko / (ko + en)
# 같은 여행지라도 한국인이 특히 많이 찾아보는 곳은 어디일까?
wide_lang = df.pivot_table(index=["place", "month"], columns="lang", values="views").reset_index()
wide_lang["ko_share"] = wide_lang["ko"] / (wide_lang["ko"] + wide_lang["en"])

share_total = (wide_lang.groupby("place")[["ko", "en"]].sum()
               .assign(ko_share=lambda d: d["ko"] / (d["ko"] + d["en"]))
               .sort_values("ko_share", ascending=False))
print(share_total)

share_trend = wide_lang.pivot(index="month", columns="place", values="ko_share")
share_trend.rolling(12).mean().plot(figsize=(12, 5), title="한국어 조회 비중 추이 (12개월 이동평균)")
plt.ylabel("ko / (ko + en)")
plt.tight_layout()
plt.savefig(OUT / "05_ko_share.png", dpi=150)
plt.show()

# %% 7. 코로나 영향 & 회복률 (2019년 = 100 기준)
period = {"2019(코로나 전)": [2019], "2020-21(코로나)": [2020, 2021], "2023-25(회복기)": [2023, 2024, 2025]}
rows = []
for name, years in period.items():
    avg = df[df["year"].isin(years)].groupby(["place", "lang"])["views"].mean()
    rows.append(avg.rename(name))
covid = pd.concat(rows, axis=1)
covid_idx = (covid.div(covid["2019(코로나 전)"], axis=0) * 100).round(1)
print("2019년 월평균 = 100 기준 지수\n", covid_idx)

covid_idx.xs("ko", level="lang").plot(kind="bar", figsize=(11, 5), title="코로나 전후 월평균 조회수 지수 (ko, 2019=100)")
plt.axhline(100, color="black", lw=0.8, ls="--")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "06_covid_ko.png", dpi=150)
plt.show()

# %% 8. 여행지 간 상관관계 (ko 월별 조회수)
corr = ko.pivot(index="month", columns="place", values="views").corr().round(2)
print(corr)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, corr.iat[i, j], ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("여행지 간 조회수 상관계수 (ko)")
plt.tight_layout()
plt.savefig(OUT / "07_corr_ko.png", dpi=150)
plt.show()

# %% 9. 결과 요약 저장
results = {
    "총조회수": total,
    "연도별_ko": yearly["ko"],
    "연도별_en": yearly["en"],
    "계절성_ko": season_idx.round(2),
    "한국어비중": share_total,
    "코로나영향": covid_idx,
}
for name, table in results.items():
    table.to_csv(OUT / f"{name}.csv", encoding="utf-8-sig")  # 엑셀에서 한글 안 깨지게
print("저장 완료:", OUT)
