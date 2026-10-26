"""Render README figures from bundled evidence: python assets/build_readme_figures.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
NAVY, TEAL, INK, MUTED = "#284967", "#008575", "#182c3c", "#526576"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none", "svg.hashsalt": "quiet-index-readme",
})


def frame(title, subtitle, bottom=0.26):
    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.subplots_adjust(left=0.30, right=0.92, top=0.73, bottom=bottom)
    fig.text(0.045, 0.92, title, fontsize=21, weight="bold")
    fig.text(0.045, 0.85, subtitle, fontsize=12, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e2e8ed", linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
    ax.axvline(0, color="#9aaab6", linewidth=1)
    return fig, ax


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=180)
    svg = OUT / f"{stem}.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


company = pd.read_csv(ROOT / "evidence/company_inference.csv")
company = company[(company.run_id == "calibration_main") & (company.block_label == "all")]
pairs = [("C", "C+R"), ("C+R", "B")]
rows = []
for baseline, extension in pairs:
    match = company[(company.baseline_model_id == baseline) &
                    (company.extended_model_id == extension)]
    assert len(match) == 1 and int(match.iloc[0].quarters) == 26
    rows.append(match.iloc[0])

fig, ax = frame("Recent risk delivered the larger benchmark improvement",
                "Mean quarterly reduction in log-volatility MSE · 26 quarters · 2020 Q1–2026 Q2")
for y, row, color in zip([1, 0], rows, [NAVY, TEAL]):
    value = row.mean_percentage_gain
    ax.barh(y, value, height=0.36, color=color, alpha=0.9)
    ax.errorbar(value, y, xerr=[[value-row.conditional_low], [row.conditional_high-value]],
                fmt="none", ecolor=INK, capsize=6, linewidth=1.5)
    ax.text(value, y+0.29, f"{value:.3f}%", ha="center", weight="bold", fontsize=15)
ax.set_yticks([1, 0], ["Add recent risk\nvs. conventional benchmark",
                       "Then add liquidity\nvs. benchmark with recent risk"])
ax.set_xlim(0, 15)
ax.set_ylim(-0.55, 1.65)
ax.set_xlabel("Lower forecast error →", labelpad=12)
fig.text(0.045, 0.09,
         "Bars use different reference models; percentages are not additive. Whiskers: conditional 95% intervals.",
         fontsize=11, color=MUTED)
fig.text(0.045, 0.045,
         "Recent risk changes both horizon and functional form. Intervals hold measured features and fitted forecasts fixed.",
         fontsize=10.5, color=MUTED)
save(fig, "benchmark-improvements")

groups = pd.read_csv(ROOT / "evidence/group_comparisons.csv")
groups = groups[(groups.group == "Semiconductors & Semiconductor Equipment") &
                (groups.block == "confirmation") & (groups.family == "industry_group")]
specs = [
    ("original_504_semiconductor", "Initial factor benchmark\nSector controls", NAVY),
    ("revised_semiconductor_main", "With recent risk + liquidity\nSector controls", TEAL),
    ("original_504_semiconductor_industry", "Initial factor benchmark\nIndustry controls", NAVY),
    ("revised_semiconductor_industry_group", "With recent risk + liquidity\nIndustry controls", TEAL),
]
fig, ax = frame("Semiconductor-company gains reversed with finer controls",
                "Signed SMH exposure added to each benchmark · 19 companies · 6 quarters · 2025–2026 Q2")
for y, (run_id, label, color) in zip([3, 2, 1, 0], specs):
    match = groups[groups.run_id == run_id]
    assert len(match) == 1 and int(match.iloc[0].observations) == 114
    value = match.iloc[0].mean_quarterly_gain_pct
    ax.barh(y, value, height=0.5, color=color, alpha=0.9)
    ax.text(value + (0.25 if value >= 0 else -0.25), y, f"{value:+.3f}%",
            ha="left" if value >= 0 else "right", va="center", weight="bold", fontsize=13)
ax.set_yticks([3, 2, 1, 0], [s[1] for s in specs])
ax.set_xlim(-4, 14)
ax.set_xticks([-4, 0, 4, 8, 12])
ax.set_xlabel("Mean quarterly log-volatility MSE improvement (%)", labelpad=12)
ax.axhline(1.5, color="#e2e8ed", linewidth=1)
fig.text(0.045, 0.09,
         "Positive = lower forecast error; negative = higher error. Historical point estimates, each against its own baseline.",
         fontsize=10.5, color=MUTED)
fig.text(0.045, 0.045,
         "Both models are refitted when controls change. Six reused quarters and multiple groups inspected limit the inference.",
         fontsize=10.5, color=MUTED)
save(fig, "semiconductor-controls")
