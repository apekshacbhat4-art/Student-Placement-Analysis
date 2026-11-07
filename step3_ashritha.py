# ============================================================
# STEP 3 (ASHRITHA): 2.4b Graphical Analysis (all graphs)
# Input : placement_ready.csv   (made by Ananya)
# Output: figs/fig01_...png to fig12_...png
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # lets the script run without a screen
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
os.makedirs("figs", exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})
df = pd.read_csv("placement_ready.csv")
n = len(df)

def bar_counts(series, title, fname, order=None, color=BLUE, horizontal=False, xlabel=""):
    """Bar chart of counts with the count written at the end of each bar."""
    vc = series.value_counts()
    if order is not None:
        vc = vc.reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    if horizontal:
        vc = vc.sort_values()
        ax.barh(vc.index.astype(str), vc.values, color=color, height=0.6)
        for i, v in enumerate(vc.values):
            ax.text(v + 0.3, i, f"{int(v)} ({v / n * 100:.0f}%)", va="center", fontsize=8)
        ax.set_xlabel("Number of students"); ax.grid(axis="y", visible=False)
    else:
        ax.bar(vc.index.astype(str), vc.values, color=color, width=0.6)
        for i, v in enumerate(vc.values):
            ax.text(i, v + 0.3, f"{int(v)}", ha="center", fontsize=8)
        ax.set_ylabel("Number of students"); ax.grid(axis="x", visible=False)
        ax.set_xlabel(xlabel)
        plt.xticks(rotation=20, ha="right")
    ax.set_title(title)
    fig.savefig(f"figs/{fname}"); plt.close(fig)

# Fig 1: CGPA distribution
bar_counts(df["cgpa_band"], "CGPA band of respondents", "fig01_cgpa.png",
           order=["6.0 - 7.0", "7.0 - 8.0", "8.0 - 9.0", "9.0 - 10.0"], xlabel="CGPA band")

# Fig 2: Histogram of coding problems solved (log-spaced bins, data is very skewed)
fig, ax = plt.subplots(figsize=(6, 3.4))
bins = [0, 1, 5, 10, 25, 50, 100, 301]
labels = ["0", "1-4", "5-9", "10-24", "25-49", "50-99", "100+"]
cut = pd.cut(df["problems_solved"], bins=bins, labels=labels, right=False)
cnt = cut.value_counts().reindex(labels)
ax.bar(labels, cnt.values, color=BLUE, width=0.7)
for i, v in enumerate(cnt.values):
    ax.text(i, v + 0.3, int(v), ha="center", fontsize=8)
ax.set_title("Coding problems solved (histogram)"); ax.set_xlabel("Problems solved")
ax.set_ylabel("Number of students"); ax.grid(axis="x", visible=False)
fig.savefig("figs/fig02_problems_hist.png"); plt.close(fig)

# Fig 3: Box plots BEFORE capping (shows the outliers found in 2.2)
fig, ax = plt.subplots(1, 4, figsize=(9, 3.2))
for a, (c, t) in zip(ax, [("num_languages", "Languages"), ("problems_solved", "Problems"),
                          ("hackathons", "Hackathons"), ("mock_interviews", "Mock interviews")]):
    a.boxplot(df[c], widths=0.5, patch_artist=True,
              boxprops=dict(facecolor="#cfe0f6", color=BLUE),
              medianprops=dict(color=ORANGE, linewidth=1.5),
              flierprops=dict(marker="o", markerfacecolor=BLUE, markersize=4,
                              markeredgecolor="white"))
    a.set_title(t); a.set_xticks([]); a.grid(axis="x", visible=False)
fig.suptitle("Box plots: outliers in count variables", fontsize=11, fontweight="bold", y=1.02)
fig.savefig("figs/fig03_boxplots.png"); plt.close(fig)

# Fig 4: Internship status and mode
fig, axs = plt.subplots(1, 2, figsize=(8, 3.2))
vc = df["internship"].value_counts()
axs[0].bar(vc.index, vc.values, color=[BLUE, ORANGE][:len(vc)], width=0.5)
for i, v in enumerate(vc.values):
    axs[0].text(i, v + 0.5, f"{v} ({v / n * 100:.0f}%)", ha="center", fontsize=8)
axs[0].set_title("Completed an internship?"); axs[0].set_ylabel("Students")
axs[0].grid(axis="x", visible=False)
vm = df[df["internship"] == "Yes"]["internship_mode"].value_counts()
axs[1].bar(vm.index, vm.values, color=BLUE, width=0.5)
for i, v in enumerate(vm.values):
    axs[1].text(i, v + 0.4, int(v), ha="center", fontsize=8)
axs[1].set_title("Mode (interns only)"); axs[1].grid(axis="x", visible=False)
fig.savefig("figs/fig04_internship.png"); plt.close(fig)

# Fig 5: Mock interviews attended
mi = df["mock_interviews"].round().clip(upper=5).astype(int).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar([("5+" if i == 5 else str(i)) for i in mi.index], mi.values, color=BLUE, width=0.6)
for i, v in enumerate(mi.values):
    ax.text(i, v + 0.5, int(v), ha="center", fontsize=8)
ax.set_title("Mock interviews attended"); ax.set_xlabel("Number of mock interviews")
ax.set_ylabel("Students"); ax.grid(axis="x", visible=False)
fig.savefig("figs/fig05_mock.png"); plt.close(fig)

# Fig 6: Self-ratings (aptitude, communication, resume) as small multiples
fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
specs = [("aptitude", ["Not practiced yet", "Weak", "Moderate", "Strong"], "Aptitude"),
         ("communication", ["Low confidence", "Moderate", "Very confident"], "Communication"),
         ("resume_quality", ["Needs improvement", "Average", "Good", "Excellent"],
          "Resume quality")]
for a, (c, order, t) in zip(axs, specs):
    v = df[c].value_counts().reindex(order).fillna(0)
    a.bar(range(len(order)), v.values, color=BLUE, width=0.6)
    a.set_xticks(range(len(order)))
    a.set_xticklabels([o.replace(" ", "\n", 1) for o in order], fontsize=7)
    for i, val in enumerate(v.values):
        a.text(i, val + 0.5, int(val), ha="center", fontsize=8)
    a.set_title(t); a.grid(axis="x", visible=False)
axs[0].set_ylabel("Students")
fig.savefig("figs/fig06_selfratings.png"); plt.close(fig)

# Fig 7: Expected salary
bar_counts(df["salary_expect"], "Expected salary package", "fig07_salary.png",
           order=["Below 4 LPA", "4 - 6 LPA", "6 - 10 LPA", "10 - 15 LPA", "Above 15 LPA"],
           xlabel="Expected package")

# Fig 8: Expected domain
bar_counts(df["domain_expect"], "Domain students expect to be placed in",
           "fig08_domain.png", horizontal=True)

# Fig 9: Company interest
comp = [c for c in df.columns if c.startswith("co_")]
ci = df[comp].sum().sort_values()
ci.index = [c[3:] for c in ci.index]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.barh(ci.index, ci.values, color=BLUE, height=0.6)
for i, v in enumerate(ci.values):
    ax.text(v + 0.5, i, f"{int(v)} ({v / n * 100:.0f}%)", va="center", fontsize=8)
ax.set_title("Companies students are interested in (multi-select)")
ax.set_xlabel("Number of students"); ax.grid(axis="y", visible=False)
fig.savefig("figs/fig09_companies.png"); plt.close(fig)

# Fig 10: Scatter, problems solved vs hackathons (jittered a little so dots don't hide)
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
jitter = rng.uniform(-0.12, 0.12, n)          # tiny vertical shake so dots do not hide
ax.scatter(df["problems_solved"], df["hackathons"] + jitter,
           s=28, color=BLUE, edgecolor="white", linewidth=0.8, alpha=0.85)
m, b = np.polyfit(df["problems_solved"], df["hackathons"], 1)
xs = np.linspace(0, df["problems_solved"].max(), 50)
r = df["problems_solved"].corr(df["hackathons"])
ax.plot(xs, m * xs + b, color=ORANGE, linewidth=1.5, label=f"trend line (r = {r:.2f})")
ax.set_xlabel("Coding problems solved"); ax.set_ylabel("Hackathons participated")
ax.set_title("Problem solving vs hackathon participation"); ax.legend(frameon=False, fontsize=8)
fig.savefig("figs/fig10_scatter.png"); plt.close(fig)

# Fig 11: Correlation heatmap (diverging: blue = negative, grey = 0, orange = positive)
cols = {"cgpa_enc": "CGPA", "num_languages": "Languages", "problems_solved": "Problems",
        "hackathons": "Hackathons", "mock_interviews": "Mocks", "aptitude_enc": "Aptitude",
        "communication_enc": "Communication", "resume_quality_enc": "Resume",
        "salary_enc": "Salary exp.", "dream_confidence_enc": "Dream-co. conf.",
        "internship_enc": "Internship", "num_clubs": "Clubs"}
cm = df[list(cols)].corr(); cm.index = cm.columns = list(cols.values())
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#e6e5e1", ORANGE])
fig, ax = plt.subplots(figsize=(7.4, 6.2))
im = ax.imshow(cm.values, cmap=cmap, vmin=-1, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.columns, fontsize=8)
for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(cm.values[i, j]) > 0.6 else "#0b0b0b")
ax.grid(False); ax.set_title("Correlation heatmap")
fig.colorbar(im, shrink=0.75, label="Correlation (r)")
fig.savefig("figs/fig11_heatmap.png"); plt.close(fig)

# Fig 12: Readiness Index by internship and by club membership (box plots)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
for a, c, t in [(axs[0], "internship", "By internship"),
                (axs[1], "tech_club", "By tech-club membership")]:
    groups = [df[df[c] == g]["readiness_index"] for g in ["No", "Yes"]]
    bp = a.boxplot(groups, tick_labels=["No", "Yes"], widths=0.5, patch_artist=True,
                   medianprops=dict(color="#0b0b0b", linewidth=1.5))
    for patch, col in zip(bp["boxes"], [GREY, BLUE]):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor(col)
    a.set_title(t); a.grid(axis="x", visible=False)
axs[0].set_ylabel("Readiness Index (0-100)")
fig.savefig("figs/fig12_readiness_groups.png"); plt.close(fig)

print("Saved", len(os.listdir("figs")), "figures in the figs/ folder")

# Show every graph inside Colab / Jupyter (does nothing harmful elsewhere)
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        from IPython.display import Image, display
        for f in sorted(os.listdir("figs")):
            print(f)
            display(Image(f"figs/{f}"))
except ImportError:
    pass
