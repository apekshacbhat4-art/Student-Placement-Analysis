# ============================================================
# STEP 2 (ANANYA): 2.3 Data Preprocessing + 2.4 Statistical Analysis
# Input : placement_clean.csv   (made by Apeksha)
# Output: placement_processed.csv  (all-numeric, scaled, for ML / heatmap)
#         placement_ready.csv      (clean + Readiness_Index, for graphs)
# ============================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
df = pd.read_csv("placement_clean.csv")

# ---------- 2.3 DATA PREPROCESSING ----------
# (1) ENCODING -------------------------------------------------
# (a) Ordinal encoding: categories that have a natural order
ordinal_maps = {
    "aptitude":         {"Not practiced yet": 0, "Weak": 1, "Moderate": 2, "Strong": 3},
    "communication":    {"Low confidence": 1, "Moderate": 2, "Very confident": 3},
    "dream_confidence": {"Low": 1, "Not sure": 2, "Moderate": 3, "Very confident": 4},
    "resume_quality":   {"Needs improvement": 1, "Average": 2, "Good": 3, "Excellent": 4},
    "club_helpful":     {"Not Applicable": 0, "Not much": 1, "Somewhat": 2, "Yes, a lot": 3},
}
for col, mp in ordinal_maps.items():
    df[col + "_enc"] = df[col].map(mp)

# (b) Ranges -> mid-point numbers (assumption: "Above 15" counted as 18 LPA)
salary_mid = {"Below 4 LPA": 3, "4 - 6 LPA": 5, "6 - 10 LPA": 8,
              "10 - 15 LPA": 12.5, "Above 15 LPA": 18}
cgpa_mid = {"6.0 - 7.0": 6.5, "7.0 - 8.0": 7.5, "8.0 - 9.0": 8.5, "9.0 - 10.0": 9.5}
df["salary_enc"] = df["salary_expect"].map(salary_mid)
df["cgpa_enc"] = df["cgpa_band"].map(cgpa_mid)

# (c) Binary encoding
df["internship_enc"] = (df["internship"] == "Yes").astype(int)
df["tech_club_enc"] = (df["tech_club"] == "Yes").astype(int)

# (d) One-hot encoding: categories with NO natural order
onehot = pd.get_dummies(df[["internship_mode", "domain_expect"]], dtype=int)

# (2) READINESS INDEX (used for EDA: 0 = least ready, 100 = most ready) ------
# average of min-max scaled readiness indicators (CGPA & salary are NOT included)
parts = ["problems_solved_capped", "hackathons_capped", "mock_interviews_capped",
         "aptitude_enc", "communication_enc", "resume_quality_enc", "internship_enc"]
mm_parts = pd.DataFrame(MinMaxScaler().fit_transform(df[parts]), columns=parts)
df["readiness_index"] = (mm_parts.mean(axis=1) * 100).round(1)
df["readiness_level"] = pd.cut(df["readiness_index"], bins=[-1, 33, 50, 101],
                               labels=["Low", "Medium", "High"])

# (3) NORMALIZATION (Min-Max: values squeezed into 0-1) -----------------------
cont = ["num_languages_capped", "problems_solved_capped", "hackathons_capped",
        "mock_interviews_capped", "num_companies", "num_clubs", "salary_enc", "cgpa_enc"]
norm = pd.DataFrame(MinMaxScaler().fit_transform(df[cont]),
                    columns=[c + "_norm" for c in cont])

# (4) STANDARDIZATION (Z-score: mean 0, std 1) --------------------------------
std = pd.DataFrame(StandardScaler().fit_transform(df[cont]),
                   columns=[c + "_std" for c in cont])
print("Check normalization  -> min/max of one column:",
      norm.iloc[:, 1].min(), norm.iloc[:, 1].max())
print("Check standardization -> mean/std of one column:",
      round(std.iloc[:, 1].mean(), 6), round(std.iloc[:, 1].std(ddof=0), 6))

# (5) Assemble the numeric table
enc_cols = [c + "_enc" for c in ordinal_maps] + ["internship_enc", "tech_club_enc"]
# (salary_enc and cgpa_enc are already inside `cont`, so they are not repeated here)
model_df = pd.concat([df[cont].reset_index(drop=True),
                      df[enc_cols].reset_index(drop=True),
                      onehot.reset_index(drop=True)], axis=1)

# (6) FEATURE SELECTION --------------------------------------------------------
# Target = self-rated confidence of getting the dream company (1-4)
target = "dream_confidence_enc"
X = model_df.drop(columns=[target])

# Step A: remove one of every pair of features with |correlation| > 0.8
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if (upper[c] > 0.8).any()]
print("\nDropped for high correlation (>0.8):", drop_corr)
X = X.drop(columns=drop_corr)

# Step B: drop near-constant columns (almost all students gave the same answer)
low_var = [c for c in X.columns if X[c].var() < 0.01]
print("Dropped for near-zero variance:", low_var)
X = X.drop(columns=low_var)

# Step C: rank the rest with SelectKBest (F-test) against the target
skb = SelectKBest(f_regression, k=8).fit(X, model_df[target])
scores = pd.DataFrame({"feature": X.columns, "F_score": skb.scores_,
                       "p_value": skb.pvalues_}).sort_values("F_score", ascending=False)
print("\n--- Feature ranking (F-test vs dream-company confidence) ---")
print(scores.round(3).to_string(index=False))
selected = list(X.columns[skb.get_support()])
print("\nTop-8 selected features:", selected)

model_df[[target]].join(X[selected]).to_csv("placement_processed.csv", index=False)
df.to_csv("placement_ready.csv", index=False)

# ---------- 2.4a STATISTICAL ANALYSIS ----------
numeric_view = ["cgpa_enc", "num_languages", "problems_solved", "hackathons",
                "mock_interviews", "num_companies", "num_clubs", "salary_enc",
                "aptitude_enc", "communication_enc", "resume_quality_enc",
                "dream_confidence_enc", "readiness_index"]
desc = df[numeric_view].describe().T[["count", "mean", "std", "min", "50%", "max"]]
desc["mode"] = [df[c].mode()[0] for c in numeric_view]
desc["skewness"] = [df[c].skew() for c in numeric_view]
print("\n--- Descriptive statistics ---")
print(desc.round(2).to_string())
desc.round(2).to_csv("table_descriptive_stats.csv")

print("\n--- Frequency tables (count and %) ---")
for c in ["cgpa_band", "internship", "tech_club", "aptitude", "communication",
          "resume_quality", "salary_expect", "domain_expect", "domain_language",
          "readiness_level"]:
    t = df[c].value_counts()
    pct = (t / len(df) * 100).round(1)
    print(f"\n{c}\n", pd.DataFrame({"count": t, "percent": pct}).to_string())

corr_m = df[numeric_view].corr()
corr_m.round(2).to_csv("table_correlation.csv")
pairs = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool)).stack().dropna()
print("\n--- 6 strongest positive and negative correlations ---")
print(pairs.sort_values().head(6).round(2).to_string())
print(pairs.sort_values().tail(6).round(2).to_string())

print("\n--- Mean readiness_index by group ---")
for c in ["internship", "tech_club", "cgpa_band", "domain_expect"]:
    print(df.groupby(c)["readiness_index"].agg(["count", "mean"]).round(1).to_string(), "\n")

print("Corr(readiness_index, cgpa):", round(df["readiness_index"].corr(df["cgpa_enc"]), 2))
print("Corr(readiness_index, dream confidence):",
      round(df["readiness_index"].corr(df["dream_confidence_enc"]), 2))
print("Corr(problems_solved, hackathons):",
      round(df["problems_solved"].corr(df["hackathons"]), 2))
