# ============================================================
# STEP 1 (APEKSHA): 2.1 Dataset Understanding + 2.2 Data Cleaning
# Input : responses.csv   (Google Sheet -> File -> Download -> CSV)
# Output: placement_clean.csv  (cleaned, still human-readable)
# ============================================================
import re
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------- 2.1 DATASET UNDERSTANDING ----------
raw = pd.read_csv("responses.csv")
print("Shape (rows, columns):", raw.shape)

# Long question text -> short column names
short = ["timestamp", "cgpa_band", "num_languages", "domain_language",
         "problems_solved", "hackathons", "internship", "internship_mode",
         "tech_club", "club_names", "club_helpful", "aptitude",
         "mock_interviews", "communication", "companies", "dream_confidence",
         "salary_expect", "domain_expect", "resume_quality"]
raw.columns = short

print("\n--- Data types ---")
print(raw.dtypes)
print("\n--- Missing values per column ---")
print(raw.isna().sum())
print("\n--- Duplicate rows:", raw.duplicated().sum())
print("\n--- Unique values of some categorical columns ---")
for c in ["cgpa_band", "internship", "internship_mode", "aptitude",
          "communication", "dream_confidence", "salary_expect",
          "resume_quality", "club_helpful"]:
    print(f"\n{c}:\n", raw[c].value_counts(dropna=False).to_string())

df = raw.copy()

# ---------- 2.2 DATA CLEANING ----------
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# (a) Drop almost-empty responses (people who filled only the first 2-3 questions)
missing_count = df.drop(columns="timestamp").isna().sum(axis=1)
print("\nRows with 10 or more blanks (to drop):", (missing_count >= 10).sum())
df = df[missing_count < 10].copy()

# (b) Free text -> numbers. People typed "300+", "50-60", "Leetcode", "Nope"...
ZERO_WORDS = r"\b(none|nope|nil|zero|no|not yet|nothing|not yet started|none yet|nahi)\b"
WORD_NUM = {"one": 1, "two": 2, "three": 3}

def to_number(x):
    """Return a number from messy text, or NaN if it cannot be understood."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    nums = re.findall(r"\d+\.?\d*", s)
    if nums:
        nums = [float(n) for n in nums]
        return float(np.mean(nums[:2])) if len(nums) >= 2 else nums[0]  # "50-60" -> 55
    if re.search(ZERO_WORDS, s) or s in ("-", "--"):
        return 0.0
    for w, v in WORD_NUM.items():
        if re.search(rf"\b{w}\b", s):
            return float(v)
    return np.nan  # "Leetcode", "Hdhs", "Many", "Idk", "Few" ... cannot be a number

for col in ["problems_solved", "hackathons", "mock_interviews"]:
    df[col + "_raw"] = df[col]            # keep original for the cleaning log
    df[col] = df[col].apply(to_number)

# one answer talks about *conducting* 70+ mock interviews for juniors -> not attended
df.loc[df["mock_interviews_raw"].astype(str).str.contains("seniors", na=False),
       "mock_interviews"] = np.nan

# (c) Number of programming languages: sometimes a number, sometimes a list of names
KNOWN = {"c", "c++", "cpp", "c+", "java", "python", "js", "javascript", "html",
         "css", "perl", "shell", "sql", "reactjs"}

def count_languages(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    nums = re.findall(r"\d+\.?\d*", s)
    if nums:
        nums = [float(n) for n in nums]
        return float(np.mean(nums[:2])) if len(nums) >= 2 else nums[0]
    if s in WORD_NUM or s == "one":
        return 1.0
    s = s.replace("react js", "reactjs").replace("(as per syllabus )", "")
    tokens = [t for t in re.split(r"[,\s]+|\band\b", s) if t]
    found = [t for t in tokens if t in KNOWN]
    return float(len(found)) if found else np.nan   # "Bsbz" -> NaN

df["num_languages_raw"] = df["num_languages"]
df["num_languages"] = df["num_languages"].apply(count_languages)

# (d) Fix spelling / inconsistent labels
df["internship_mode"] = df["internship_mode"].replace({"Not Apllicable": "Not Applicable"})
df["club_helpful"] = df["club_helpful"].replace({"Not applicable": "Not Applicable"})
df["domain_language"] = df["domain_language"].replace({"No idea": "Not sure"})

# (e) Logical inconsistencies
#   internship = No but a mode was given  -> mode must be 'Not Applicable'
bad = (df["internship"] == "No") & (~df["internship_mode"].isin(["Not Applicable"])) \
      & df["internship_mode"].notna()
print("Internship=No but mode given:", bad.sum(), "rows -> fixed")
df.loc[df["internship"] == "No", "internship_mode"] = "Not Applicable"

#   club name given but tech_club blank -> student is in a club
fix_club = df["tech_club"].isna() & df["club_names"].notna() \
           & (df["club_names"] != "Not Applicable")
print("Club name given but tech_club blank:", fix_club.sum(), "rows -> set to Yes")
df.loc[fix_club, "tech_club"] = "Yes"
df.loc[df["tech_club"] == "No", "club_names"] = "Not Applicable"
df.loc[df["tech_club"] == "No", "club_helpful"] = "Not Applicable"

# (f) Group the expected-domain answers into clean categories
def clean_domain(x):
    if pd.isna(x):
        return np.nan
    s = x.lower()
    if "core" in s:                       return "Core CS"
    if "data" in s or s.strip() == "ai":  return "Data Science / AI"
    if "web" in s or "frontend" in s:     return "Web Development"
    if "development" in s:                return "Software Development"
    if "not decided" in s:                return "Not decided"
    return "Other"                        # Managerial, Testing, Call center
df["domain_expect"] = df["domain_expect"].apply(clean_domain)

# (g) Multi-select columns -> counts (needed later for encoding)
COMPANIES = ["Infosys", "TCS", "Accenture", "Amazon", "Microsoft", "HP/HPE", "Jio"]
CLUBS = ["FLC", "ACM", "CSI", "IEEE", "HackerEarth", "Wiki Club", "Other"]

def has_item(text, item):
    return int(isinstance(text, str) and item.lower() in text.lower())

for c in COMPANIES:
    df["co_" + c] = df["companies"].apply(lambda t: has_item(t, c))
for c in CLUBS:
    df["club_" + c] = df["club_names"].apply(lambda t: has_item(t, c))
df["num_companies"] = df[[f"co_{c}" for c in COMPANIES]].sum(axis=1)
df["num_clubs"] = df[[f"club_{c}" for c in CLUBS]].sum(axis=1)

# (h) Missing values: median for numbers, mode for categories
num_cols = ["num_languages", "problems_solved", "hackathons", "mock_interviews"]
cat_cols = ["cgpa_band", "domain_language", "internship", "internship_mode",
            "tech_club", "club_helpful", "aptitude", "communication",
            "dream_confidence", "salary_expect", "domain_expect", "resume_quality"]

log = []
for c in num_cols:
    n = int(df[c].isna().sum()); med = float(np.floor(df[c].median()))  # whole number
    df[c + "_was_missing"] = df[c].isna().astype(int)
    df[c] = df[c].fillna(med)
    log.append((c, n, f"median = {med}"))
for c in cat_cols:
    n = int(df[c].isna().sum()); mode = df[c].mode()[0]
    df[c] = df[c].fillna(mode)
    log.append((c, n, f"mode = {mode}"))
print("\n--- Missing-value log (column, how many filled, filled with) ---")
for row in log:
    print(row)

# (i) Outliers: IQR rule, then cap (winsorize) instead of deleting rows
print("\n--- Outliers (IQR rule) ---")
for c in num_cols:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
    print(f"{c}: Q1={q1}, Q3={q3}, upper fence={hi:.1f}, outliers={n_out}")
    df[c + "_capped"] = df[c].clip(lower=max(lo, 0), upper=hi)

# (j) Final checks and save
df = df.drop(columns=["companies", "club_names"])   # now represented by 0/1 columns
print("\nFinal shape:", df.shape)
print("Remaining missing values:", int(df[num_cols + cat_cols].isna().sum().sum()))
df.to_csv("placement_clean.csv", index=False)
print("Saved placement_clean.csv")
