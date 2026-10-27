"""Generate the synthetic student performance dataset used in this project.

Run from the project root:  python data/generate_data.py
The output is data/student_performance.csv. The seed is fixed, so everyone gets the same file.

Each row is one student of a fictional college. The final exam score follows realistic rules
(previous GPA, the early internal test, attendance, study hours and assignment submission help;
a long commute, a heavy part-time job and very little sleep hurt) plus random noise.
Gender has NO effect on the score by design, which lets us check later whether the model is fair.

A few rows are made messy on purpose (missing values, impossible values such as attendance 120%,
duplicated rows) so the notebook can show how to clean them.
"""
import os

import numpy as np
import pandas as pd

SEED = 7
N = 2000


def make_dataset(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    gender = rng.choice(["Female", "Male"], size=n, p=[0.48, 0.52])
    age = rng.choice([18, 19, 20, 21, 22, 23, 24], size=n, p=[0.18, 0.27, 0.24, 0.16, 0.08, 0.05, 0.02])
    family_support = rng.choice(["Low", "Medium", "High"], size=n, p=[0.25, 0.45, 0.30])
    part_time_job = rng.choice(["Yes", "No"], size=n, p=[0.28, 0.72])
    internet_access = rng.choice(["Yes", "No"], size=n, p=[0.85, 0.15])
    extracurricular = rng.choice(["Yes", "No"], size=n, p=[0.40, 0.60])

    # A hidden "motivation" value links several habits together (motivated students attend AND study more).
    motivation = rng.normal(0, 1, n)
    support_boost = pd.Series(family_support).map({"Low": -0.4, "Medium": 0.0, "High": 0.3}).to_numpy()

    previous_sem_gpa = (6.6 + 1.1 * motivation + 0.4 * support_boost + rng.normal(0, 0.8, n)).clip(3.0, 10.0).round(2)
    study_hours = (12 + 4.5 * motivation - 3 * (part_time_job == "Yes") + rng.normal(0, 4, n)).clip(0, 40).round(1)
    attendance = (80 + 7 * motivation - 3 * (part_time_job == "Yes") + rng.normal(0, 8, n)).clip(30, 100).round(1)
    assignments = (82 + 9 * motivation - 6 * (internet_access == "No") + rng.normal(0, 10, n)).clip(0, 100).round(0)
    commute = rng.gamma(2.2, 14, n).clip(5, 120).round(0)
    sleep = (7.0 - 0.6 * (part_time_job == "Yes") + rng.normal(0, 0.9, n)).clip(4, 10).round(1)
    # Internal test (out of 50) is written in week 5 and already reflects ability + effort.
    internal = (27 + 2.2 * (previous_sem_gpa - 6.6) + 0.25 * (study_hours - 12)
                + rng.normal(0, 5, n)).clip(0, 50).round(0)

    final_score = (
        -8.7
        + 3.2 * previous_sem_gpa
        + 0.55 * internal
        + 0.22 * attendance
        + 0.35 * study_hours
        + 0.08 * assignments
        - 0.06 * commute
        - 2.5 * np.abs(sleep - 7.5)                 # too little (or too much) sleep hurts
        + 4 * support_boost
        - 2 * (internet_access == "No")
        + 1 * (extracurricular == "Yes")
        + rng.normal(0, 7, n)                        # everything we cannot measure
    ).clip(0, 100).round(0)

    df = pd.DataFrame({
        "student_id": [f"S{i:05d}" for i in range(1, n + 1)],
        "gender": gender, "age": age,
        "study_hours_per_week": study_hours, "attendance_pct": attendance,
        "previous_sem_gpa": previous_sem_gpa, "assignments_submitted_pct": assignments,
        "internal_test_score": internal, "part_time_job": part_time_job,
        "commute_minutes": commute, "family_support": family_support,
        "internet_access": internet_access, "extracurricular": extracurricular,
        "sleep_hours": sleep, "final_score": final_score,
    })

    # ---- realistic messiness ----
    # 1. missing values (forms left blank)
    for col, k in (("study_hours_per_week", 45), ("attendance_pct", 30), ("sleep_hours", 40),
                   ("family_support", 35), ("internet_access", 20), ("commute_minutes", 25)):
        idx = rng.choice(n, size=k, replace=False)
        if df[col].dtype == object or df[col].dtype.name in ("str", "string"):
            df[col] = df[col].astype("object")
        else:
            df[col] = df[col].astype("float")
        df.loc[idx, col] = np.nan
    # 2. impossible values (typing mistakes)
    df.loc[rng.choice(n, 6, replace=False), "attendance_pct"] = [120, 105, 150, 101.5, 110, 999]
    df.loc[rng.choice(n, 4, replace=False), "sleep_hours"] = [0, 24, 30, -1]
    df.loc[rng.choice(n, 3, replace=False), "study_hours_per_week"] = [-5, 200, 168]
    # 3. the same student entered twice
    dupes = df.sample(30, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_performance.csv")
    data = make_dataset()
    data.to_csv(out, index=False)
    print(f"Wrote {len(data)} rows to {out}")
    print(f"Share of students with final_score < 40: {(data['final_score'] < 40).mean():.1%}")
