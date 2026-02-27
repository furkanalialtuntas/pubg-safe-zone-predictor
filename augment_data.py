import math
import pandas as pd
import numpy as np

from constants import (
    PHASE_RADIUS,
    PHASE1_X_MIN, PHASE1_X_MAX,
    PHASE1_Y_MIN, PHASE1_Y_MAX,
)

NUM_SYNTHETIC_MATCHES = 2000
REAL_DATA_PATH        = "zone_data.csv"
OUTPUT_PATH           = "data_augmented.csv"


def generate_match(match_num: int, rng: np.random.Generator) -> list[dict]:
    match_id = f"mac{match_num:04d}"
    rows: list[dict] = []

    cx = rng.uniform(PHASE1_X_MIN, PHASE1_X_MAX)
    cy = rng.uniform(PHASE1_Y_MIN, PHASE1_Y_MAX)

    for phase in range(1, 5):
        r = PHASE_RADIUS[phase]
        rows.append({
            "Match_ID": match_id,
            "Phase":    phase,
            "White_X":  round(cx, 2),
            "White_Y":  round(cy, 2),
            "White_R":  r,
        })

        if phase < 4:
            next_r     = PHASE_RADIUS[phase + 1]
            max_offset = r - next_r

            angle = rng.uniform(0, 2 * math.pi)
            # Beta(0.5, 2.0): gerçek oyundaki asimetrik zone kaymasını taklit eder
            dist  = rng.beta(0.5, 2.0) * max_offset

            cx = cx + dist * math.cos(angle)
            cy = cy + dist * math.sin(angle)

    return rows


def validate_match(rows: list[dict]) -> bool:
    for i in range(len(rows) - 1):
        curr, nxt = rows[i], rows[i + 1]
        prev_r = PHASE_RADIUS[curr["Phase"]]
        next_r = PHASE_RADIUS[nxt["Phase"]]
        d = math.hypot(
            nxt["White_X"] - curr["White_X"],
            nxt["White_Y"] - curr["White_Y"],
        )
        if d > (prev_r - next_r) + 1e-6:
            return False
    return True


def main() -> None:
    rng = np.random.default_rng(seed=42)

    real_df = pd.read_csv(REAL_DATA_PATH)
    print(f"Gerçek veri   : {len(real_df)} satır  ({real_df['Match_ID'].nunique()} maç)")

    existing_nums = []
    for mid in real_df["Match_ID"].unique():
        try:
            existing_nums.append(int(mid.replace("mac", "")))
        except ValueError:
            pass
    start_num = max(existing_nums) + 1 if existing_nums else 7

    synthetic_rows: list[dict] = []
    invalid_count = 0

    for match_num in range(start_num, start_num + NUM_SYNTHETIC_MATCHES):
        rows = generate_match(match_num, rng)
        if not validate_match(rows):
            invalid_count += 1
            continue
        synthetic_rows.extend(rows)

    synthetic_df = pd.DataFrame(synthetic_rows)
    print(f"Sentetik veri : {len(synthetic_df)} satır  ({synthetic_df['Match_ID'].nunique()} maç)")

    if invalid_count:
        print(f"  ⚠  {invalid_count} maç kural ihlali nedeniyle atlandı.")

    combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
    combined_df["Phase"]   = combined_df["Phase"].astype(int)
    combined_df["White_R"] = combined_df["White_R"].astype(int)

    combined_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nToplam        : {len(combined_df)} satır  ({combined_df['Match_ID'].nunique()} maç)")
    print(f"Çıktı         : '{OUTPUT_PATH}'")

    print("\n── Faz bazlı yarıçap (sentetik) ──")
    for phase in range(1, 5):
        vals = synthetic_df[synthetic_df["Phase"] == phase]["White_R"].unique()
        print(f"  Faz {phase}: R = {vals}  (beklenen: {PHASE_RADIUS[phase]})")

    print("\n── Koordinat aralıkları (sentetik, Faz 1) ──")
    faz1 = synthetic_df[synthetic_df["Phase"] == 1]
    print(f"  X: [{faz1['White_X'].min():.1f},  {faz1['White_X'].max():.1f}]")
    print(f"  Y: [{faz1['White_Y'].min():.1f},  {faz1['White_Y'].max():.1f}]")


if __name__ == "__main__":
    main()
