"""Domain features for bodyfat, written from the schema and description only."""
import numpy as np
import pandas as pd


def siri(df):
    # description: Siri's equation, percent body fat = 495/D - 450
    return 495.0 / df["Density"].replace(0, np.nan) - 450.0


def brozek(df):
    # OUTSIDE KNOWLEDGE: Brozek equation, percent body fat = 457/D - 414.2
    return 457.0 / df["Density"].replace(0, np.nan) - 414.2


def bmi(df):
    # OUTSIDE KNOWLEDGE: BMI = 703 * lb / in^2
    return 703.0 * df["Weight"] / (df["Height"] ** 2).replace(0, np.nan)


def waist_ratios(df):
    # central adiposity: abdomen relative to hip, height (cm) and chest
    return pd.DataFrame({"abd_hip": df["Abdomen"] / df["Hip"].replace(0, np.nan),
                         "abd_height": df["Abdomen"] / (df["Height"] * 2.54).replace(0, np.nan),
                         "abd_chest": df["Abdomen"] / df["Chest"].replace(0, np.nan)})


def abdomen_minus_wrist(df):
    # abdomen (fat) minus wrist (frame size, little fat): classic body-fat predictor contrast
    return df["Abdomen"] - df["Wrist"]


def frame_adjusted_weight(df):
    # weight per frame size (wrist, ankle, knee are bony sites)
    frame = (df["Wrist"] + df["Ankle"] + df["Knee"]).replace(0, np.nan)
    return pd.DataFrame({"wt_per_frame": df["Weight"] / frame, "wt_per_wrist_ht": df["Weight"] / (df["Wrist"] * df["Height"]).replace(0, np.nan)})


def navy_method(df):
    # OUTSIDE KNOWLEDGE: US Navy circumference formula for men (inches): 86.01*log10(abdomen-neck) - 70.041*log10(height) + 36.76
    diff = ((df["Abdomen"] - df["Neck"]) / 2.54).clip(lower=1e-3)
    return 86.010 * np.log10(diff) - 70.041 * np.log10(df["Height"].clip(lower=1e-3)) + 36.76


def body_volume_proxy(df):
    # mass / density gives body volume; fat mass via Siri
    vol = (df["Weight"] * 0.4536) / df["Density"].replace(0, np.nan)
    return pd.DataFrame({"body_volume_l": vol, "fat_mass_kg": siri(df) / 100.0 * df["Weight"] * 0.4536})


PROPOSALS = [("siri", siri), ("brozek", brozek), ("bmi", bmi), ("waist_ratios", waist_ratios),
             ("abdomen_minus_wrist", abdomen_minus_wrist), ("frame_adjusted_weight", frame_adjusted_weight),
             ("navy_method", navy_method), ("body_volume_proxy", body_volume_proxy)]
