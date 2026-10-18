"""Domain features for colic, written from the schema and description only."""
import numpy as np
import pandas as pd


def _m(df, c, mapping):
    return df[c].astype(object).map(mapping).astype(float)


def perfusion_score(df):
    # circulatory compromise (endotoxic/hypovolaemic shock): cold extremities, weak pulse, cyanotic membranes, slow refill
    te = _m(df, "temp_extremities", {"normal": 0, "warm": 0, "cool": 1, "cold": 2})
    pp = _m(df, "peripheral_pulse", {"normal": 0, "increased": 0, "reduced": 1, "absent": 2})
    mm = _m(df, "mucous_membranes", {"normal pink": 0, "bright pink": 1, "pale pink": 1, "pale cyanotic": 2,
                                     "bright red": 2, "dark cyanotic": 3})
    crt = _m(df, "capillary_refill_time", {"<3": 0, ">=3": 1})
    return pd.DataFrame({"perfusion_sum": te.fillna(0) + pp.fillna(0) + mm.fillna(0) + crt.fillna(0),
                         "membranes_ord": mm, "extrem_ord": te})


def pain_ord(df):
    # pain severity is a key surgical indicator: continuous severe pain suggests strangulating lesion
    return _m(df, "pain", {"alert no pain": 0, "depressed": 1, "intermittent mild pain": 2,
                           "intermittent severe pain": 3, "continuous severe pain": 4})


def obstruction_score(df):
    # GI obstruction signs: silent gut, distension, large gastric reflux, absent faeces, distended small intestine
    per = _m(df, "peristalsis", {"normal": 0, "hypermotile": 0, "hypomotile": 1, "absent": 2})
    dis = _m(df, "abdominal_distension", {"none": 0, "slight": 1, "moderate": 2, "severe": 3})
    ref = _m(df, "nasogastric_reflux", {"none": 0, "<1l": 1, ">1l": 2})
    rec = _m(df, "rectal_examination", {"normal": 0, "increased": 0, "decreased": 1, "absent": 2})
    si = (df["abdomen"].astype(object) == "distended small intestine").astype(float)
    return pd.DataFrame({"obstruction_sum": per.fillna(0) + dis.fillna(0) + ref.fillna(0) + rec.fillna(0) + 2 * si,
                         "distension_ord": dis, "peristalsis_ord": per, "small_int_distended": si})


def vitals_deviation(df):
    # tachycardia and fever/hypothermia: pulse well above ~40 bpm resting suggests pain/shock in a horse
    # OUTSIDE KNOWLEDGE: normal adult horse temp ~37.8 C, resting pulse 30-40 bpm
    return pd.DataFrame({"temp_dev": (df["rectal_temperature"] - 37.8).abs(),
                         "pulse_over_40": df["pulse"] / 40.0,
                         "pulse_per_resp": df["pulse"] / df["respiratory_rate"].replace(0, np.nan)})


def hemoconcentration(df):
    # dehydration raises PCV; PCV/protein separates fluid loss from protein loss into gut
    tp = df["total_protein"].where(df["total_protein"] < 20, df["total_protein"] / 10.0)  # g/L entries -> g/dl
    return pd.DataFrame({"pcv": df["packed_cell_volume"], "tp_gdl": tp,
                         "pcv_over_tp": df["packed_cell_volume"] / tp.replace(0, np.nan),
                         "pcv_high": (df["packed_cell_volume"] > 50).astype(float)})


def abdominocentesis(df):
    # bloody (serosanguinous) peritoneal fluid with high protein indicates devitalised bowel
    a = _m(df, "abdominocentesis_appearance", {"clear": 0, "cloudy": 1, "serosanguinous": 2})
    return pd.DataFrame({"abdo_app_ord": a, "abdo_tp": df["abdomcentesis_total_protein"],
                         "abdo_app_x_tp": a.fillna(0) * df["abdomcentesis_total_protein"].fillna(0)})


def outcome_surgery(df):
    # outcome and whether surgery was performed: non-survival and surgical treatment track lesion severity
    o = _m(df, "outcome", {"lived": 0, "euthanized": 1, "died": 1})
    s = _m(df, "surgery", {"no": 0, "yes": 1})
    return pd.DataFrame({"not_survived": o, "surgery_yes": s, "n_missing": df.isna().sum(axis=1).astype(float)})


PROPOSALS = [("perfusion_score", perfusion_score), ("pain_ord", pain_ord), ("obstruction_score", obstruction_score),
             ("vitals_deviation", vitals_deviation), ("hemoconcentration", hemoconcentration),
             ("abdominocentesis", abdominocentesis), ("outcome_surgery", outcome_surgery)]
