"""Domain features for naval, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: in the CODLAG dataset the removed constant inlet values are 288 K and 0.998 bar.
_T1 = 288.0
_P1 = 0.998
_GAMMA = 1.4


def _dev_from_operating_point(df, col):
    # Deviation from the median at the same lever position: decay shifts readings at fixed load.
    g = df.groupby("lever_position")[col].transform("median")
    return df[col] - g


# Compressor isentropic efficiency: degraded compressors need more temperature rise for the same pressure ratio.
# OUTSIDE KNOWLEDGE: isentropic efficiency formula eta = (PR^((g-1)/g) - 1) / (T2/T1 - 1).
def compressor_efficiency(df):
    pr = df["gt_compressor_outlet_air_pressure"] / _P1
    tr = df["gt_compressor_outlet_air_temperature"] / _T1
    return (pr ** ((_GAMMA - 1) / _GAMMA) - 1) / (tr - 1).replace(0, np.nan)


# Efficiency relative to its operating-point median (removes the load effect).
def compressor_efficiency_dev(df):
    eff = compressor_efficiency(df)
    tmp = pd.DataFrame({"lever_position": df["lever_position"], "eff": eff})
    return eff - tmp.groupby("lever_position")["eff"].transform("median")


# Compressor outlet pressure and temperature deviations at fixed lever position.
def compressor_deviations(df):
    return pd.DataFrame({
        "p2_dev": _dev_from_operating_point(df, "gt_compressor_outlet_air_pressure"),
        "t2_dev": _dev_from_operating_point(df, "gt_compressor_outlet_air_temperature"),
        "ggrpm_dev": _dev_from_operating_point(df, "gas_generator_rate_of_revolutions"),
    }, index=df.index)


# Fuel and turbine-side deviations at fixed lever position (degradation costs fuel).
def fuel_deviations(df):
    return pd.DataFrame({
        "fuel_dev": _dev_from_operating_point(df, "fuel_flow"),
        "tic_dev": _dev_from_operating_point(df, "turbine_injecton_control"),
        "t48_dev": _dev_from_operating_point(df, "hp_turbine_exit_temperature"),
        "torque_dev": _dev_from_operating_point(df, "gas_turbine_shaft_torque"),
    }, index=df.index)


# Compressor temperature rise per unit pressure ratio (work per compression).
def temp_rise_per_pr(df):
    return (df["gt_compressor_outlet_air_temperature"] - _T1) / (df["gt_compressor_outlet_air_pressure"] / _P1).replace(0, np.nan)


# Specific fuel consumption: fuel per shaft power (torque * rpm).
def specific_fuel(df):
    power = df["gas_turbine_shaft_torque"] * df["gas_turbine_rate_of_revolutions"]
    return df["fuel_flow"] / power.replace(0, np.nan)


# Compressor pressure ratio per gas-generator speed: a fouled compressor delivers less pressure at a given speed.
def pressure_per_ggspeed(df):
    return df["gt_compressor_outlet_air_pressure"] / df["gas_generator_rate_of_revolutions"].replace(0, np.nan)


# HP turbine expansion ratio relative to compressor ratio.
def turbine_vs_compressor_pr(df):
    return df["hp_turbine_exit_pressure"] / df["gt_compressor_outlet_air_pressure"].replace(0, np.nan)


PROPOSALS = [
    ("compressor_efficiency", compressor_efficiency),
    ("compressor_efficiency_dev", compressor_efficiency_dev),
    ("compressor_deviations", compressor_deviations),
    ("fuel_deviations", fuel_deviations),
    ("temp_rise_per_pr", temp_rise_per_pr),
    ("specific_fuel", specific_fuel),
    ("pressure_per_ggspeed", pressure_per_ggspeed),
    ("turbine_vs_compressor_pr", turbine_vs_compressor_pr),
]
