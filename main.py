#Tariff Design
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

app = FastAPI()

# --------- MODELOS DE ENTRADA ---------

class STR1Inputs(BaseModel):
    upper_limit: float
    customer_charge: float
    first_block: float
    second_block: float
    data: list

class STR2Inputs(BaseModel):
    upper_limit_std: float
    customer_charge_std: float
    first_block_std: float
    second_block_std: float
    upper_limit_bs: float
    customer_charge_bs: float
    first_block_bs: float
    break_even: float
    revenue_target: float
    data: list

class BDERInputs(BaseModel):
    data: list
    demand_lookup: list
    power_factor: float
    revenue_target: float
    energy_lrmc: float
    demand_lrmc: float
    customer_lrmc: float
    on_peak_demand_share: float
    partial_peak_demand_share: float
    off_peak_demand_share: float
    lrmc_on_peak: float
    lrmc_partial_peak: float
    lrmc_off_peak: float
    initial_off_peak_demand_charge: float
    on_peak_energy_share: float
    partial_peak_energy_share: float
    off_peak_energy_share: float
    initial_off_peak_energy_charge: float

class DPrePaidInputs(BaseModel):
    upper_limit: float
    early_payment_incentive: float
    revenue_target: float
    initial_second_block_charge: float
    data: list

# --------- ENDPOINTS ---------

@app.post("/str1")
def calculate_str1(inputs: STR1Inputs):
    df = pd.DataFrame(inputs.data)
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    for month in months:
        df[f"STR1_{month}"] = (
            inputs.customer_charge +
            (df[month].clip(upper=inputs.upper_limit) * inputs.first_block) +
            ((df[month] - inputs.upper_limit).clip(lower=0) * inputs.second_block)
        )

    df["SumSTR1"] = df[[f"STR1_{month}" for month in months]].sum(axis=1)
    total_revenue = df["SumSTR1"].sum()

    return {"total_revenue": total_revenue}

@app.post("/str2")
def calculate_str2(inputs: STR2Inputs):
    df = pd.DataFrame(inputs.data)
    df["Average"] = df[[ "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]].mean(axis=1)
    is_standard = df["Average"] > inputs.break_even
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    def objective(x):
        first_block_std = x
        second_block_std = inputs.second_block_std

        for month in months:
            billing_std = (
                inputs.customer_charge_std +
                (df[month].clip(upper=inputs.upper_limit_std) * first_block_std) +
                ((df[month] - inputs.upper_limit_std).clip(lower=0) * second_block_std)
            )

            billing_bs = (
                inputs.customer_charge_bs +
                (df[month].clip(upper=inputs.upper_limit_bs) * inputs.first_block_bs) +
                ((df[month] - inputs.upper_limit_bs).clip(lower=0) * inputs.first_block_bs)
            )

            df[f"STR2_{month}"] = billing_std.where(is_standard, billing_bs)

        df["SumSTR2"] = df[[f"STR2_{month}" for month in months]].sum(axis=1)
        total = df["SumSTR2"].sum()
        return total - inputs.revenue_target

    solution = fsolve(objective, inputs.first_block_std)[0]

    # Calcular revenue final con el valor optimizado
    for month in months:
        billing_std = (
            inputs.customer_charge_std +
            (df[month].clip(upper=inputs.upper_limit_std) * solution) +
            ((df[month] - inputs.upper_limit_std).clip(lower=0) * inputs.second_block_std)
        )

        billing_bs = (
            inputs.customer_charge_bs +
            (df[month].clip(upper=inputs.upper_limit_bs) * inputs.first_block_bs) +
            ((df[month] - inputs.upper_limit_bs).clip(lower=0) * inputs.first_block_bs)
        )

        df[f"STR2_{month}"] = billing_std.where(is_standard, billing_bs)

    df["SumSTR2"] = df[[f"STR2_{month}" for month in months]].sum(axis=1)
    revenue_total_str2 = df["SumSTR2"].sum()

    return {
        "first_block_std_final": solution,
        "revenue_total_str2_final": revenue_total_str2
    }

@app.post("/b_der")
def calculate_bder(inputs: BDERInputs):
    df = pd.DataFrame(inputs.data)
    demand_lookup = pd.DataFrame(inputs.demand_lookup)
    demand_lookup["consumption_kWh"] = demand_lookup["consumption_kWh"].astype(int)
    demand_lookup = demand_lookup.sort_values(by="consumption_kWh", ascending=False)

    demand_dict = demand_lookup.set_index("consumption_kWh")["demand_kW"].to_dict()
    consumption_values = list(demand_dict.keys())
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

    original_consumption = df[months].copy()

    for month in months:
        df[month] = df[month].astype(float)
        df[f"demand_{month}"] = 0
        mask = df[month] >= 1
        df.loc[mask, f"demand_{month}"] = None

        df.loc[mask, month] = df.loc[mask, month].apply(lambda x: max(1, int(x)))

        def find_nearest_lower(x):
            for val in consumption_values:
                if val <= x:
                    return demand_dict[val]
            return 0

        df.loc[mask, f"demand_{month}"] = df.loc[mask, month].apply(find_nearest_lower)
        df.loc[df[f"demand_{month}"] > 0, f"demand_{month}"] /= inputs.power_factor

    df[months] = original_consumption

    df["demand_Sum"] = df[[f"demand_{month}" for month in months]].sum(axis=1)
    total_demand = df["demand_Sum"].sum()

    # TOU Demand Charges
    def demand_objective(x):
        off_peak = x
        partial_peak = off_peak * inputs.lrmc_partial_peak
        on_peak = off_peak * inputs.lrmc_on_peak
        demand_revenue = (off_peak * total_demand * inputs.off_peak_demand_share +
                          partial_peak * total_demand * inputs.partial_peak_demand_share +
                          on_peak * total_demand * inputs.on_peak_demand_share)
        return inputs.demand_lrmc / (inputs.energy_lrmc + inputs.demand_lrmc + inputs.customer_lrmc) * inputs.revenue_target - demand_revenue

    optimized_off_peak_demand = fsolve(demand_objective, inputs.initial_off_peak_demand_charge)[0]

    # TOU Energy Charges
    def energy_objective(x):
        off_peak = x
        partial_peak = off_peak * inputs.lrmc_partial_peak
        on_peak = off_peak * inputs.lrmc_on_peak
        energy_revenue = (off_peak * inputs.revenue_target * inputs.off_peak_energy_share +
                          partial_peak * inputs.revenue_target * inputs.partial_peak_energy_share +
                          on_peak * inputs.revenue_target * inputs.on_peak_energy_share)
        return inputs.energy_lrmc / (inputs.energy_lrmc + inputs.demand_lrmc + inputs.customer_lrmc) * inputs.revenue_target - energy_revenue

    optimized_off_peak_energy = fsolve(energy_objective, inputs.initial_off_peak_energy_charge)[0]

    return {
        "optimized_off_peak_demand_charge": optimized_off_peak_demand,
        "optimized_off_peak_energy_charge": optimized_off_peak_energy,
        "total_demand_estimated": total_demand
    }

@app.post("/d_prepaid")
def calculate_dprepaid(inputs: DPrePaidInputs):
    df = pd.DataFrame(inputs.data)
    df["Average"] = df[["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]].mean(axis=1)

    def objective(x):
        pp_second_block = x
        df["PP_Bill_Month"] = (
            0 +
            df["Average"].clip(upper=inputs.upper_limit) * inputs.upper_limit +
            (df["Average"] - inputs.upper_limit).clip(lower=0) * pp_second_block
        )
        df["PP_Bill_Annual"] = df["PP_Bill_Month"] * 12
        revenue_total_pp = df["PP_Bill_Annual"].sum()
        return inputs.revenue_target - revenue_total_pp

    solution = fsolve(objective, inputs.initial_second_block_charge)[0]

    # Calcular revenue final
    df["PP_Bill_Month"] = (
        0 +
        df["Average"].clip(upper=inputs.upper_limit) * inputs.upper_limit +
        (df["Average"] - inputs.upper_limit).clip(lower=0) * solution
    )
    df["PP_Bill_Annual"] = df["PP_Bill_Month"] * 12
    revenue_total_pp = df["PP_Bill_Annual"].sum()

    return {
        "optimized_second_block_charge": solution,
        "revenue_total_pp": revenue_total_pp
    }
