#Tariff Design
from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

app = FastAPI()

months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

@app.post("/str1")
async def calculate_str1_from_csv(
    file: UploadFile = File(...),
    upper_limit: float = Form(...),
    customer_charge: float = Form(...),
    first_block: float = Form(...),
    second_block: float = Form(...)
):
    df = pd.read_csv(file.file)

    for month in months:
        df[f"STR1_{month}"] = (
            customer_charge +
            (df[month].clip(upper=upper_limit) * first_block) +
            ((df[month] - upper_limit).clip(lower=0) * second_block)
        )

    df["SumSTR1"] = df[[f"STR1_{month}" for month in months]].sum(axis=1)
    total_revenue = df["SumSTR1"].sum()

    return {"total_revenue": total_revenue}

@app.post("/str2")
async def calculate_str2_from_csv(
    file: UploadFile = File(...),
    upper_limit_std: float = Form(...),
    customer_charge_std: float = Form(...),
    first_block_std: float = Form(...),
    second_block_std: float = Form(...),
    sb_aspercentageoffb: float = Form(...),
    upper_limit_bs: float = Form(...),
    customer_charge_bs: float = Form(...),
    first_block_bs: float = Form(...),
    break_even: float = Form(...),
    revenue_target: float = Form(...)
):
    df = pd.read_csv(file.file)
    df["Average"] = df[months].mean(axis=1)
    is_standard = df["Average"] > break_even

    def objective(x):
        first_std = x[0]
        second_std = second_block_std*sb_aspercentageoffb

        for month in months:
            billing_std = (
                customer_charge_std +
                (df[month].clip(upper=upper_limit_std) * first_std) +
                ((df[month] - upper_limit_std).clip(lower=0) * second_std)
            )

            billing_bs = (
                customer_charge_bs +
                (df[month].clip(upper=upper_limit_bs) * first_block_bs) +
                ((df[month] - upper_limit_bs).clip(lower=0) * first_block_bs)
            )

            df[f"STR2_{month}"] = billing_std.where(is_standard, billing_bs)

        df["SumSTR2"] = df[[f"STR2_{month}" for month in months]].sum(axis=1)
        return df["SumSTR2"].sum() - revenue_target

    solution = fsolve(objective, [first_block_std])[0]

    for month in months:
        billing_std = (
            customer_charge_std +
            (df[month].clip(upper=upper_limit_std) * solution) +
            ((df[month] - upper_limit_std).clip(lower=0) * second_block_std)
        )

        billing_bs = (
            customer_charge_bs +
            (df[month].clip(upper=upper_limit_bs) * first_block_bs) +
            ((df[month] - upper_limit_bs).clip(lower=0) * first_block_bs)
        )

        df[f"STR2_{month}"] = billing_std.where(is_standard, billing_bs)

    df["SumSTR2"] = df[[f"STR2_{month}" for month in months]].sum(axis=1)
    total_revenue = df["SumSTR2"].sum()

    return {
        "first_block_std_final": solution,
        "revenue_total_str2_final": total_revenue
    }

@app.post("/demand_total")
async def calculate_total_demand(
    file: UploadFile = File(...),
    demand_file: UploadFile = File(...),
    power_factor: float = Form(...)
):
    df = pd.read_csv(file.file)
    demand_lookup = pd.read_csv(demand_file.file)
    demand_lookup = demand_lookup.sort_values(by="consumption_kWh", ascending=False)
    demand_dict = demand_lookup.set_index("consumption_kWh")["demand_kW"].to_dict()
    consumption_values = list(demand_dict.keys())

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
        df.loc[df[f"demand_{month}"] > 0, f"demand_{month}"] /= power_factor

    df["demand_Sum"] = df[[f"demand_{month}" for month in months]].sum(axis=1)
    total_demand = df["demand_Sum"].sum()

    return {"total_demand_estimated": total_demand}

@app.post("/dprepaid")
async def calculate_dprepaid(
    file: UploadFile = File(...),
    upper_limit: float = Form(...),
    early_payment_incentive: float = Form(...),
    revenue_target: float = Form(...),
    initial_second_block_charge: float = Form(...),
    str2_customer_charge_std: float = Form(...),
    str2_first_block_std: float = Form(...),
    str2_first_block_bs: float = Form(...)
):
    df = pd.read_csv(file.file)
    df["Average"] = df[months].mean(axis=1)
    customer_charge = 0

    pp_first_block = ((str2_customer_charge_std - early_payment_incentive) / upper_limit +
                      (str2_first_block_std + str2_first_block_bs) / 2)

    def objective(x):
        second_block = x[0]
        df["PP_Bill_Month"] = (
            customer_charge +
            df["Average"].clip(upper=upper_limit) * pp_first_block +
            (df["Average"] - upper_limit).clip(lower=0) * second_block
        )
        df["PP_Bill_Annual"] = df["PP_Bill_Month"] * 12
        return revenue_target - df["PP_Bill_Annual"].sum()

    solution = fsolve(objective, [initial_second_block_charge])[0]

    df["PP_Bill_Month"] = (
        customer_charge +
        df["Average"].clip(upper=upper_limit) * pp_first_block +
        (df["Average"] - upper_limit).clip(lower=0) * solution
    )
    df["PP_Bill_Annual"] = df["PP_Bill_Month"] * 12
    revenue_total = df["PP_Bill_Annual"].sum()

    return {
        "pp_first_block": pp_first_block,
        "pp_second_block": solution,
        "revenue_total_pp": revenue_total
    }
