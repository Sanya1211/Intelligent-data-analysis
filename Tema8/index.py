import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isclose

# ---- Output directory (change if you want another path) ----
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# ---- Financial helper functions ----
def npv(discount_rate, cashflows):
    return sum(cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cashflows))

def irr(cashflows, guess=0.1, max_iter=1000):
    try:
        import numpy as _np
        # newer numpy may deprecate np.irr; try numpy_financial if available
        val = None
        if hasattr(_np, "irr"):
            val = _np.irr(cashflows)
        else:
            try:
                import numpy_financial as _nf
                val = _nf.irr(cashflows)
            except Exception:
                val = None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise Exception("irr not available or returned NaN")
        return float(val)
    except Exception:
        rate = guess
        for _ in range(max_iter):
            npv_val = sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cashflows))
            d_npv = sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cashflows) if i > 0)
            if isclose(d_npv, 0, abs_tol=1e-12):
                break
            new_rate = rate - npv_val / d_npv
            if abs(new_rate - rate) < 1e-8:
                return new_rate
            rate = new_rate
        return float('nan')

def payback_period(cashflows):
    cum = 0.0
    for i, cf in enumerate(cashflows):
        cum += cf
        if cum >= 0:
            prev_cum = cum - cf
            if isclose(cf, 0.0):
                return float(i)
            fraction = (0 - prev_cum) / cf if prev_cum < 0 else 0.0
            return i - 1 + fraction if i > 0 else 0.0
    return None

def lcoe(total_discounted_costs, total_discounted_energy):
    return total_discounted_costs / total_discounted_energy if total_discounted_energy > 0 else float('inf')

# ---- Parameters (example) ----
params = {
    "system_capacity_kw": 30.0,
    "system_cost_per_kw": 1000.0,
    "installation_cost": 500.0,
    "incentive_upfront": 3000.0,
    "annual_o_and_m_per_kw": 10.0,
    "tariff_per_kwh": 0.18,
    "capacity_factor": 0.12,
    "degradation_rate": 0.005,
    "project_life_years": 25,
    "discount_rate": 0.06,
    "inflation_rate": 0.02,
    "inverter_replace_cost": 2000.0,
    "inverter_replace_year": 12,
    "electricity_price_escalation": 0.01
}

# ---- Derived initial values ----
capex = params["system_capacity_kw"] * params["system_cost_per_kw"] + params["installation_cost"]
capex_after_incentive = capex - params["incentive_upfront"]

years = list(range(0, params["project_life_years"] + 1))
annual_production = []
annual_revenue = []
annual_om = []
annual_net = []

for y in years:
    if y == 0:
        annual_production.append(0.0)
        annual_revenue.append(0.0)
        annual_om.append(0.0)
        annual_net.append(-capex_after_incentive)
    else:
        degradation = (1 - params["degradation_rate"]) ** (y - 1)
        prod_kwh = params["system_capacity_kw"] * params["capacity_factor"] * 8760 * degradation
        tariff = params["tariff_per_kwh"] * ((1 + params["electricity_price_escalation"]) ** (y - 1))
        revenue = prod_kwh * tariff
        o_and_m = params["annual_o_and_m_per_kw"] * params["system_capacity_kw"]
        replacement = params["inverter_replace_cost"] if y == params["inverter_replace_year"] else 0.0
        net = revenue - o_and_m - replacement
        annual_production.append(prod_kwh)
        annual_revenue.append(revenue)
        annual_om.append(o_and_m + replacement)
        annual_net.append(net)

discount_rate = params["discount_rate"]
discount_factors = [(1 + discount_rate) ** (-y) for y in years]

total_discounted_costs = (-annual_net[0]) + sum(
    (params["annual_o_and_m_per_kw"] * params["system_capacity_kw"] + (params["inverter_replace_cost"] if y == params["inverter_replace_year"] else 0.0)) * discount_factors[y]
    for y in years if y > 0
)
total_discounted_energy = sum(annual_production[y] * discount_factors[y] for y in years if y > 0)

cashflows = annual_net.copy()

npv_value = npv(discount_rate, cashflows)
irr_value = irr(cashflows)
payback = payback_period(cashflows)
lcoe_value = lcoe(total_discounted_costs, total_discounted_energy)

df = pd.DataFrame({
    "year": years,
    "production_kwh": np.round(annual_production, 1),
    "revenue_eur": np.round(annual_revenue, 2),
    "o_and_m_plus_repl_eur": np.round(annual_om, 2),
    "net_cashflow_eur": np.round(annual_net, 2),
    "discount_factor": [round(d, 4) for d in discount_factors],
    "discounted_cashflow_eur": np.round([annual_net[i] * discount_factors[i] for i in range(len(years))], 2)
})

df["cumulative_cashflow_eur"] = df["net_cashflow_eur"].cumsum().round(2)

summary = {
    "System capacity (kW)": params["system_capacity_kw"],
    "Initial CAPEX (EUR)": round(capex, 2),
    "CAPEX after incentive (EUR)": round(capex_after_incentive, 2),
    "Project life (years)": params["project_life_years"],
    "Discount rate": params["discount_rate"],
    "NPV (EUR)": round(npv_value, 2),
    "IRR": round(irr_value, 4) if not np.isnan(irr_value) else None,
    "Payback (years, fractional)": round(payback, 2) if payback is not None else None,
    "LCOE (EUR/kWh)": round(lcoe_value, 4)
}

# Save plots to output_dir
cumcash_path = os.path.join(output_dir, "solar_economics_analysis_cumcash.png")
production_path = os.path.join(output_dir, "solar_economics_production.png")

plt.figure(figsize=(10, 5))
plt.plot(df["year"], df["cumulative_cashflow_eur"], marker='o')
plt.title("Cumulative cashflow over project life")
plt.xlabel("Year")
plt.ylabel("Cumulative cashflow (EUR)")
plt.grid(True)
plt.tight_layout()
plt.savefig(cumcash_path)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["year"], df["production_kwh"], marker='o')
plt.title("Annual energy production (kWh)")
plt.xlabel("Year")
plt.ylabel("Production (kWh)")
plt.grid(True)
plt.tight_layout()
plt.savefig(production_path)
plt.close()

# Save CSV to output_dir
csv_path = os.path.join(output_dir, "solar_economics_yearly.csv")
df.to_csv(csv_path, index=False)

# Print summary and file locations
print("Summary:")
for k, v in summary.items():
    print(f"- {k}: {v}")

print("\nFiles saved to:", os.path.abspath(output_dir))
print("-", csv_path)
print("-", cumcash_path)
print("-", production_path)

# Also expose results dict
results = {
    "params": params,
    "summary": summary,
    "yearly_df_path": csv_path,
    "plot_cumulative_path": cumcash_path,
    "plot_production_path": production_path
}

results
