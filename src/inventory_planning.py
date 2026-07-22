import numpy as np
import pandas as pd
from scipy.stats import norm


# Z-scores for common service levels (probability of NOT stocking out)
SERVICE_LEVEL_Z_SCORES = {
    "90%": 1.28,
    "95%": 1.65,
    "97.5%": 1.96,
    "99%": 2.33,
}


def calculate_safety_stock(
    daily_demand: pd.Series,
    lead_time_days: int = 7,
    service_level: str = "95%",
) -> float:
    """
    Safety stock = Z * std(daily demand) * sqrt(lead time in days)

    Higher service level -> higher Z -> more safety stock -> less stockout risk,
    but more tied-up capital. This tradeoff is the whole point of the metric.
    """
    z = SERVICE_LEVEL_Z_SCORES.get(service_level, 1.65)
    daily_std = daily_demand.std()
    return float(z * daily_std * np.sqrt(lead_time_days))


def calculate_reorder_point(
    daily_demand: pd.Series,
    lead_time_days: int = 7,
    service_level: str = "95%",
) -> dict:
    """
    Reorder point = (average daily demand * lead time) + safety stock

    Once inventory drops to this level, a new order should be placed --
    it will arrive (on average) just as stock would otherwise run out.
    """
    avg_daily_demand = daily_demand.mean()
    safety_stock = calculate_safety_stock(daily_demand, lead_time_days, service_level)
    reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

    return {
        "avg_daily_demand": round(avg_daily_demand, 1),
        "lead_time_days": lead_time_days,
        "service_level": service_level,
        "safety_stock_units": round(safety_stock, 1),
        "reorder_point_units": round(reorder_point, 1),
    }


def recommend_order_quantity(
    forecast: pd.Series,
    current_inventory: float,
    reorder_point: float,
) -> dict:
    """
    Simple decision layer: should we order now, and if so, how much to cover
    the forecasted demand over the next period?
    """
    should_reorder = current_inventory <= reorder_point
    forecasted_demand_over_period = float(forecast.sum())

    recommended_qty = max(0, forecasted_demand_over_period - current_inventory) if should_reorder else 0

    return {
        "current_inventory": round(current_inventory, 1),
        "reorder_point": round(reorder_point, 1),
        "should_reorder_now": should_reorder,
        "forecasted_demand_over_period": round(forecasted_demand_over_period, 1),
        "recommended_order_quantity": round(recommended_qty, 1),
    }


if __name__ == "__main__":
    from src.preprocessing import load_raw_data, build_category_series
    from src.forecasting import forecast_future

    df = load_raw_data()
    daily_demand = build_category_series(df, "Electronics")

    reorder_info = calculate_reorder_point(daily_demand, lead_time_days=7, service_level="95%")
    print("=== Reorder Point Analysis: Electronics ===")
    for k, v in reorder_info.items():
        print(f"  {k}: {v}")

    weekly_demand = daily_demand.resample("W").sum()
    future = forecast_future(weekly_demand, horizon_days=28)  # next 4 weeks

    # Simulate a current inventory level for demonstration
    simulated_current_inventory = reorder_info["reorder_point_units"] * 0.8

    order_recommendation = recommend_order_quantity(
        forecast=future,
        current_inventory=simulated_current_inventory,
        reorder_point=reorder_info["reorder_point_units"],
    )
    print("\n=== Order Recommendation ===")
    for k, v in order_recommendation.items():
        print(f"  {k}: {v}")