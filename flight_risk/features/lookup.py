"""
flight_risk.features.lookup
===========================
Historical delay lookup tables (per route and per airline × hour) plus the
join that enriches the main dataframe with them.

Both tables are persisted (``route_stats.pkl`` / ``airline_hour_stats.pkl``)
so they can be reused at inference time without recomputing on the full
dataset.

NOTE
----
These statistics are computed over the *entire* dataframe before any temporal
split, then joined back onto every row. This leaks information from the test
period (future) into the training rows. If you want test metrics to reflect
real-world performance, compute the tables on the training slice only and join
them onto val/test as a lookup. Kept as-is here to preserve the original
behaviour of the pipeline.
"""

import pandas as pd


def build_route_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical delay statistics aggregated by route.

    Used both to enrich the training dataset and as a lookup table at
    inference time (saved as ``route_stats.pkl``). For unseen routes, callers
    should fall back to the global mean of each statistic.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe containing ``route``, ``dep_delay_min``, and
        ``dep_is_delayed`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``route`` with columns:

        * ``route_hist_delay_mean`` — mean departure delay in minutes.
        * ``route_hist_delay_std``  — standard deviation of departure delay.
        * ``route_hist_delay_rate`` — fraction of flights delayed > 15 min.
    """
    stats = df.groupby("route")["dep_delay_min"].agg(
        route_hist_delay_mean="mean",
        route_hist_delay_std="std",
    )
    stats["route_hist_delay_rate"] = df.groupby("route")["dep_is_delayed"].mean()
    return stats


def build_airline_hour_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical delay statistics aggregated by airline and departure hour.

    Captures the punctuality profile of each airline at each hour of the day.
    Used both to enrich the training dataset and as a lookup table at inference
    time (saved as ``airline_hour_stats.pkl``).

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe containing ``airline_icao``, ``dep_hour``,
        ``dep_is_delayed``, and ``dep_delay_min`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``(airline_icao, dep_hour)`` with columns:

        * ``airline_hour_delay_rate`` — fraction of flights delayed > 15 min.
        * ``airline_hour_delay_mean`` — mean departure delay in minutes.
    """
    stats = df.groupby(["airline_icao", "dep_hour"]).agg(
        airline_hour_delay_rate=("dep_is_delayed", "mean"),
        airline_hour_delay_mean=("dep_delay_min",  "mean"),
    ).round(4)
    return stats


def join_lookup_tables(
    df: pd.DataFrame,
    route_stats: pd.DataFrame,
    airline_hour_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join both lookup tables onto the main dataframe.

    Missing values that arise from unseen keys are filled with the global mean
    of each statistic, so the output contains no NaN in these columns.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched dataframe with ``route``, ``airline_icao``, and ``dep_hour``
        columns.
    route_stats : pd.DataFrame
        Output of :func:`build_route_stats`.
    airline_hour_stats : pd.DataFrame
        Output of :func:`build_airline_hour_stats`.

    Returns
    -------
    pd.DataFrame
        Dataframe with six additional historical statistics columns.
    """
    df = df.join(route_stats, on="route")
    df["route_hist_delay_mean"].fillna(route_stats["route_hist_delay_mean"].mean(), inplace=True)
    df["route_hist_delay_std"].fillna(route_stats["route_hist_delay_std"].mean(),   inplace=True)
    df["route_hist_delay_rate"].fillna(route_stats["route_hist_delay_rate"].mean(), inplace=True)

    df = df.join(airline_hour_stats, on=["airline_icao", "dep_hour"])
    df["airline_hour_delay_rate"].fillna(airline_hour_stats["airline_hour_delay_rate"].mean(), inplace=True)
    df["airline_hour_delay_mean"].fillna(airline_hour_stats["airline_hour_delay_mean"].mean(), inplace=True)

    return df
