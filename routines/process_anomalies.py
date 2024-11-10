"""Processes factor anomalies to be used in the RPPCA, originally sourced from https://sites.google.com/site/serhiykozak/data."""

if __name__ == "__main__":
    import pandas as pd
    import os

    fldr = f"{os.getcwd()}/data"
    flist = [x for x in os.listdir(f"{fldr}/sorts") if "ret10_" in x]

    collect_sorts = []
    for fname in flist:
        anom_label = fname.replace(".csv", "").replace("ret10_", "")
        raw = pd.read_csv(f"{fldr}/sorts/{fname}")
        raw["date"] = pd.DatetimeIndex(raw["date"])
        raw.set_index("date", inplace=True)
        raw = raw.resample("M").last()

        cols = pd.MultiIndex.from_tuples(
            [(anom_label, x) for x in raw.columns], names=["anomaly", "port"]
        )
        raw.columns = cols
        collect_sorts += [raw]

    sorts = pd.concat(collect_sorts, axis=1)
    sorts.to_pickle(f"{fldr}/sorts.pkl")
