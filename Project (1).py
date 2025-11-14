import argparse
import logging
from pathlib import Path
from typing import List, Dict
import re
import json
import os
import glob
import sys

import numpy as np
import pandas as pd

def load_transactions(file_paths: List[str]) -> pd.DataFrame:
    required = {"date", "description", "amount"}
    frames = []

    for p in file_paths:
        path = Path(p)
        if not path.exists() or not path.is_file():
            raise ValueError(f"Input file not found: {p}")

        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]

        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Missing required columns in {p}: {sorted(missing)}")

        raw = df["amount"].astype(str)
        had_parens = raw.str.contains(r"\(", regex=True)
        cleaned = (
            raw.str.replace(r"[\$,]", "", regex=True)
            .str.replace(r"[()]", "", regex=True)
            .str.strip()
        )

        df["amount"] = pd.to_numeric(cleaned, errors="coerce")
        df.loc[had_parens, "amount"] = -df.loc[had_parens, "amount"].abs()

        d1 = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
        d2 = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = d1.fillna(d2)

        if "category" not in df.columns:
            df["category"] = np.nan
        if "account" not in df.columns:
            df["account"] = np.nan

        frames.append(df)

    if not frames:
        cols = ["date", "description", "amount", "category", "account"]
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)

def clean_transactions(
        df: "pd.DataFrame",
        rules: Dict[str, list] | None = None,
        prefer_source_category: bool = True
) -> "pd.DataFrame":

    if df.empty:
        return df.copy()
    df = df.drop_duplicates().reset_index(drop=True)

    def normalize_merchant(s: str) -> str:
        s = str(s).lower().strip()
        s = re.sub(r"#\d+", "", s)
        s = re.sub(r"-\s*\d+", "", s)
        s = re.sub(r"[^a-z0-9\s]", " ", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    df["merchant_clean"] = df["description"].map(normalize_merchant)

    if rules is None:
        try:
            with open("categories.json", "r") as f:
                rules = json.load(f)
        except FileNotFoundError:
            rules= {"misc": [".*"]}

    compiled = []
    for cat, pats in (rules or {}).items():
        cps = []
        for p in pats:
            try:
                cps.append(re.compile(p, re.I))
            except re.error:
                continue
        compiled.append((cat, cps))

    placeholder_vals = {"", "uncategorized", "unassigned", "pending", "misc", None}
    categories = []

    for _, row in df.iterrows():
        base_cat = row.get("category")
        has_source = pd.notna(base_cat) and str(base_cat).strip().lower() not in placeholder_vals

        if prefer_source_category and has_source:
            categories.append(base_cat)
            continue

        assigned = "misc"
        merchant = row["merchant_clean"]
        for cat, patterns in compiled:
            if any(p.search(merchant) for p in patterns):
                assigned = cat
                break
        categories.append(assigned)

    df["category"] = categories

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)

    return df

def aggregate_by_category(df: "pd.DataFrame") -> Dict[str, float]:

    if df.empty:
        return{}
    if "category" not in df.columns:
        raise ValueError("DataFrame missing 'category' column. Did you run clean_transactions()?")

    grouped = (
        df.dropna(subset=["amount"])
        .groupby("category", dropna=False)["amount"]
        .sum()
        .to_dict()
    )

    grouped = dict(sorted(grouped.items(), key=lambda kv: abs(kv[1]), reverse=True))
    return grouped

def generate_budget_alerts(cat_totals: Dict[str, float], budgets: Dict[str, float]) -> list[dict]:
    alerts: list[dict] = []
    for cat, cap in (budgets or {}).items():
        total = float(cat_totals.get(cat, 0.0))
        spend = abs(total)
        if spend > float(cap):
            alerts.append({
                "category": cat,
                "total": total,
                "budget": float(cap),
                "over_by": round(spend - float(cap), 2)
            })
    return alerts

def aggregate_by_month_category(df: "pd.DataFrame") -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()
    if "month" not in df.columns:
        df = df.copy()
        df["month"] = df["date"].dt.to_period("M").astype(str)
    pivot = df.pivot_table(
        index="month", columns="category", values="amount", aggfunc="sum", fill_value=0.0
    ).sort_index()

    pivot.columns = [str(c) for c in pivot.columns]
    pivot = pivot.reset_index()
    return pivot

def compute_income_expense_by_month(df: "pd.DataFrame") -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame(columns=["month", "income", "expenses", "net"])

    if "month" not in df.columns:
        df = df.copy()
        df["month"] = df["date"].dt.to_period("M").astype(str)


    g = df.groupby("month")["amount"]
    income = g.apply(lambda s: s[s > 0].sum()).rename("income")
    expenses = g.apply(lambda s: s[s < 0].sum()).rename("expenses")

    out = pd.concat([income, expenses], axis=1).fillna(0.0).reset_index()
    out["net"] = out["income"] + out["expenses"]
    return out.sort_values("month")

def _collect_paths(inputs: List[str] | None, folder: str | None) -> List[str]:
    if inputs and folder:
        raise ValueError("Use either --inputs or --folder, not both.")
    if inputs:
        return inputs
    if folder:
        return sorted(glob.glob(os.path.join(folder, "*.csv")))
    raise ValueError("You must provide --inputs file1.csv ... or --folder /path/to/csv(s)")

def _detect_csvs_current_dir() -> list[str]:
    paths = sorted(glob.glob(os.path.join(".", "*.csv")))

    seen = set()
    out = []

    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def interactive_mode() -> None:
    print("Welcome to the Expense & Budget Analyzer")
    print("Please drag and drop your transactions csv into project folder.")
    print("press Enter when ready; or type a file path (or paths, comma-separated).")

    while True:
        input("Press Enter to scan for CSV files... ")
        csv_paths = _detect_csvs_current_dir()

        if csv_paths:
            break
        print("No CSV files found in this folder. Please add at least one .csv and try again.\n")

    budgets_path = "budgets.json" if os.path.exists("budgets.json") else None

    rules = None

    if os.path.exists("categories.json"):
        try:
            with open("categories.json", "r") as f:
                rules = json.load(f)
        except Exception:
            print("Warning: categories.json exists but is invalid JSON. Using default rules.")
            rules = None

    print("\n== Preflight ==")
    print("Transactions (using all CSVs found):")
    for p in csv_paths:
        print(f" - {p}")
    if budgets_path:
        print(f"Budgets: {budgets_path} (found)")
    else:
        print("Budgets: (none) - budget alerts will show as None")
    print("Categories: categories.json" + (" (found)" if rules else " (default rules)"))

    df = load_transactions(csv_paths)
    cleaned = clean_transactions(df, rules)
    cleaned = cleaned.dropna(subset=["date"])

    cat_totals = aggregate_by_category(cleaned)
    income_expense = compute_income_expense_by_month(cleaned)

    budgets = None
    if budgets_path:
        try:
            with open(budgets_path, "r") as f:
                budgets = json.load(f)
        except Exception:
            print("Warning: budgets.json exists but is invalid JSON. Alerts will be skipped.")
            budgets = None

    alerts = generate_budget_alerts(cat_totals, budgets) if budgets else []

    def _print_totals():
        print("\n== Category Totals ==")
        if not cat_totals:
            print("(none)")
        else:
            rows = [[cat, _fmt_num(amt)] for cat, amt in cat_totals.items()]
            print(render_table(rows, ["Category", "Total"]))

    def _print_top():
        print("\n== Spend Bar (Top 10 by |amount|) ==")
        expenses_only = [(cat, amt) for cat, amt in cat_totals.items() if amt < 0]
        if not expenses_only:
            print("(no expenses)")
            return
        top_n = expenses_only[:10]
        print(ascii_bar(top_n))

    def _print_iem():
        print("\n== Income / Expense / Net by Month ==")
        if income_expense.empty:
            print("(none)")
        else:
            rows = [
                [r["month"], _fmt_num(r["income"]), _fmt_num(r["expenses"]), _fmt_num(r["net"])]
                for _, r in income_expense.iterrows()
            ]
            print(render_table(rows, ["Month", "Income", "Expenses", "Net"]))

    def _print_alerts():
        print("\n== Budget Alerts ==")
        if alerts:
            rows = [
                [a["category"], _fmt_num(a["total"]), _fmt_num(a["budget"]), _fmt_num(a["over_by"])]
                for a in alerts
            ]
            print(render_table(rows, ["Category", "Total", "Budget", "Over By"]))
        else:
            print("None - you are within your spending limit! ✅")

    while True:
        print(
            "\nChoose an option:\n"
            "  [s] Summary\n"
            "  [t] Category Totals\n"
            "  [o] Top 10 Bar\n"
            "  [i] Income / Expense / Net by Month\n"
            "  [b] Budget Alerts\n"
            "  [q] Quit"
        )

        cmd = input("> ").strip().lower()
        if cmd == "q":
            print("Goodbye!")
            return
        elif cmd == "s":
            _print_totals()
            _print_top()
            _print_iem()
            _print_alerts()
        elif cmd == "t":
            _print_totals()
        elif cmd == "o":
            _print_top()
        elif cmd == "i":
            _print_iem()
        elif cmd == "b":
            _print_alerts()
        else:
            print("Unknown command. Please choose one of: s, t, o, i, b, q.")



def main() -> None:
    parser = argparse.ArgumentParser(description="Expense & Budget Analyzer")
    mx = parser.add_mutually_exclusive_group(required=True)
    mx.add_argument("--inputs", nargs="+", help="CSV file(s) to load")
    mx.add_argument("--folder", help="Folder containing CSV files (glob: *.csv)")

    parser.add_argument("--rules", default="categories.json",
                        help="Path to categories JSON (default: categories.json)")
    parser.add_argument("--budgets", default=None,
                        help="Path to budgets JSON (optional)")
    parser.add_argument("--outdir", default=None,
                        help="Write outputs (CSV/JSON) to this folder if provided")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    paths = _collect_paths(args.inputs, args.folder)
    logging.info("Found %d CSV file(s).", len(paths))

    df = load_transactions(paths)
    logging.info("Loaded %d rows; columns: %s", len(df), list(df.columns))

    rules = None
    if args.rules and os.path.exists(args.rules):
        try:
            with open(args.rules, "r") as f:
                rules = json.load(f)
        except Exception as e:
            logging.warning("Could not read rules file '%s' : %s", args.rules, e)

    cleaned = clean_transactions(df, rules)
    cleaned = cleaned.dropna(subset=["date"])

    cat_totals = aggregate_by_category(cleaned)
    month_cat = aggregate_by_month_category(cleaned)
    income_expense = compute_income_expense_by_month(cleaned)


    budgets = None
    if args.budgets:
        if os.path.exists(args.budgets):
            try:
                with open(args.budgets, "r") as f:
                    budgets = json.load(f)
            except Exception as e:
                logging.warning("Could not read budgets file '%s' : %s", args.budgets, e)
        else:
            logging.warning("Budgets file not found: %s", args.budgets)

    alerts = generate_budget_alerts(cat_totals, budgets) if budgets else []

    print("\n== Category Totals ==")
    if not cat_totals:
        print("(none)")
    else:
        rows = [[cat, _fmt_num(amt)] for cat, amt in cat_totals.items()]
        print(render_table(rows, ["Category", "Total"]))

    top_n = list(cat_totals.items())[:10]

    expenses_only = [(cat, amt) for cat, amt in cat_totals.items() if amt < 0]
    print("\n== Spend Bar (Top 10 by |amount|) ==")
    if expenses_only:
        top_n = expenses_only[:10]
        print(ascii_bar(top_n))
    else:
        print("(no expenses)")

    print("\n== Income / Expense / Net by Month ==")
    if income_expense.empty:
        print("(none)")
    else:
        rows = [
            [r["month"], _fmt_num(r["income"]), _fmt_num(r["expenses"]), _fmt_num(r["net"])]
            for _, r in income_expense.iterrows()
        ]
        print(render_table(rows, ["Month", "Income", "Expenses", "Net"]))

    print("\n== Budget Alerts ==")

    if alerts:
        rows = [
            [a["category"], _fmt_num(a["total"]), _fmt_num(a["budget"]), _fmt_num(a["over_by"])]
            for a in alerts
        ]
        print(render_table(rows, ["Category", "Total", "Budget", "Over By"]))
    else:
        print("None")

def _fmt_num(x: float) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

def render_table(rows: list[list[str]], headers: list[str]) -> str:

    data = [headers] + rows

    widths = [max(len(str(row[i])) for row in data) for i in range (len(headers))]

    def is_num(s: str) -> bool:
        s = str(s).replace(",", "").replace(".", "", 1).lstrip("-")
        return s.isdigit()

    def fmt_row(row: list[str]) -> str:
        cells = []
        for i, val in enumerate(row):
            sval = str(val)
            if is_num(sval):
                cells.append(sval.rjust(widths[i]))
            else: cells.append(sval.ljust(widths[i]))
        return "  ".join(cells)

    sep = ["-" * w for w in widths]
    out = [fmt_row(headers), fmt_row(sep)]
    out += [fmt_row(r) for r in rows]
    return "\n".join(out)

def ascii_bar(series: list[tuple[str, float]], width: int = 40) -> str:
    if not series:
        return ""
    maxv = max(abs(v) for _, v in series) or 1.0
    lines = []
    for k, v in series:
        n = int(abs(v) /maxv * width)
        lines.append(f"{k:15} | " + ("#" * n))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()



