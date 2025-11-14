import pandas as pd
import pytest
from pathlib import Path
from project import load_transactions


def test_load_transactions_multi_file(tmp_path: Path):

    csv1 = """Date,Description,Amount\n01/05/2025,HEB #1234,$-45.67\n01/06/2025,Payroll,"$1,200.00"\n"""
    csv2 = "date,description,amount\n2025-01-07,Starbucks,(5.25)\n01/07/2025,Exxon,-30.00\n"

    f1 = tmp_path / "a.csv"; f1.write_text(csv1)
    f2 = tmp_path / "b.csv"; f2.write_text(csv2)

    df = load_transactions([str(f1), str(f2)])

    assert len(df) == 4

    assert pytest.approx(float(df.loc[0, "amount"]), 0.001) == -45.67
    assert pytest.approx(float(df.loc[1, "amount"]), 0.001) == 1200.00
    assert pytest.approx(float(df.loc[2, "amount"]), 0.001) == -5.25


    assert pd.api.types.is_datetime64_any_dtype(df["date"])

    assert "category" in df.columns
    assert "account" in df.columns

def test_clean_transactions_category_mapping(tmp_path: Path):
    from project import clean_transactions
    import json

    csv =(
    "Date,Description,Amount\n"
    "01/05/2025,HEB #1234,$-45.67\n"
    "01/06/2025,Payroll,$1200.00\n"
    "01/07/2025,Starbucks,-5.25\n"
    )

    f = tmp_path / "mini.csv"

    f.write_text(csv)

    df = pd.read_csv(f)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    rules = {
        "groceries": ["heb"],
        "income": ["payroll"],
        "eating_out": ["starbucks"],
        "misc": [".*"]
    }

    cleaned = clean_transactions(df, rules)

    assert "month" in cleaned.columns
    assert cleaned["month"].iloc[0] == "2025-01"
    assert set(cleaned["category"]) == {"groceries", "income", "eating_out"}

def test_aggregate_by_category_simple(tmp_path: Path):
    from project import aggregate_by_category, clean_transactions

    csv = (
        "Date,Description,Amount,Category\n"
        "01/05/2025,HEB #1234,$-45.00,\n"
        "01/06/2025,Payroll,$1200.00,\n"
        "01/07/2025,Starbucks,-5.99,\n"
        "01/08/2025,Starbucks,-2.50,\n"
    )

    f = tmp_path / "agg.csv"
    f.write_text(csv)

    df = pd.read_csv(f, dtype=str)
    df.columns = [c.lower() for c in df.columns]

    raw = df["amount"].astype(str)
    had_parens = raw.str.contains(r"\(", regex =True)
    cleaned_amt = (
        raw.str.replace(r"[\$,]", "", regex=True)
        .str.replace(r"[()]", "", regex=True)
        .str.strip()
    )

    df["amount"] = pd.to_numeric(cleaned_amt, errors="coerce")
    df.loc[had_parens, "amount"] = -df.loc[had_parens, "amount"].abs()
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    rules = {
        "groceries": ["heb"],
        "income": ["payroll"],
        "eating_out": ["starbucks"],
        "misc": [".*"],
    }

    cleaned = clean_transactions(df, rules)
    totals = aggregate_by_category(cleaned)

    assert pytest.approx(totals["income"], 0.001) == 1200.00
    assert pytest.approx(totals["groceries"], 0.001) == -45.00
    assert pytest.approx(totals["eating_out"], 0.001) == -8.49

    assert set(totals.keys()) >= {"income", "groceries", "eating_out"}

def test_generate_budget_alerts_boundary():
    from project import generate_budget_alerts

    cat_totals = {
        "groceries": -300.00,
        "eating_out": -121.00,
        "income": 1200.00
    }

    budgets = {
        "groceries": 300.00,
        "eating_out": 120.00
    }

    alerts = generate_budget_alerts(cat_totals, budgets)

    assert not any(a["category"] == "groceries" for a in alerts)

    eo = next(a for a in alerts if a["category"] == "eating_out")
    assert eo["budget"] == 120.0
    assert eo["total"] == -121.0
    assert eo["over_by"] == 1.0


def _mk_df_for_month_tests():
    data = {
        "date": pd.to_datetime(["01/05/2025", "01/06/2025", "01/07/2025", "02/02/2025"]),
        "description": ["HEB", "Payroll", "Starbucks", "HEB"],
        "amount": [-45.0, 1200.0, -5.0, -60.0],
        "category": ["groceries", "income", "eating_out", "groceries"],
    }

    df = pd.DataFrame(data)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df

def test_aggregate_by_month_category_pivot():
    from project import aggregate_by_month_category
    df = _mk_df_for_month_tests()
    pivot = aggregate_by_month_category(df)

    assert set(pivot["month"]) == {"2025-01", "2025-02"}
    jan = pivot.loc[pivot["month"]=="2025-01", "groceries"].iloc[0]
    feb = pivot.loc[pivot["month"]=="2025-02", "groceries"].iloc[0]

    assert jan == -45.0
    assert feb == -60.0


def test_compute_income_expense_by_month():
    from project import compute_income_expense_by_month
    df = _mk_df_for_month_tests()
    ie = compute_income_expense_by_month(df)
    jan = ie[ie["month"]=="2025-01"].iloc[0]
    feb = ie[ie["month"]=="2025-02"].iloc[0]
    assert jan["income"] == 1200.0
    assert jan["expenses"] == -50.0
    assert jan["net"] == 1150.0
    assert feb["income"] == 0.0
    assert feb["expenses"] == -60.0
    assert feb["net"] == -60.0
