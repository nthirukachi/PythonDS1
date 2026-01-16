# Problem Statement

## 1. The Real-World Challenge
When we build Machine Learning models in a lab (Jupyter Notebook), the data is static. We train, test, and if accuracy is high, we celebrate.

However, in the **Real World (Production)**, data is **ALIVE**. It changes over time.
- **Inflation** changes prices (Income data shifts).
- **Policies** change rules (Loan approval logic shifts).
- **Sensors** break (Null values appear).

If we deploy a model and forget about it, it will eventually fail. We need a **Monitoring System** to watch over it, just like a security guard watches a bank.

## 2. The Objective
We need to design a robust monitoring plan for a deployed ML system.
We must distinguish between:
1.  **Data Drift ($P(X)$)**: The input data has changed (e.g., everyone is richer).
2.  **Concept Drift ($P(Y|X)$)**: The rules of the world have changed (e.g., bank requires higher income for loans).

## 3. The Solution Strategy
We will build a Python system that:
1.  **Generates Synthetic Data**: Simulates normal days, drift days, and rule-change days.
2.  **Runs Data Quality Checks**: Ensures data isn't broken (Nulls, negative values).
3.  **Runs Drift Checks**: Uses statistics (KS Test) to detect if input patterns changed.
4.  **Triggers Alerts**: Tells the MLOps engineer exactly what to do.

## 4. Success Criteria
- A system that **passes** on normal data.
- **Alerts** on Data Drift.
- **Explains** why it might miss Concept Drift if we assume we don't have immediate labels.
