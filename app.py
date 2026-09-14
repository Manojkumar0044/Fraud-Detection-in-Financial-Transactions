import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🛡️", layout="wide")

df = pd.read_csv("data/financial_transactions.csv")

st.title("🛡️ Fraud Detection in Financial Transactions")
st.caption("Codec Technologies Internship Project")

total = len(df)
fraud = int(df["is_fraud"].sum())
legit = total - fraud
fraud_rate = fraud / total * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Fraud Transactions", f"{fraud:,}")
c3.metric("Legitimate Transactions", f"{legit:,}")
c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

st.divider()

left, right = st.columns(2)

with left:
    fig = px.pie(
        df,
        names="is_fraud",
        title="Fraud vs Legitimate Transactions",
        labels={"is_fraud": "Class"}
    )
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with right:
    type_rate = df.groupby("transaction_type", as_index=False)["is_fraud"].mean()
    type_rate["fraud_rate"] *= 100
    fig = px.bar(
        type_rate,
        x="transaction_type",
        y="is_fraud",
        title="Fraud Rate by Transaction Type",
        labels={"is_fraud": "Fraud Rate (%)", "transaction_type": "Transaction Type"}
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Transaction Amount Distribution")
fig = px.histogram(
    df, x="amount", color="is_fraud", nbins=50,
    title="Transaction Amount Distribution"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Fraud Activity by Hour")
hour_rate = df.groupby("hour", as_index=False)["is_fraud"].mean()
hour_rate["is_fraud"] *= 100
fig = px.line(
    hour_rate, x="hour", y="is_fraud", markers=True,
    labels={"is_fraud": "Fraud Rate (%)", "hour": "Hour of Day"},
    title="Fraud Rate by Transaction Hour"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("High-Risk Transactions")
threshold = st.slider("Minimum transaction amount", 0.0, float(df["amount"].quantile(0.99)), float(df["amount"].median()))
view = df[df["amount"] >= threshold].sort_values("amount", ascending=False)
st.dataframe(view.head(100), use_container_width=True)

st.info("This dashboard is for educational analysis. It is not a production banking fraud-decision system.")
