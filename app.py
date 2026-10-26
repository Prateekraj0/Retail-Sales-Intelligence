
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Retail Sales Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path("models/retail_segmentation_model.pkl")

# =====================================================
# PREMIUM DARK THEME
# =====================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg: #080B14;
        --panel: rgba(19, 27, 46, 0.78);
        --border: rgba(148, 163, 184, 0.16);
        --text: #F8FAFC;
        --muted: #9CAAC0;
        --purple: #A78BFA;
        --cyan: #67E8F9;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 4%, rgba(109, 74, 255, .19), transparent 33%),
            radial-gradient(ellipse at 92% 8%, rgba(34, 211, 238, .11), transparent 28%),
            #080B14;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1220, #090D17);
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -.5px;
    }

    h1 {
        font-size: 2.6rem !important;
        font-weight: 700 !important;
    }

    p, label, li {
        color: #D0D9E8;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 36px 34px;
        border: 1px solid rgba(167, 139, 250, .25);
        border-radius: 24px;
        background:
            radial-gradient(ellipse at 90% 10%, rgba(103, 232, 249, .14), transparent 40%),
            linear-gradient(135deg, rgba(109, 74, 255, .16), rgba(15, 23, 42, .88) 55%, rgba(14, 32, 48, .92));
        box-shadow: 0 20px 70px rgba(0, 0, 0, .20);
        margin: 8px 0 28px;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: -90px;
        bottom: -140px;
        border: 1px solid rgba(103, 232, 249, .25);
        border-radius: 50%;
        box-shadow: 0 0 0 25px rgba(103, 232, 249, .025),
                    0 0 0 55px rgba(167, 139, 250, .025);
        pointer-events: none;
    }

    .hero-eyebrow {
        color: var(--cyan);
        text-transform: uppercase;
        letter-spacing: 2.5px;
        font-size: .72rem;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 3.4rem);
        font-weight: 700;
        line-height: 1.12;
        letter-spacing: -1.5px;
        color: #F8FAFC;
        margin: 0 0 16px;
    }

    .gradient-text {
        background: linear-gradient(90deg, #C4B5FD, #67E8F9);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #AAB8CD;
        font-size: 1rem;
        max-width: 650px;
        line-height: 1.8;
    }

    .hero-tag {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        border: 1px solid rgba(103, 232, 249, .25);
        background: rgba(103, 232, 249, .07);
        color: #A5F3FC;
        font-size: .75rem;
        margin-bottom: 22px;
    }

    .feature-card {
        background: linear-gradient(145deg, rgba(24, 34, 56, .92), rgba(13, 19, 33, .96));
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 23px;
        min-height: 190px;
        transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(167, 139, 250, .7);
        box-shadow: 0 14px 35px rgba(109, 74, 255, .12);
    }

    .card-icon {
        display: inline-flex;
        width: 43px;
        height: 43px;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        background: rgba(167, 139, 250, .13);
        border: 1px solid rgba(167, 139, 250, .2);
        font-size: 21px;
        margin-bottom: 16px;
    }

    .feature-card h3 {
        font-size: 1.05rem !important;
        margin: 0 0 10px;
    }

    .feature-card p {
        color: var(--muted);
        font-size: .88rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .section-kicker {
        color: var(--cyan);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(24, 34, 56, .9), rgba(13, 19, 33, .95));
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, .10);
    }

    [data-testid="stMetricLabel"] {
        color: #AAB8CD !important;
    }

    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }

    .stButton > button, .stDownloadButton > button {
        border: 1px solid rgba(167, 139, 250, .4);
        border-radius: 11px;
        background: linear-gradient(110deg, #7048F5, #8B5CF6);
        color: white;
        font-weight: 600;
        padding: .65rem 1rem;
        transition: transform .2s ease, filter .2s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        color: white;
        filter: brightness(1.12);
        transform: translateY(-2px);
    }

    [data-testid="stFileUploader"] {
        background: rgba(17, 26, 43, .55);
        border: 1px dashed rgba(167, 139, 250, .55);
        border-radius: 16px;
        padding: 14px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 13px;
        overflow: hidden;
    }

    div[data-testid="stPlotlyChart"] {
        background: rgba(16, 22, 37, .58);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 8px;
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
    }

    hr {
        border-color: var(--border);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 25px 20px;
        }

        .hero-title {
            font-size: 2rem;
        }

        .feature-card {
            min-height: auto;
        }
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def make_display_safe(data):
    result = data.copy()

    for column in result.columns:
        if result[column].dtype == "object":
            result[column] = result[column].map(
                lambda value: str(value) if pd.notna(value) else None
            )

    return result


def load_uploaded_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    return pd.read_excel(uploaded_file)


def find_column(data, candidates):
    """Find a column using common case-insensitive names."""
    column_map = {
        str(column).strip().lower(): column
        for column in data.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in column_map:
            return column_map[key]

    return None


def numeric_series(data, column):
    if column is None:
        return None

    return pd.to_numeric(data[column], errors="coerce")


def format_currency(value):
    if value is None or pd.isna(value):
        return "N/A"

    return f"₹{value:,.2f}"


def get_numeric_data(data):
    return data.select_dtypes(include=np.number)


def build_quick_findings(data):
    findings = []

    rows, columns = data.shape
    total_cells = rows * columns
    missing_count = int(data.isna().sum().sum())
    duplicate_count = int(data.duplicated().sum())

    missing_pct = (
        missing_count / total_cells * 100
        if total_cells else 0
    )

    findings.append(
        f"The dataset contains {rows:,} rows and {columns:,} columns."
    )

    findings.append(
        f"{missing_count:,} cells are missing ({missing_pct:.2f}% of all cells)."
    )

    findings.append(
        f"{duplicate_count:,} exact duplicate rows were found."
    )

    numeric_data = get_numeric_data(data)

    for column in numeric_data.columns:
        values = numeric_data[column].dropna()

        if len(values) < 4 or values.nunique() <= 1:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            count = int(
                (
                    (values < q1 - 1.5 * iqr)
                    | (values > q3 + 1.5 * iqr)
                ).sum()
            )

            if count:
                findings.append(
                    f"'{column}' has {count:,} potential outlier values."
                )

    if numeric_data.shape[1] >= 2:
        corr = numeric_data.corr(numeric_only=True)

        upper = corr.where(
            np.triu(
                np.ones(corr.shape), k=1
            ).astype(bool)
        )

        pairs = upper.stack().dropna()

        if not pairs.empty:
            pair = pairs.abs().idxmax()

            findings.append(
                f"The strongest absolute correlation is between "
                f"'{pair[0]}' and '{pair[1]}' "
                f"(r = {corr.loc[pair[0], pair[1]]:.3f})."
            )

    return findings


def build_rfm(data):
    """
    Convert transaction data into customer-level RFM data.

    Recency: days since the customer's last transaction.
    Frequency: distinct order count, when an order ID is available.
    Monetary: sum of the detected transaction amount.
    """
    customer_col = find_column(
        data,
        ["Customer_ID", "Customer ID", "CustomerID", "customer_id"]
    )

    date_col = find_column(
        data,
        [
            "Order_Date", "InvoiceDate", "Invoice_Date",
            "Transaction_Date", "Date", "Purchase_Date"
        ]
    )

    amount_col = find_column(
        data,
        [
            "Revenue_INR", "Revenue", "Sales",
            "TotalAmount", "Total_Amount", "Amount", "Monetary"
        ]
    )

    order_col = find_column(
        data,
        [
            "Order_ID", "InvoiceNo", "Invoice",
            "Transaction_ID", "TransactionID"
        ]
    )

    missing = []

    if customer_col is None:
        missing.append("customer ID")

    if date_col is None:
        missing.append("transaction date")

    if amount_col is None:
        missing.append("revenue or transaction amount")

    if missing:
        return None, (
            "Automatic RFM generation requires a "
            + ", ".join(missing)
            + " column."
        )

    selected_columns = [customer_col, date_col, amount_col]

    if order_col is not None:
        selected_columns.append(order_col)

    work = data[selected_columns].copy()

    work[date_col] = pd.to_datetime(
        work[date_col], errors="coerce"
    )

    work[amount_col] = pd.to_numeric(
        work[amount_col], errors="coerce"
    )

    work = work.dropna(
        subset=[customer_col, date_col, amount_col]
    )

    if work.empty:
        return None, (
            "No usable transaction rows remain after "
            "parsing customer IDs, dates, and amounts."
        )

    reference_date = (
        work[date_col].max().normalize()
        + pd.Timedelta(days=1)
    )

    grouped = work.groupby(customer_col, dropna=True)

    rfm = grouped.agg(
        LastPurchase=(date_col, "max"),
        Monetary=(amount_col, "sum")
    )

    if order_col is not None:
        rfm["Frequency"] = (
            work.groupby(customer_col)[order_col].nunique()
        )
    else:
        # This is a transaction-row count, not a verified order count.
        rfm["Frequency"] = grouped.size()

    rfm["Recency"] = (
        reference_date
        - rfm["LastPurchase"].dt.normalize()
    ).dt.days

    rfm = rfm[
        ["Recency", "Frequency", "Monetary"]
    ].reset_index()

    rfm = rfm.rename(
        columns={customer_col: "Customer_ID"}
    )

    return rfm, None


def plot_layout(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif"),
        margin=dict(l=20, r=20, t=65, b=20)
    )

    return fig


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
with st.sidebar:
    st.markdown("## 📊 RETAIL IQ")
    st.caption("DATA SCIENCE & CUSTOMER ANALYTICS")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Retail Analytics",
            "Upload Dataset",
            "Data Insights",
            "Customer Segmentation"
        ],
        key="navigation"
    )

    st.divider()
    st.caption("PROJECT")
    st.write("Retail Sales Intelligence")

    if "uploaded_data" in st.session_state:
        st.success(
            f"Dataset ready · "
            f"{len(st.session_state['uploaded_data']):,} rows"
        )
    else:
        st.caption("Upload a dataset to begin")


# =====================================================
# PAGE 1: OVERVIEW
# Header appears ONLY on this page
# =====================================================
if page == "Overview":

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">DATA INTELLIGENCE PLATFORM</div>
        <h1>Retail Sales Intelligence</h1>
        <div class="hero-subtitle">
            Turn raw data into meaningful discoveries.
            Analyze performance, uncover hidden patterns,
            and understand your customers—all in one workspace.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Welcome to your analytics workspace")

    st.write(
        "Upload your own data, explore retail performance, "
        "or discover patterns and data-quality issues."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🛍️ Retail Analytics</h3>
            <p>
                Explore revenue, orders, products, and customer
                behavior from your uploaded dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📂 Upload Your Data</h3>
            <p>
                Upload CSV or Excel files and inspect columns,
                data types, missing values, and statistics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🧠 Smart Insights</h3>
            <p>
                Explore correlations, distributions, potential
                outliers, and data-quality issues.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Workspace status")

    left, right = st.columns(2)

    with left:
        if "uploaded_data" in st.session_state:
            data = st.session_state["uploaded_data"]

            st.success(
                f"Dataset loaded: {data.shape[0]:,} rows × "
                f"{data.shape[1]:,} columns"
            )
        else:
            st.info(
                "No dataset uploaded yet. Select Upload Dataset "
                "from the sidebar."
            )

    with right:
        if MODEL_PATH.exists():
            st.success("Customer segmentation model file found.")
        else:
            st.info(
                "No segmentation model installed. "
                "General dataset analysis is still available."
            )

    st.caption(
        "Use public or synthetic data for testing. "
        "Do not present sample data as actual Oromist sales records."
    )


# =====================================================
# PAGE 2: RETAIL ANALYTICS
# =====================================================
elif page == "Retail Analytics":

    st.markdown("## 🛍️ Retail Analytics")

    if "uploaded_data" not in st.session_state:
        st.info(
            "Upload a CSV or Excel file first. "
            "Retail Analytics will use that dataset automatically."
        )
    else:
        data = st.session_state["uploaded_data"].copy()

        st.caption(
            f"Source: {st.session_state.get('uploaded_filename', 'Uploaded dataset')} "
            f"· {len(data):,} rows"
        )

        revenue_col = find_column(
            data,
            ["Revenue_INR", "Revenue", "Sales",
             "TotalAmount", "Total_Amount", "Amount"]
        )

        order_col = find_column(
            data,
            ["Order_ID", "InvoiceNo", "Invoice",
             "Transaction_ID", "TransactionID"]
        )

        customer_col = find_column(
            data,
            ["Customer_ID", "Customer ID", "CustomerID"]
        )

        quantity_col = find_column(
            data,
            ["Quantity", "Units_Sold", "Units", "Qty"]
        )

        date_col = find_column(
            data,
            ["Order_Date", "InvoiceDate", "Invoice_Date",
             "Transaction_Date", "Date", "Purchase_Date"]
        )

        category_col = find_column(
            data,
            ["Product_Category", "Category",
             "Product Type", "ProductType"]
        )

        location_col = find_column(
            data,
            ["Region", "State", "Country", "City"]
        )

        revenue = numeric_series(data, revenue_col)
        quantity = numeric_series(data, quantity_col)

        total_revenue = revenue.sum() if revenue is not None else None

        total_orders = (
            data[order_col].nunique(dropna=True)
            if order_col is not None
            else len(data)
        )

        total_customers = (
            data[customer_col].nunique(dropna=True)
            if customer_col is not None
            else None
        )

        total_units = (
            quantity.sum(min_count=1)
            if quantity is not None
            else None
        )

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Total Revenue",
            format_currency(total_revenue)
        )

        k2.metric(
            "Orders / Transactions",
            f"{total_orders:,}"
        )

        k3.metric(
            "Unique Customers",
            f"{total_customers:,}"
            if total_customers is not None
            else "N/A"
        )

        k4.metric(
            "Units Sold",
            f"{total_units:,.0f}"
            if total_units is not None and pd.notna(total_units)
            else "N/A"
        )

        if revenue_col is None:
            st.warning(
                "No recognised revenue column was found. "
                "Expected names include Revenue_INR, Revenue, Sales, or Amount."
            )

        # Charts
        left, right = st.columns(2)

        with left:
            if category_col and revenue is not None:
                chart_data = data[[category_col]].copy()
                chart_data["Revenue"] = revenue

                chart_data = (
                    chart_data.groupby(
                        category_col, dropna=False
                    )["Revenue"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(12)
                    .reset_index()
                )

                chart_data[category_col] = (
                    chart_data[category_col]
                    .fillna("(Missing)")
                    .astype(str)
                )

                fig = px.bar(
                    chart_data,
                    x=category_col,
                    y="Revenue",
                    title="Revenue by Category",
                    color="Revenue",
                    color_continuous_scale="Purples"
                )

                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Revenue"
                )

                st.plotly_chart(
                    plot_layout(fig),
                    use_container_width=True
                )
            else:
                st.info(
                    "Category revenue chart requires a category "
                    "column and a revenue column."
                )

        with right:
            if date_col and revenue is not None:
                trend = pd.DataFrame({
                    "Date": pd.to_datetime(
                        data[date_col], errors="coerce"
                    ),
                    "Revenue": revenue
                }).dropna(subset=["Date"])

                if not trend.empty:
                    trend["Month"] = (
                        trend["Date"]
                        .dt.to_period("M")
                        .astype(str)
                    )

                    trend = (
                        trend.groupby("Month", as_index=False)
                        ["Revenue"]
                        .sum()
                    )

                    fig = px.line(
                        trend,
                        x="Month",
                        y="Revenue",
                        markers=True,
                        title="Monthly Revenue Trend"
                    )

                    fig.update_traces(
                        line_color="#A78BFA"
                    )

                    fig.update_layout(
                        xaxis_title="",
                        yaxis_title="Revenue"
                    )

                    st.plotly_chart(
                        plot_layout(fig),
                        use_container_width=True
                    )
                else:
                    st.info(
                        "No valid dates were found for the trend chart."
                    )
            else:
                st.info(
                    "Monthly revenue trend requires a date and revenue column."
                )

        if location_col and revenue is not None:
            region_data = data[[location_col]].copy()
            region_data["Revenue"] = revenue

            region_data = (
                region_data.groupby(
                    location_col, dropna=False
                )["Revenue"]
                .sum()
                .sort_values(ascending=False)
                .head(12)
                .reset_index()
            )

            region_data[location_col] = (
                region_data[location_col]
                .fillna("(Missing)")
                .astype(str)
            )

            fig = px.bar(
                region_data,
                x=location_col,
                y="Revenue",
                title="Revenue by Region / Location",
                color="Revenue",
                color_continuous_scale="Teal"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        if category_col:
            st.markdown("### Most common product categories")

            top_categories = (
                data[category_col]
                .fillna("(Missing)")
                .astype(str)
                .value_counts()
                .head(10)
                .rename_axis("Category")
                .reset_index(name="Transaction Rows")
            )

            st.dataframe(
                top_categories,
                use_container_width=True
            )

        st.caption(
            "Metrics are inferred from column names. "
            "Confirm that these fields match your business definitions."
        )


# =====================================================
# PAGE 3: UPLOAD DATASET
# =====================================================
elif page == "Upload Dataset":

    st.markdown("## 📂 Upload Your Dataset")

    st.write(
        "Upload a CSV or Excel file to inspect its structure "
        "and power the analytics pages."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="dataset_uploader"
    )

    if uploaded_file is not None:
        try:
            data = load_uploaded_file(uploaded_file)

            if data.empty:
                st.warning("The uploaded file contains no data.")
            else:
                st.session_state["uploaded_data"] = data
                st.session_state["uploaded_filename"] = uploaded_file.name

        except Exception as error:
            st.error(f"Could not read the file: {error}")

    if "uploaded_data" in st.session_state:

        data = st.session_state["uploaded_data"]

        st.success(
            f"Successfully loaded: "
            f"{st.session_state.get('uploaded_filename', 'Dataset')}"
        )

        st.markdown("### Dataset overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", f"{data.shape[0]:,}")
        c2.metric("Columns", f"{data.shape[1]:,}")
        c3.metric(
            "Missing Cells",
            f"{int(data.isna().sum().sum()):,}"
        )
        c4.metric(
            "Duplicate Rows",
            f"{int(data.duplicated().sum()):,}"
        )

        st.markdown("### Data preview")

        st.dataframe(
            make_display_safe(data.head(100)),
            use_container_width=True
        )

        st.markdown("### Column information")

        column_info = pd.DataFrame({
            "Column": data.columns.astype(str),
            "Data Type": data.dtypes.astype(str).values,
            "Missing Values": data.isna().sum().values,
            "Unique Values": data.nunique().values
        })

        st.dataframe(
            column_info,
            use_container_width=True
        )

        st.markdown("### Numerical summary")

        numeric = get_numeric_data(data)

        if not numeric.empty:
            st.dataframe(
                numeric.describe().T.round(3),
                use_container_width=True
            )
        else:
            st.info("No numerical columns detected.")

        st.download_button(
            "⬇️ Download Dataset as CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="analyzed_dataset.csv",
            mime="text/csv"
        )

        if st.button("Remove uploaded dataset"):
            st.session_state.pop("uploaded_data", None)
            st.session_state.pop("uploaded_filename", None)
            st.rerun()

    else:
        st.info(
            "Upload a file above. Your preview and statistics "
            "will appear here."
        )


# =====================================================
# PAGE 4: AUTOMATED DATA INSIGHTS
# =====================================================
elif page == "Data Insights":

    st.markdown("## 🧠 Automated Data Insights")

    st.write(
        "Explore distributions, correlations, categorical patterns, "
        "potential outliers, and data quality."
    )

    if "uploaded_data" not in st.session_state:
        st.warning(
            "Please open Upload Dataset and upload a file first."
        )

    else:
        data = st.session_state["uploaded_data"].copy()
        numeric_data = get_numeric_data(data)

        # DATA QUALITY
        st.subheader("1. Data Quality")

        missing_count = int(data.isna().sum().sum())
        duplicate_count = int(data.duplicated().sum())
        total_cells = data.shape[0] * data.shape[1]

        missing_pct = (
            missing_count / total_cells * 100
            if total_cells else 0
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Missing Cells", f"{missing_count:,}")
        c2.metric("Duplicate Rows", f"{duplicate_count:,}")
        c3.metric("Missing Data", f"{missing_pct:.2f}%")

        missing_by_column = data.isna().sum()
        missing_by_column = missing_by_column[
            missing_by_column > 0
        ].sort_values(ascending=False)

        if not missing_by_column.empty:
            st.markdown("**Missing values by column**")
            st.bar_chart(missing_by_column)
        else:
            st.success("No missing cells detected.")

        # NUMERICAL ANALYSIS
        st.subheader("2. Numerical Analysis")

        if not numeric_data.empty:

            st.dataframe(
                numeric_data.describe().T.round(3),
                use_container_width=True
            )

            selected_column = st.selectbox(
                "Select a numerical column",
                numeric_data.columns.tolist()
            )

            values = numeric_data[selected_column].dropna()

            if not values.empty:

                c1, c2, c3 = st.columns(3)

                c1.metric("Mean", f"{values.mean():,.3f}")
                c2.metric("Median", f"{values.median():,.3f}")
                c3.metric("Std. Deviation", f"{values.std():,.3f}")

                fig = px.histogram(
                    data,
                    x=selected_column,
                    marginal="box",
                    title=f"Distribution of {selected_column}",
                    color_discrete_sequence=["#8B5CF6"]
                )

                st.plotly_chart(
                    plot_layout(fig),
                    use_container_width=True
                )

                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1

                if iqr > 0:
                    outlier_count = int(
                        (
                            (values < q1 - 1.5 * iqr)
                            | (values > q3 + 1.5 * iqr)
                        ).sum()
                    )

                    st.write(
                        f"Potential outliers in **{selected_column}**: "
                        f"**{outlier_count:,}** using the 1.5×IQR rule."
                    )
                else:
                    st.write(
                        "The selected column has no IQR spread, "
                        "so this rule cannot flag outliers."
                    )

        else:
            st.info("No numerical columns were detected.")

        # CORRELATION
        st.subheader("3. Correlation Analysis")

        if numeric_data.shape[1] >= 2:

            corr = numeric_data.corr(numeric_only=True)

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Heatmap"
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

            upper = corr.where(
                np.triu(
                    np.ones(corr.shape), k=1
                ).astype(bool)
            )

            pairs = upper.stack().dropna()

            if not pairs.empty:
                pair = pairs.abs().idxmax()

                st.write(
                    f"Strongest absolute correlation: "
                    f"**{pair[0]}** and **{pair[1]}** "
                    f"(r = {corr.loc[pair[0], pair[1]]:.3f})."
                )

                st.caption(
                    "Correlation does not establish causation."
                )

        else:
            st.info(
                "At least two numerical columns are needed "
                "for correlation analysis."
            )

        # CATEGORICAL ANALYSIS
        st.subheader("4. Categorical Analysis")

        categorical_columns = data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        # Include columns with missing values even if no object columns exist.
        if categorical_columns:

            category = st.selectbox(
                "Select a categorical column",
                categorical_columns
            )

            top_categories = (
                data[category]
                .fillna("(Missing)")
                .astype(str)
                .value_counts()
                .head(10)
                .rename_axis(category)
                .reset_index(name="Count")
            )

            fig = px.bar(
                top_categories,
                x="Count",
                y=category,
                orientation="h",
                title=f"Most Frequent Values: {category}",
                color="Count",
                color_continuous_scale="Purples"
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                plot_layout(fig),
                use_container_width=True
            )

        else:
            st.info("No categorical columns were detected.")

        # QUICK FINDINGS
        st.subheader("5. Quick Findings")

        findings = build_quick_findings(data)

        for finding in findings:
            st.markdown(f"- {finding}")

        report = (
            "RETAIL SALES INTELLIGENCE - DATA INSIGHTS REPORT\n\n"
            f"Dataset: {st.session_state.get('uploaded_filename', 'Uploaded dataset')}\n\n"
            + "\n".join(f"- {item}" for item in findings)
        )

        st.download_button(
            "⬇️ Download Insights Report",
            data=report,
            file_name="data_insights_report.txt",
            mime="text/plain"
        )


# =====================================================
# PAGE 5: CUSTOMER SEGMENTATION
# =====================================================
elif page == "Customer Segmentation":

    st.markdown("## 👥 Customer Segmentation")

    st.write(
        "Generate customer-level RFM features from transactions "
        "and apply your trained K-Means model if its preprocessing matches."
    )

    if "uploaded_data" not in st.session_state:

        st.info(
            "Upload your dataset first using the Upload Dataset page."
        )

    else:

        source = st.session_state["uploaded_data"].copy()

        # Try to use existing RFM columns first, otherwise aggregate transactions.
        rfm_columns = ["Recency", "Frequency", "Monetary"]

        if all(column in source.columns for column in rfm_columns):

            rfm = source.copy()

            if "Customer_ID" not in rfm.columns:
                rfm["Customer_ID"] = np.arange(1, len(rfm) + 1)

            rfm = rfm[
                ["Customer_ID"] + rfm_columns
            ]

            rfm_error = None

        else:
            rfm, rfm_error = build_rfm(source)

        if rfm is None:

            st.warning(rfm_error)

            st.caption(
                "For automatic RFM generation, provide a customer ID, "
                "a transaction date, and a revenue or amount column."
            )

        else:

            st.markdown("### Customer RFM preview")

            st.caption(
                f"Prepared {len(rfm):,} customer-level records."
            )

            st.dataframe(
                rfm.head(100),
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download RFM Dataset",
                data=rfm.to_csv(index=False).encode("utf-8"),
                file_name="customer_rfm.csv",
                mime="text/csv"
            )

            if not MODEL_PATH.exists():

                st.warning(
                    "RFM features are ready, but the trained model file "
                    "was not found in the models folder."
                )

                st.code(
                    "models/retail_segmentation_model.pkl"
                )

            else:

                try:
                    bundle = joblib.load(MODEL_PATH)

                    if (
                        not isinstance(bundle, dict)
                        or "scaler" not in bundle
                        or "kmeans" not in bundle
                    ):

                        st.error(
                            "The model bundle must contain "
                            "'scaler' and 'kmeans'."
                        )

                    else:

                        st.success("Trained retail model loaded.")

                        log_transform = bundle.get(
                            "log_transform", None
                        )

                        if log_transform is None:

                            st.warning(
                                "SuccessFull"
                            )

                        else:

                            features = rfm[
                                rfm_columns
                            ].apply(pd.to_numeric, errors="coerce")

                            if features.isna().any().any():

                                st.error(
                                    "RFM contains missing or non-numeric values."
                                )

                            elif not np.isfinite(
                                features.to_numpy(dtype=float)
                            ).all():

                                st.error(
                                    "RFM contains infinite values."
                                )

                            elif (features < 0).any().any():

                                st.error(
                                    "Negative RFM values were detected. "
                                    "Review your dataset before prediction."
                                )

                            else:

                                model_features = (
                                    np.log1p(features)
                                    if log_transform
                                    else features
                                )

                                expected_features = getattr(
                                    bundle["scaler"],
                                    "n_features_in_",
                                    model_features.shape[1]
                                )

                                if (
                                    expected_features
                                    != model_features.shape[1]
                                ):

                                    st.error(
                                        f"The scaler expects {expected_features} "
                                        f"features, but {model_features.shape[1]} "
                                        "were prepared."
                                    )

                                else:

                                    scaled = bundle[
                                        "scaler"
                                    ].transform(model_features)

                                    segmented = rfm.copy()

                                    segmented["Cluster"] = bundle[
                                        "kmeans"
                                    ].predict(scaled)

                                    st.markdown("### Segmentation results")

                                    a, b = st.columns(2)

                                    a.metric(
                                        "Customers Segmented",
                                        f"{len(segmented):,}"
                                    )

                                    b.metric(
                                        "Segments Found",
                                        f"{segmented['Cluster'].nunique():,}"
                                    )

                                    summary = (
                                        segmented.groupby("Cluster")
                                        .agg(
                                            Customers=("Customer_ID", "count"),
                                            Avg_Recency=("Recency", "mean"),
                                            Avg_Frequency=("Frequency", "mean"),
                                            Avg_Monetary=("Monetary", "mean")
                                        )
                                        .round(2)
                                        .reset_index()
                                    )

                                    st.dataframe(
                                        summary,
                                        use_container_width=True
                                    )

                                    fig = px.scatter(
                                        segmented,
                                        x="Frequency",
                                        y="Monetary",
                                        color=segmented["Cluster"].astype(str),
                                        hover_data=[
                                            "Customer_ID",
                                            "Recency"
                                        ],
                                        title="Customer Segments",
                                        labels={
                                            "color": "Cluster",
                                            "Frequency": "Purchase Frequency",
                                            "Monetary": "Total Monetary Value"
                                        }
                                    )

                                    st.plotly_chart(
                                        plot_layout(fig),
                                        use_container_width=True
                                    )

                                    st.dataframe(
                                        segmented,
                                        use_container_width=True
                                    )

                                    st.download_button(
                                        "⬇️ Download Segmented Customers",
                                        data=segmented.to_csv(
                                            index=False
                                        ).encode("utf-8"),
                                        file_name="customer_segments.csv",
                                        mime="text/csv"
                                    )

                except Exception as error:

                    st.error(
                        f"Could not load or apply the model: {error}"
                    )

                    st.caption(
                        "Check the model file, its training feature order, "
                        "and the preprocessing used when it was saved."
                    )
