import pandas as pd

# Load Bronze
df = pd.read_parquet("data/bronze/credit_card_default.parquet")

# 1. Rename columns to snake_case
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# 2. Fix invalid values
df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})  # group invalids into "others"
df["marriage"] = df["marriage"].replace({0: 3})  # group invalid into "others"

# 3. Ensure correct dtypes
categorical_cols = ["sex", "education", "marriage", "default"]
df[categorical_cols] = df[categorical_cols].astype("int")

float_cols = [c for c in df.columns if "bill_amt" in c or "pay_amt" in c or c == "limit_bal"]
df[float_cols] = df[float_cols].astype("float")

# Save Silver
df.to_parquet("data/silver/credit_card_default.parquet", index=False)

print("✅ Silver dataset created at data/silver/credit_card_default.parquet")
