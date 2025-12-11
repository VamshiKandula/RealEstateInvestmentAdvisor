import pandas as pd

# 1. Load the raw dataset
df = pd.read_csv(r"D:\Labmentix\RealEstateInvestmentAdvisor\data\india_housing_prices.csv")

print("Initial data shape:", df.shape)

# 2. Drop duplicate rows
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)

# 3. Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

print("\nNumeric columns:", list(numeric_cols))
print("\nCategorical columns:", list(categorical_cols))

# 4. Handle missing values

# Fill numeric NaNs with median
for col in numeric_cols:
    if df[col].isna().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled NaNs in numeric column '{col}' with median: {median_val}")

# Fill categorical NaNs with mode
for col in categorical_cols:
    if df[col].isna().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Filled NaNs in categorical column '{col}' with mode: {mode_val}")

print("\nMissing values after cleaning:")
print(df.isna().sum())

# 5. Feature Engineering

# 5.1 Price_per_SqFt
if "Price_per_SqFt" not in df.columns:
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
    print("\nCreated 'Price_per_SqFt' feature.")
else:
    df["Price_per_SqFt"] = df["Price_per_SqFt"].fillna(
        (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
    )
    print("\n'Price_per_SqFt' already existed – filled missing values if any.")

# 5.2 Age_of_Property
if "Age_of_Property" not in df.columns:
    CURRENT_YEAR = 2024  # you can adjust this later if needed
    df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
    print("Created 'Age_of_Property' feature.")
else:
    print("'Age_of_Property' already exists.")

# 6. Create Good_Investment label
# Rule: If Price_per_SqFt <= city median Price_per_SqFt → Good investment (1), else 0

if "City" in df.columns:
    df["City_Median_Price_per_SqFt"] = df.groupby("City")["Price_per_SqFt"].transform("median")
    df["Good_Investment"] = (df["Price_per_SqFt"] <= df["City_Median_Price_per_SqFt"]).astype(int)
    print("\nCreated 'Good_Investment' label based on city median price per sqft.")
else:
    print("\nWARNING: 'City' column not found – could not create Good_Investment label.")

# 7. Save the cleaned dataset
output_path = r"D:\Labmentix\RealEstateInvestmentAdvisor\data\cleaned_housing_data.csv"
df.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")