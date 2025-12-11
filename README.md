🚀 Real Estate Investment Advisor Dashboard

This project is an end-to-end Real Estate Investment Advisor that uses Machine Learning models, EDA, data preprocessing, and a Streamlit dashboard to predict:

Whether a property is a Good Investment

The 5-year future price

Local market behaviour (Price per SqFt, BHK distribution, scatter trends)

City-level property insights

It is designed to help stakeholders make data-driven real-estate decisions.

📂 Project Structure
RealEstateInvestmentAdvisor/
│
├── app_dashboard.py                  # Streamlit Dashboard
├── requirements.txt                  # Dependencies
│
├── data/
│   ├── cleaned_housing_data.csv      # Processed dataset used by the dashboard
│   └── india_housing_prices.csv      # Raw dataset
│
├── models/
│   ├── investment_classifier_memsafe.pkl       # Classification Model
│   └── future_price_regressor_memsafe.pkl      # Regression Model
│
├── notebooks/
│   └── eda.ipynb                     # Exploratory Data Analysis Notebook
│
├── src/
│   ├── preprocess.py                 # Data cleaning & feature engineering
│   ├── train_classification_memory_safe.py     # Classification model training
│   └── train_regression_memory_safe.py         # Regression model training
│
└── README.md                         # Project Documentation

🔍 1. Exploratory Data Analysis (EDA)

File: notebooks/eda.ipynb

This notebook includes:

Data overview and missing value checks

Price & SqFt distribution

Outlier detection

BHK-level analysis

Locality and city-level pricing trends

Correlation analysis

Key insights used for model building

🧹 2. Data Preprocessing & Feature Engineering

File: src/preprocess.py

Includes:

Cleaning raw dataset

Handling missing data

Standardizing text fields

Creating derived features (Price_per_SqFt)

Frequency encoding for high-cardinality columns

Saving the processed dataset as cleaned_housing_data.csv

Models use this standardized dataset for training.

🤖 3. Machine Learning Models

Training scripts:

train_classification_memory_safe.py

train_regression_memory_safe.py

Models:

Model	Purpose	File
Classification Model	Predicts Good Investment (1/0)	investment_classifier_memsafe.pkl
Regression Model	Predicts 5-year future price	future_price_regressor_memsafe.pkl
🖥️ 4. Streamlit Dashboard

File: app_dashboard.py

Key Features:

Cascading dropdowns → State → City → Locality

Enter property details (BHK, Size, Furnished Status, Amenities etc.)

Investment Recommendation

Confidence Score

Future Price Prediction

Local Market Snapshot

Interactive Plotly Charts:
✓ Price per SqFt distribution
✓ BHK distribution
✓ Price vs Size scatter
✓ Now vs 5Y prediction

▶️ How to Run Locally
1️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # Mac/Linux

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app_dashboard.py


It will open at:

http://localhost:8501/

🌐 Deployment (Streamlit Cloud)

To deploy:

Push the project to GitHub

Go to https://streamlit.io/cloud

Click New App

Select your GitHub repo

Choose app_dashboard.py

Click Deploy

Your dashboard will run online at a public URL like:

https://your-app-name.streamlit.app/

(This is what you will submit as your working dashboard.)

🎥 Submission Requirements

Submit:

GitHub Repository Link

Streamlit Cloud App Link

Video Explanation (5–8 minutes)

Your video should explain:

EDA overview

Preprocessing steps

ML model training summary

Dashboard walkthrough (inputs → prediction → charts)

Insights and conclusion

👤 Author

VK
Real Estate Analytics & Machine Learning Dashboard Developer