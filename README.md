# Blood-Donation-Forecasting-4
This project utilizes machine learning and the RFMTC marketing model to predict whether a past blood donor will return to donate blood again. By accurately forecasting returning donors, blood collection managers can run targeted campaigns to prevent supply shortages.

## 📊 The Dataset
The dataset consists of a random sample of 748 donors from the Blood Transfusion Service Center in Hsin-Chu City, Taiwan. 

The data is structured around the **RFMTC** marketing model:
* **R (Recency):** Months since the last donation.
* **F (Frequency):** Total number of donations.
* **M (Monetary):** Total volume of blood donated (in c.c.).
* **T (Time):** Months since the first donation.
* **Target:** Did they donate blood in March 2007? (Binary: 1 = Yes, 0 = No).

## 🚀 Methodology & Models
This project compares an Automated Machine Learning approach against a custom-tuned linear model.

1. **Automated Machine Learning (TPOT):** Utilized the `TPOTClassifier` to automatically explore and evaluate various machine learning pipelines to establish a high-performing baseline AUC score.
2. **Data Preprocessing (Log Normalization):** Identified high variance in the 'Monetary' feature and applied logarithmic normalization to stabilize the data distribution.
3. **Logistic Regression:** Trained a custom Logistic Regression model on the normalized dataset, resulting in a highly interpretable model that competes directly with the AutoML pipeline.

## 🛠️ Tech Stack
* **Language:** Python 3
* **AutoML:** TPOT
* **Machine Learning:** Scikit-Learn (Logistic Regression)
* **Data Manipulation:** Pandas, NumPy

## 💻 How to Run
1. Clone the repository.
2. Install the required dependencies: `pip install pandas numpy scikit-learn tpot==0.12.2`
3. Ensure the `transfusion.data` file is located in the `datasets/` directory.
4. Open and run the `bloodDonation main code.ipynb` file in Jupyter or Google Colab to view the data exploration, training process, and final AUC scores.
