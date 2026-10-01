# Value-for-Money Car Recommender

An SMLD/ML coursework project that estimates expected car prices using Linear Regression and ranks vehicles based on a custom **Value Score**.

## Overview
This system processes a Kaggle car-pricing dataset, trains a machine learning regression model to predict fair market prices, and identifies deals where vehicles are listed below their predicted market value.

---

## Core Formula

$$\text{Value Score} = \text{Predicted Price} - \text{Listed Price}$$

* **Positive Value Score:** Listed below predicted market value (Good Deal).
* **Negative Value Score:** Listed above predicted market value (Overpriced).

---

## Execution Steps

1. **Place Raw Data:** Ensure `car_price.csv` is present in the root folder.
2. **Data Cleaning & EDA:**
   ```bash
   python clean_data.py