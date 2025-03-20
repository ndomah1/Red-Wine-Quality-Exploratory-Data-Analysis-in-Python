# Red Wine Quality Exploratory Data Analysis in Python

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
  - [Features](#features)
    - [Independent Variables](#independent-variables)
    - [Dependent Variable](#dependent-variable)
- [Goals and Key Questions](#goals-and-key-questions)
- [Exploratory Data Analysis (EDA) Steps](#exploratory-data-analysis-eda-steps)
  - [1. Data Inspection](#1-data-inspection)
  - [2. Handling Duplicates](#2-handling-duplicates)
  - [3. Distribution of Wine Quality Scores](#3-distribution-of-wine-quality-scores)
  - [4. Correlation Analysis](#4-correlation-analysis)
  - [5. Feature Distributions & Outlier Detection](#5-feature-distributions--outlier-detection)
  - [6. Pairwise Relationships](#6-pairwise-relationships)
- [Conclusions & Recommendations](#conclusions--recommendations)
  - [Key Takeaways](#key-takeaways)
  - [Limitations of the Analysis](#limitations-of-the-analysis)
  - [Next Steps](#next-steps)
- [Usage Instructions](#usage-instructions)


## **Project Overview**

This project explores the **Red Wine Quality dataset** from the **UCI Machine Learning Repository (**https://archive.ics.uci.edu/dataset/186/wine+quality**)**. The dataset contains **physicochemical attributes** of different red wine samples, along with a **quality score (target variable)**. The objective is to perform **exploratory data analysis (EDA)** to uncover key patterns, relationships, and data characteristics that influence wine quality.

## **Dataset Description**

The dataset consists of **1,599 records** and **12 features** (11 independent variables + 1 target variable). The **input features** include various physicochemical properties, while the **output variable (`quality`)** is an **integer score ranging from 3 to 8**, representing the wine’s quality.

### **Features:**

#### **Independent Variables (Physicochemical Properties)**

- **Fixed Acidity** – Primary acid in wine.
- **Volatile Acidity** – Acetic acid amount (high values give vinegar-like taste).
- **Citric Acid** – Contributes to wine freshness.
- **Residual Sugar** – Natural sugar left after fermentation.
- **Chlorides** – Salt content in wine.
- **Free Sulfur Dioxide** – SO2 not bound to molecules, helps prevent spoilage.
- **Total Sulfur Dioxide** – Total amount of SO2 (both bound and free).
- **Density** – Wine density relative to water.
- **pH** – Acidity or alkalinity level.
- **Sulfates** – Additive for preserving wine.
- **Alcohol** – Alcohol content percentage.

#### **Dependent Variable (Target)**

- **Quality Score (3 to 8)** – The higher the score, the better the wine quality.

## **Goals and Key Questions**

This EDA aims to answer the following questions:

1. **Are there missing values or duplicate records?**
2. **What is the distribution of wine quality scores?**
3. **Which features strongly correlate with wine quality?**
4. **Are there any significant outliers in the dataset?**
5. **How do different features influence wine quality?**

## **Exploratory Data Analysis (EDA) Steps**

### **1. Data Inspection**

We begin by loading the dataset and inspecting its structure:

```python
df_red.info()
df_red.describe()
df_red.head()
```

#### **Findings**

- All features are **numerical** (either float or integer).
- No **missing values** exist in the dataset.
- **Duplicate records detected.**

### **2. Handling Duplicates**

The dataset contained **240 duplicate records**, which were removed using:

```python
df_red = df_red.drop_duplicates()
```

#### **Updated Dataset Shape**

**1,359 rows, 12 columns** (after duplicate removal).

### **3. Distribution of Wine Quality Scores**

A **count plot** was used to analyze the distribution of wine quality ratings:

```python
sns.countplot(x=df_red["quality"], palette="Set2")
```

![image.png](image.png)

#### **Findings**

- The dataset is **imbalanced**:
    - Most wines are rated **5 or 6**.
    - Very few wines have scores **3, 4, 7, or 8**.
    - No wines were rated **1, 2, 9, or 10**.

### **4. Correlation Analysis**

To identify relationships between features, a **correlation heatmap** was plotted:

```python
sns.heatmap(df_red.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
```

![image.png](image%201.png)

#### **Key Observations**

- **Alcohol content** has the **strongest positive correlation with quality** (**0.48**).
- **Volatile acidity** negatively correlates with quality (-**0.40**).
- **Density and pH have weak correlations** with quality.
- **Citric acid and fixed acidity** show **moderate inter-correlations**, suggesting possible multicollinearity.

### **5. Feature Distributions & Outlier Detection**

To analyze **feature distributions**, **histograms** and **boxplots** were used.

#### **Histograms for Feature Distribution**

```python
for col in df_red.columns:
    sns.histplot(df_red[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()
```

![image.png](image%202.png)

#### **Finding**

- Most features have **right-skewed distributions**.
- Alcohol levels range from **8% to 15%**, peaking at **10-12%**.

#### **Boxplots for Outlier Detection**

```python
sns.boxplot(data=df_red, palette="Set3")
```

![image.png](image%203.png)

#### **Outliers Identified**

- **Volatile Acidity** contains multiple high values.
- **Sulfur Dioxide Levels** have extreme values.
- **Residual Sugar** shows a right-skewed distribution.

### **6. Pairwise Relationships**

A **pairplot** was used to **visualize interactions** between features:

```python
sns.pairplot(df_red, hue="quality", palette="husl")
```

![image.png](image%204.png)

#### **Findings**

- Wines with **higher alcohol content** tend to have **higher quality**.
- High **volatile acidity** is associated with **lower quality wines**.
- **Citric acid and fixed acidity** exhibit a **moderate positive correlation**.

## **Conclusions & Recommendations**

### **Key Takeaways**

- **No missing values**, but **240 duplicate records** were removed.
- **Alcohol content is the strongest predictor of wine quality**.
- **Volatile acidity negatively affects quality**—higher acidity results in lower quality wines.
- The dataset is **imbalanced**, with most wines rated **5 or 6**.
- Outliers were detected in **volatile acidity, residual sugar, and sulfur dioxide levels**.

### **Limitations of the Analysis**

1. **Imbalanced Data** – Most wines are rated **5 or 6**, making the dataset skewed and potentially biasing models.
2. **Missing Contextual Factors** – The dataset lacks **grape variety, region, and production year**, which impact wine quality.
3. **Correlation vs. Causation** – While **alcohol content** strongly correlates with quality, it doesn’t **cause** better wine.
4. **Outliers** – **Volatile acidity, residual sugar, and sulfur dioxide** contain extreme values that may need further handling.
5. **Limited Feature Scope** – No **categorical data** (e.g., wine type, vineyard), reducing potential insights.

### **Next Steps**

1. **Feature Engineering**: Consider binning quality scores into broader categories (Low, Medium, High).
2. **Data Preprocessing**: Handle outliers using **capping, log transformations, or standardization**.
3. **Predictive Modeling**: Train a **classification model** (e.g., Logistic Regression, Random Forest) to predict wine quality.
4. **Visualization Enhancements**: Create **interactive dashboards** using **Tableau or Power BI**.

## **Usage Instructions**

To reproduce this analysis:

1. Install dependencies:
    
    ```bash
    pip install ucimlrepo numpy pandas seaborn matplotlib math
    ```
    
2. Run the Jupyter Notebook:
    
    ```bash
    jupyter notebook wine_quality_EDA.ipyn
    ```
    
3. Explore the visualizations and insights.

This **EDA provides critical insights** into the wine quality dataset, serving as a foundation for further machine learning and predictive modeling.
