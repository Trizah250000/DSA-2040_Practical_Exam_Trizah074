# Trizah250000-DSA-2040_Practical_Exam_Trizah074
### End Semester Project
#### Overview
The exam evaluates practical skills in data warehousing and data mining. I implemented the data warehousing section (Tasks 1–3) using synthetic retail data, including a star schema, ETL process, OLAP queries, visualization, and analysis report. For data mining, I completed all tasks (4–6) using the Iris dataset and synthetic transactional data, covering preprocessing, clustering, classification, and association rule mining. Below, I detail the datasets and how to run the code

#### Datasets

Data Warehousing (Tasks 1–3): Generated synthetic retail data using Python with Faker, mimicking the UCI "Online Retail" dataset. It has ~1000 initial rows (reduced to 47 after ETL), with columns like InvoiceNo, ProductID, Category (Electronics, Clothing, Home Decor, Toys), Quantity, UnitPrice, InvoiceDate, CustomerID, Country, and TotalSales. Used random.seed(42) for reproducibility. Files: retail_data.csv and retail_dw.db.
Data Mining (Tasks 4–6): Used the Iris dataset from scikit-learn (150 samples, 4 features, 3 species) and synthetic transactional data (30 transactions, 20 items like 'milk', 'bread'). Preprocessed Iris data is saved as Iris_preprocessed.csv.

Repository Structure
```
Data_Mining/
├── Task1_Data_Preprocessing_and_Exploration/
│   ├── Visualizations/
│   │   ├── iris_pairplot.png
│   │   ├── iris_heatmap.png
│   │   └── iris_boxplots.png
│   ├── Iris_preprocessed.csv
│   └── Preprocessing_iris.ipynb
├── Task2_Clustering/
│   ├── Visualizations/
│   │   ├── clusters_k2.png
│   │   ├── clusters_k3.png
│   │   ├── clusters_k4.png
│   │   └── elbow_curve.png
│   └── Clustering_iris.ipynb
├── Task3_Classification_and_Association_Rule_Mining/
│   ├── Visualization/
│   │   └── decision_tree.png
│   └── mining_iris_basket.ipynb
Data_Warehousing/
├── Task1_Data_Warehouse_Design/
│   ├── Retail_Schema_Diagram.png
│   └── retail.sql
├── Task2_ETL_Process_Implementation/
│   ├── etl_retail.ipynb
│   ├── retail_data.csv
│   └── retail_dw.db
├── Task3_OLAP_Queries_and_Analysis/
│   ├── OLAP_Analysis.ipynb
│   ├── OLAP_Queries.sql
│   ├── retail_dw.db
│   └── sales_by_country.png
├── .gitignore
├── LICENSE
└── README.md
```
How to Run the Code
Requirements

Python: Version 3.x with libraries: pandas, numpy, faker, sqlite3, matplotlib, seaborn, scikit-learn, mlxtend.
SQL Client: DB Browser for SQLite (optional to view retail_dw.db).
Install dependencies:pip install pandas numpy faker matplotlib seaborn scikit-learn mlxtend



Steps

Clone the Repository:
git clone https://github.com/yourusername/DSA_2040_Practical_Exam_JohnDoe123.git
cd DSA_2040_Practical_Exam_JohnDoe123


Task 1: Data Warehouse Design:

View schema_diagram.png for the star schema.
Run schema.sql in an SQL client or Python to create tables:sqlite3 retail_dw.db < schema.sql




Task 2: ETL Process:

Run etl_retail.py to generate synthetic data, transform it, and load it into retail_dw.db:python etl_retail.py


Outputs: retail_data.csv (1000 rows) and retail_dw.db (47 sales records).


Task 3: OLAP Queries and Analysis:

Run olap_queries.sql in an SQL client to execute queries:-- Roll-Up: Total sales by country and quarter
SELECT c.Country AS country, t.quarter AS quarter, SUM(s.TotalSales) AS total_sales
FROM SalesFact s
JOIN CustomerDim c ON s.CustomerPK = c.CustomerPK
JOIN TimeDim t ON s.TimePK = t.TimePK
WHERE t.Year = 2024 AND t.Month BETWEEN 7 AND 9
GROUP BY c.Country, t.quarter
ORDER BY total_sales DESC;

-- Drill-Down: Sales in Liechtenstein by month
SELECT t.Month AS month, SUM(s.TotalSales) AS total_sales
FROM SalesFact s
JOIN CustomerDim c ON s.CustomerPK = c.CustomerPK
JOIN TimeDim t ON s.TimePK = t.TimePK
WHERE t.Year = 2024 AND c.Country = 'Liechtenstein'
GROUP BY t.Month
ORDER BY t.Month;

-- Slice: Total sales for Electronics category
SELECT c.Country AS country, SUM(s.TotalSales) AS total_sales
FROM SalesFact s
JOIN CustomerDim c ON s.CustomerPK = c.CustomerPK
JOIN ProductDim p ON s.ProductPK = p.ProductPK
WHERE p.Category = 'Electronics'
GROUP BY c.Country
ORDER BY total_sales DESC;


Run the visualization code in etl_retail.py to generate sales_by_country.png:import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn = sqlite3.connect('retail_dw.db')
query = """
SELECT c.Country AS country, SUM(s.TotalSales) AS total_sales
FROM SalesFact s
JOIN CustomerDim c ON s.CustomerPK = c.CustomerPK
JOIN TimeDim t ON s.TimePK = t.TimePK
WHERE t.Year = 2024 AND t.Month BETWEEN 7 AND 9
GROUP BY c.Country
ORDER BY total_sales DESC
"""
df = pd.read_sql_query(query, conn)
plt.figure(figsize=(10, 6))
plt.bar(df['country'], df['total_sales'], color='skyblue')
plt.xlabel('Country')
plt.ylabel('Total Sales ($)')
plt.title('Total Sales by Country (Q3 2024)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('sales_by_country.png')
plt.show()
plt.close()
conn.close()


View report.md for the analysis.


Task 4: Data Preprocessing and Exploration:

Run preprocessing_iris.py to preprocess the Iris dataset and generate visualizations:import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

def check_missing_values(df):
    print("Missing values per column:")
    print(df.isnull().sum())
    return df
check_missing_values(df)

def normalize_features(df, feature_cols):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_scaled
feature_cols = df.columns[:-1].tolist()
df_scaled = normalize_features(df, feature_cols)

from sklearn.preprocessing import LabelEncoder
def encode_labels(df, label_col='species'):
    le = LabelEncoder()
    df_encoded = df.copy()
    df_encoded[label_col + '_encoded'] = le.fit_transform(df[label_col])
    return df_encoded, le
df_encoded, le = encode_labels(df_scaled)
print(df_encoded.head())
print("Encoded classes:", list(le.classes_))

print("\nSummary Statistics:")
print(df.describe())

sns.pairplot(df, hue='species')
plt.savefig('iris_pairplot.png')
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Iris Features')
plt.savefig('iris_heatmap.png')
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=df)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.savefig('iris_boxplots.png')
plt.show()
plt.close()

def split_data(df, test_size=0.2, random_state=42):
    features = [col for col in df.columns if col != 'species']
    X = df[features]
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = split_data(df)
print(f"\nTrain set shape (X_train): {X_train.shape}")
print(f"Test set shape (X_test): {X_test.shape}")
print(f"Train labels shape (y_train): {y_train.shape}")
print(f"Test labels shape (y_test): {y_test.shape}")

df.to_csv('Iris_preprocessed.csv', index=False)


Outputs: Iris_preprocessed.csv, iris_pairplot.png, iris_heatmap.png, iris_boxplots.png.


Task 5: Clustering:

Run clustering_iris.py to perform K-Means clustering and generate visualizations:import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import warnings

warnings.filterwarnings('ignore')
if not os.path.exists('images'):
    os.makedirs('images')

data = pd.read_csv('Iris_preprocessed.csv')
X_scaled = data.drop('species', axis=1)
y = data['species']

def apply_kmeans(X, k, y_true, save_plot=False, k_value=None):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X)
    ari = adjusted_rand_score(y_true, clusters)
    if save_plot:
        if 'petal length (cm)' not in X.columns or 'petal width (cm)' not in X.columns:
            print("Error: 'petal length (cm)' or 'petal width (cm)' not found in X. Check column names.")
            return ari, kmeans.inertia_
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X['petal length (cm)'], y=X['petal width (cm)'],
                        hue=clusters, palette='viridis', style=y_true, markers=['o', 's', '^'])
        plt.title(f'K-Means Clustering (k={k_value})')
        plt.xlabel('Petal Length (Normalized)')
        plt.ylabel('Petal Width (Normalized)')
        plt.legend(title='Cluster / True Class')
        plt.savefig(f'images/clusters_k{k_value}.png')
        plt.show()
        plt.close()
    return ari, kmeans.inertia_

ari_k3, _ = apply_kmeans(X_scaled, k=3, y_true=y, save_plot=True, k_value=3)
print(f"Adjusted Rand Index for k=3: {ari_k3:.3f}")

ari_k2, _ = apply_kmeans(X_scaled, k=2, y_true=y, save_plot=True, k_value=2)
print(f"Adjusted Rand Index for k=2: {ari_k2:.3f}")

ari_k4, _ = apply_kmeans(X_scaled, k=4, y_true=y, save_plot=True, k_value=4)
print(f"Adjusted Rand Index for k=4: {ari_k4:.3f}")

inertias = []
k_values = range(1, 7)
for k in k_values:
    _, inertia = apply_kmeans(X_scaled, k, y, save_plot=False)
    inertias.append(inertia)

plt.figure(figsize=(8, 6))
plt.plot(k_values, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Curve for K-Means')
plt.savefig('images/elbow_curve.png')
plt.show()
plt.close()


Outputs: clusters_k3.png, clusters_k2.png, clusters_k4.png, elbow_curve.png.


Task 6: Classification and Association Rule Mining:

Run mining_iris.py to perform classification and association rule mining:import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
from mlxtend.frequent_patterns import apriori, association_rules

warnings.filterwarnings('ignore')
if not os.path.exists('images'):
    os.makedirs('images')

data = pd.read_csv('Iris_preprocessed.csv')
X = data.drop('species', axis=1)
y = data['species']

def split_train_test(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_count = int(test_size * len(X))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')
print(f"Decision Tree Metrics:")
print(f"Accuracy: {accuracy_dt:.3f}")
print(f"Precision: {precision_dt:.3f}")
print(f"Recall: {recall_dt:.3f}")
print(f"F1-Score: {f1_dt:.3f}")

plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['Setosa', 'Versicolor', 'Virginica'], filled=True)
plt.title("Decision Tree Classifier")
plt.savefig('images/decision_tree.png')
plt.show()
plt.close()

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
print(f"\nKNN Metrics (k=5):")
print(f"Accuracy: {accuracy_knn:.3f}")
print(f"Precision: {precision_knn:.3f}")
print(f"Recall: {recall_knn:.3f}")
print(f"F1-Score: {f1_knn:.3f}")

if accuracy_dt > accuracy_knn:
    print("\nDecision Tree is better than KNN because it has higher accuracy, likely because it finds clearer rules in the data.")
else:
    print("\nKNN is better than Decision Tree because it has higher accuracy, possibly because it works well with nearby data points.")

items = ['milk', 'bread', 'beer', 'diapers', 'eggs', 'cheese', 'juice', 'butter', 'yogurt', 
         'coffee', 'tea', 'sugar', 'flour', 'rice', 'pasta', 'chicken', 'fish', 'apples', 'bananas', 'oranges']
weights = [0.8, 0.7, 0.5, 0.6, 0.7, 0.4, 0.5, 0.6, 0.4, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.5, 0.7, 0.7, 0.6]
transactions = []
for _ in range(30):
    num_items = random.randint(3, 8)
    basket = random.choices(items, weights=weights, k=num_items)
    transactions.append(list(set(basket)))

encoded_vals = []
for transaction in transactions:
    labels = {item: (item in transaction) for item in items}
    encoded_vals.append(labels)
dataset = pd.DataFrame(encoded_vals)

frequent_itemsets = apriori(dataset, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules.sort_values('lift', ascending=False).head(5)

print("\nTop 5 Association Rules (sorted by lift):")
for index, rule in rules.iterrows():
    print(f"Rule: {set(rule['antecedents'])} -> {set(rule['consequents'])}")
    print(f"Support: {rule['support']:.3f}")
    print(f"Confidence: {rule['confidence']:.3f}")
    print(f"Lift: {rule['lift']:.3f}\n")


Outputs: decision_tree.png, top 5 association rules printed.



Task Details and Outputs
Task 1: Data Warehouse Design (15 Marks)

Star Schema: Designed with Sales_Fact (measures: sales_amount, quantity_sold; foreign keys: customer_id, product_id, time_id, store_id) and four dimension tables:
Customer_Dim: customer_id, first_name, last_name, gender, age, city, state, country.
Product_Dim: product_id, product_name, category, brand, unit_price.
Time_Dim: time_id, sale_date, day_of_week, month, quarter, year.
Store_Dim: store_id, store_name, city, state, country.


Diagram: See schema_diagram.png (posted by you).
Why Star Schema?: Chose star schema over snowflake for simpler queries and faster performance, minimizing joins for sales analysis by category, customer, or time.
SQL: See schema.sql (your provided code).

Task 2: ETL Process Implementation (20 Marks)

Code: etl_retail.py generates 1000 synthetic rows, transforms (handles missing values, outliers, calculates TotalSales), and loads into retail_dw.db.
Output: 
retail_data.csv: 1000 rows.
retail_dw.db: 47 sales records, 64 customers, 12 products, 47 time entries.
Logs: Row counts (e.g., 493 after missing CustomerID, 47 after date filter).


Sample Output:CustomerDim Preview:
   CustomerCode  Country  CustomerPK
0           3.0  Liechtenstein           1
1           4.0      Vanuatu           2
SalesFact Preview:
   InvoiceNo  TimePK  ProductPK  CustomerPK  Quantity  UnitPrice  TotalSales
0  4173d89e...       1          1          55         9      70.85      637.65



Task 3: OLAP Queries and Analysis (15 Marks)

Queries: See olap_queries.sql (updated with your roll-up and added drill-down, slice).
Visualization: Bar chart for roll-up query saved as sales_by_country.png.
Report: See report.md (your provided analysis, completed to 235 words).

Task 4: Data Preprocessing and Exploration (15 Marks)

Code: preprocessing_iris.py loads Iris data, normalizes features, encodes labels, and visualizes with pairplot, heatmap, and boxplots.
Output: Iris_preprocessed.csv, iris_pairplot.png, iris_heatmap.png, iris_boxplots.png.
Sample Output:Summary Statistics:
     sepal length (cm)  ...  petal width (cm)
count         150.000000  ...        150.000000
mean            5.843333  ...          1.199333



Task 5: Clustering (15 Marks)

Code: clustering_iris.py applies K-Means with k=2, 3, 4, computes ARI, and plots an elbow curve.
Output: clusters_k3.png, clusters_k2.png, clusters_k4.png, elbow_curve.png.
Sample Output:Adjusted Rand Index for k=3: 0.716
Adjusted Rand Index for k=2: 0.540
Adjusted Rand Index for k=4: 0.598


Analysis: K=3 clusters match Iris species well (ARI ~0.716), with Setosa clearly separated. K=2 and K=4 show lower ARI due to misclassifications. The elbow curve supports k=3 as optimal.

Task 6: Classification and Association Rule Mining (20 Marks)

Code: mining_iris.py trains Decision Tree and KNN classifiers, visualizes the Decision Tree, and applies Apriori for association rules.
Output: 
decision_tree.png.
Metrics:Decision Tree Metrics:
Accuracy: 1.000
Precision: 1.000
Recall: 1.000
F1-Score: 1.000
KNN Metrics (k=5):
Accuracy: 1.000
Precision: 1.000
Recall: 1.000
F1-Score: 1.000
KNN is better than Decision Tree because it has higher accuracy, possibly because it works well with nearby data points.


Top 5 Association Rules (example, dependent on random data):Top 5 Association Rules (sorted by lift):
Rule: {'bread'} -> {'butter'}
Support: 0.233
Confidence: 0.567
Lift: 1.345




Analysis: Decision Tree and KNN both achieved perfect scores, likely due to the small, well-separated Iris dataset. The bread -> butter rule suggests a retail strategy of bundling these items to boost sales.


