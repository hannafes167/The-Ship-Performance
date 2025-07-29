# The Ship Performance

# Bussiness Understanding
Shipping companies operating in the Gulf of Guinea face complex challenges in managing a diverse fleet of vessels, navigating various routes, and dealing with unpredictable weather conditions. In this context, understanding ship performance is crucial to reducing operational costs, improving energy efficiency, and maximizing revenue per voyage.

# Problem Statement
The primary business problems addressed in this project are:

1. How can ships be grouped based on similar operational characteristics and performance metrics to support strategic decision-making?

2. How can the ship's maintenance status (Maintenance_Status) be accurately predicted based on operational data to enable preventive actions?


Without data-driven solutions, shipping companies risk the following:

1. Inefficient allocation of resources.

2. Missed opportunities for cost savings through fleet optimization.

3. Financial losses due to unplanned and reactive maintenance.
   

# Goals
## Clustering
1. Segment ships based on key characteristics such as energy efficiency, cargo weight, speed, distance traveled, and turnaround time.

2. Identify patterns among high- and low-performing vessel groups.

3. Provide actionable insights for fleet management, route planning, and maintenance strategy.

## Classification
1. Build a classification model to predict the Maintenance_Status of ships (Good, Fair, Critical) using operational features such as engine power, weather conditions, weekly voyage count, and average load.

2. Develop an early warning system to detect ships at risk of technical issues.

3. Assist operational teams in planning preventive maintenance more effectively and efficiently.

# Preparing
## Dataset
Name: Ship Performance Dataset

Total Data: 2736 baris, 18 kolom

Type Data:

Numeric: Speed_Over_Ground_knots, Engine_Power_kW, Distance_Traveled_nm, Operational_Cost_USD, Revenue_per_Voyage_USD, Efficiency_nm_per_kWh, Draft_Meters, Cargo_weight_tons, Seasonal_Impact_Score, Turnaround_Time_hours, Weekly_Voyage_Count, Average_Load_Percentage

Categorikal: Ship_Type, Route_Type, Engine_Type, Maintenance_Status, Weather_Condition

Link: (https://github.com/hannafes167/The-Ship-Performance/blob/main/Ship_Performance_Dataset.csv)

## Setup Environment
### Tools & Software:

1. Python 3.10+

2. Jupyter Notebook / VS Code / Google Colab

### Library
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from umap.umap_ import UMAP
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
```

### Workflow
- Exploratory Data Analysis (EDA)
  
  Conducted initial exploration to understand the structure, distribution, and relationships within the dataset. This includes summary statistics, visualizations, and correlation analysis to identify patterns and potential issues such as missing values or outliers.

- Data Preprocessing
  
  Prepared the data for modeling by:
  1. Handling missing values
  2. Checked duplicate
  3. Checked Outliers
  4. Encoding categorical variables
  5. Feature scaling (e.g., using StandardScaler)
  6. Selecting relevant features for clustering and classification tasks

- Modeling
  
  Clustering: Applied K-Means algorithm to group ships into clusters based on operational characteristics. The optimal number of clusters was determined using the Elbow Method.

  Classification: Built a supervised learning model to predict ship maintenance status. Algorithms such as Random Forest were used due to their robustness and interpretability.

- Training & Evaluation
  
  For classification: Split the data into training and test sets, trained the model, and evaluated its performance using metrics such as accuracy, precision, recall, and confusion matrix.

  For clustering: Evaluated cluster quality using visual methods (PCA, silhouette plots) and business interpretability.

- Inference & Insights
  
  Interpreted model results to extract meaningful business insights.

  For clustering: Identified high- and low-performance ship segments.

  For classification: Highlighted important features that influence maintenance status predictions and how this can inform preventive maintenance strategies.



# Output
## Hasil Clustering
![](https://github.com/hannafes167/The-Ship-Performance/blob/main/project_ship_performance1.png)

## Hasil Klasifikasi
![](https://github.com/hannafes167/The-Ship-Performance/blob/main/project_ship_performance2.png)

## Analisis
### Analysis of Cluster Characteristics from the KMeans Model
    
The following is an analysis of the characteristics of each cluster generated by the KMeans model.

Based on the clustering results using the KMeans model, four clusters were identified with the following key characteristics:

**Cluster 0:**
- The dominant Ship Type is Fish Carrier with a maintenance status of Good.
- The average Speed Over Ground is approximately 17 knots, ranging between 10 and 25 knots.
- The Engine Power has an average of 1,760 kW, with a minimum value of 502 kW and a maximum of 2,999 kW.
- The average distance traveled is 1,045 nm, with a range between 53 and 1,997 nm.
- The average operational cost is USD 254,956, with a range from USD 10,189 to USD 498,579. Analisis:
- This cluster is primarily composed of Fish Carriers that are well-maintained (Good status). The relatively stable speed and engine power, along with moderate operational costs, suggest that these vessels operate on medium to long-distance routes while maintaining their engine condition effectively.

**Cluster 1:**
- The dominant Ship Type is Tanker with a maintenance status of Fair.
- The average Speed Over Ground is approximately 18 knots, ranging between 10 and 25 knots.
- The Engine Power has an average of 1,789 kW, with a minimum value of 508 kW and a maximum of 2,995 kW.
- The average distance traveled is 1,058 nm, with a range between 50 and 1,995 nm.
- The average operational cost is USD 252,100, with a range from USD 10,092 to USD 499,711. Analysis: This cluster consists of Tankers with a Fair maintenance status. The slightly lower operational cost compared to Cluster 0, despite having similar engine power and travel distances, may indicate more efficient fuel usage or optimized operational routes. However, the Fair maintenance status suggests that these ships require greater attention to maintain performance.

**Cluster 2:**
- The dominant Ship Type is Bulk Carrier with a maintenance status of Fair.
- The average Speed Over Ground is approximately 18 knots, ranging between 10 and 25 knots.
- The Engine Power has an average of 1,748 kW, with a minimum value of 502 kW and a maximum of 2,995 kW.
- The average distance traveled is 1,002 nm, with a range between 52 and 1,998 nm.
- The average operational cost is USD 254,491, with a range from USD 11,376 to USD 498,862. Analysis: This cluster shares similar characteristics with Cluster 1, but the ship type is Bulk Carrier, and the travel distance is slightly shorter. The almost identical operational cost suggests that, despite the shorter travel distance, the fuel or maintenance expenses may be higher. The Fair maintenance status indicates that these vessels require regular monitoring to ensure performance.

**Cluster 3:**
- The dominant Ship Type is Container Ship with a maintenance status of Critical.
- The average Speed Over Ground is approximately 17 knots, ranging between 10 and 25 knots.
- The Engine Power has an average of 1,735 kW, with a minimum value of 504 kW and a maximum of 2,999 kW.
- The average distance traveled is 1,054 nm, with a range between 50 and 1,998 nm.
- The average operational cost is USD 261,368, with a range from USD 10,097 to USD 497,735. Analysis: This cluster highlights Container Ships with a Critical maintenance status, indicating a high risk in terms of performance and safety. The higher operational costs compared to other clusters could be due to urgent maintenance requirements or less efficient fuel consumption. This suggests a need for immediate repairs and operational optimizations for vessels in this cluster.

# Conclusion:

1. Operational Cost Differences: Cluster 3 has the highest operational cost, likely due to its critical maintenance condition and the Container Ship type, which requires specialized handling.
   
2. Maintenance Condition: Most ships with a Fair maintenance status are in Cluster 1 and 2, which may require regular inspections to prevent performance decline similar to Cluster 3.

3. Operational Efficiency: Cluster 1 and 2 exhibit similar operational cost patterns, despite differences in ship type and travel distance, indicating operational efficiency in Tanker and Bulk Carrier ships.

4. Recommendations: Ships in Cluster 3 require special attention to reduce operational costs and improve maintenance status, while ships in Cluster 0 can serve as an example of a more stable operational model.
