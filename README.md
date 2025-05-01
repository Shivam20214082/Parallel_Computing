# Parallel K-Means Clustering for Large Datasets

## 📌 Overview
This project accelerates the K-Means clustering algorithm using parallel computing techniques. K-Means is a popular unsupervised learning algorithm used in machine learning for clustering tasks. While effective, its sequential implementation becomes slow for large datasets or higher cluster counts.

This project focuses on reducing runtime by applying **data parallelism** to the most time-consuming step—**distance computation** between data points and centroids.

---

## ⚙️ Project Objectives

- Implement both sequential and parallel versions of the K-Means algorithm.
- Optimize the distance computation step using parallelization.
- Compare the runtime of both approaches using multiple datasets.
- Ensure memory efficiency and scalability.

---

## 🧠 K-Means Algorithm Steps

1. Randomly initialize `k` centroids.
2. Compute the distance between each data point and all centroids.
3. Assign each data point to the nearest centroid.
4. Update centroids based on the assigned data points.
5. Repeat Steps 2-4 until convergence (centroids don't change significantly).

> 🔍 The distance computation (Step 2) is the most computationally intensive part and is parallelized in this project.

---

## 💡 Why Parallelize?

- **Time Complexity:** O(n × k × t)
  - n = number of data points  
  - k = number of clusters  
  - t = number of iterations for convergence
- With large datasets or high cluster counts, the execution time increases significantly.
- Since each distance calculation is independent, it's ideal for **parallelization**.

---

## 🛠️ Implementation Details

- **Language:** Python
- **Parallelization Tools:** Python's multiprocessing / NumPy optimization
- **Main Script:** `pkmeans.py`
- **Data Folder:** Contains sample datasets for testing
- **Outputs:** Cluster assignments, centroid locations, and performance comparison

---

## 📊 Performance Evaluation

- Compare execution time between sequential and parallel versions.
- Evaluate scalability using datasets of increasing size and dimensions.
- Analyze improvements in runtime and resource usage.

---

## 🌍 Applications

- Market Segmentation
- Image Recognition and Compression
- Anomaly Detection
- Document Clustering
- Sensor Data Analysis

---

## 👨‍💻 Team Members

| Name                     | Registration No. |
|--------------------------|------------------|
| Shivam Kumar Gupta       | 20214082         |
| Tony Jain                | 20214005         |
| Shresth Sonkar           | 20214272         |
| Shweta Sonkar            | 20214298         |
| Sanchita Sharma          | 20214210         |
| Sumit Kasaudhan          | 20214250         |
| Sangamalla Santhosh Kumar| 20214277         |
| Sundram Mishra           | 20214020         |
| Seera Yashwant           | 20214521         |
| Sanidhya Diwakar         | 20214010         |

---


