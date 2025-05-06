# 🚦 AI-Powered Traffic Light Management System

This project uses **Machine Learning algorithms** to optimize traffic light timings dynamically based on real-time traffic data. The goal is to reduce overall vehicle wait times and congestion by making traffic signals adaptive and intelligent.

---

## 🧠 Algorithms Used

- **K-Nearest Neighbors (KNN)**  
- **Decision Tree**  
- **Random Forest** (Best Performing Model)

Each algorithm was trained and evaluated to predict the optimal green light duration for each lane depending on live traffic conditions such as vehicle count and lane density.

---

## 🎯 Objective

- Analyze real-time traffic data (simulated or sensor-based).
- Dynamically adjust signal durations based on traffic patterns.
- Reduce overall wait time, congestion, and fuel consumption.
- Compare multiple ML models to identify the most efficient algorithm.

---

## ⚙️ Tech Stack

| Category            | Tools/Technologies       |
|---------------------|---------------------------|
| Programming Language| Python                   |
| ML Libraries        | Scikit-learn, Pandas, NumPy |
| Visualization       | Matplotlib, Seaborn       |
| Development Tools   | Jupyter Notebook, VS Code |
| Version Control     | Git, GitHub               |
| (Optional) DevOps   | Docker, AWS (future scope) |

---

## 🧪 Model Performance

| Model         | Accuracy | Remarks                      |
|---------------|----------|------------------------------|
| KNN           | Moderate | Simple but less scalable     |
| Decision Tree | Good     | Fast, interpretable          |
| Random Forest | 🔥 Best   | High accuracy & robust       |

Random Forest gave the most consistent results across multiple simulations, handling noisy and imbalanced traffic patterns effectively.

---

## 📊 Sample Features Used

- Vehicle count per lane  
- Average waiting time  
- Lane congestion ratio  
- Time of day (peak/non-peak)  

---
