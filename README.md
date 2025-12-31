# 📰 Fake News Detection System (Python + GUI)

A Machine Learning based **Fake News Detection** project built using **Python**, **Random Dataset (CSV)**, and a **Graphical User Interface (GUI)**.  
This system predicts whether news is **Real** or **Fake**, shows **dataset-wide predictions**, displays **percentage graphs**, and allows **custom user input** for prediction beyond the dataset.

---

## 📌 Features

✅ Uses CSV dataset (Fake & Real news)  
✅ Predicts **Fake / Real** for the entire dataset  
✅ Displays predictions in **GUI (not terminal)**  
✅ Shows **count & percentage graph** (Bar / Pie Chart)  
✅ Allows **custom news input** for prediction  
✅ Clean and beginner-friendly project structure  
✅ Ideal for **college projects, placements & GitHub portfolio**

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Libraries Used:**
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - tkinter
  - seaborn
- **Machine Learning Model:** Logistic Regression / Naive Bayes  
- **Dataset Format:** CSV  

---

## 📂 Project Structure

Fake-News-Detection/
│
├── fake_news_dataset.csv
├── fake.py
├── requirements.txt
└── README.md

---

## 📊 Dataset Description

The dataset contains news articles with labels:

| Column Name | Description |
|------------|------------|
| text | News article content |
| label | 0 = Fake News, 1 = Real News |

---

## 🧠 How the System Works

1. Load CSV dataset  
2. Preprocess text (remove stopwords, vectorize)  
3. Train ML model  
4. Predict **Fake / Real** for all dataset rows  
5. Show results in **GUI table**  
6. Display **percentage graph**  
7. Accept **custom news input** and predict output  

---

## 🖥️ GUI Output Includes

- 📋 Dataset Prediction List  
- 📈 Real vs Fake News Percentage Graph  
- ✍️ Custom News Input Box  
- 🔍 Predict Button  
- 📊 Result Display (Fake / Real)

---

## 🚀 How to Run the Project

### Step 1: Clone Repository
```bash
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
```
