import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------
df = pd.read_csv("fake_news_dataset.csv")

X = df["text"]
y = df["label"]

# ---------------- TRAIN MODEL ----------------
vectorizer = TfidfVectorizer(stop_words="english")
X_tfidf = vectorizer.fit_transform(X)

model = PassiveAggressiveClassifier(max_iter=100)
model.fit(X_tfidf, y)

# Predict entire dataset
df["Predicted"] = model.predict(X_tfidf)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Fake News Detection Dashboard")
root.geometry("1100x650")
root.configure(bg="#f2f3f4")

# ---------------- TITLE ----------------
tk.Label(
    root,
    text="Fake News Detection System",
    font=("Helvetica", 22, "bold"),
    bg="#f2f3f4"
).pack(pady=10)

# ---------------- TABLE ----------------
frame_table = tk.Frame(root)
frame_table.pack(padx=10, pady=10, fill="both", expand=True)

columns = ("Title", "Actual", "Predicted")
tree = ttk.Treeview(frame_table, columns=columns, show="headings", height=10)

tree.heading("Title", text="News Title")
tree.heading("Actual", text="Actual Label")
tree.heading("Predicted", text="Predicted Label")

tree.column("Title", width=500)
tree.column("Actual", width=120, anchor="center")
tree.column("Predicted", width=120, anchor="center")

for _, row in df.iterrows():
    tree.insert("", tk.END, values=(row["title"], row["label"], row["Predicted"]))

tree.pack(fill="both", expand=True)

# ---------------- GRAPH FUNCTION ----------------
def show_graph():
    counts = df["Predicted"].value_counts()
    labels = counts.index
    values = counts.values

    plt.figure(figsize=(5, 4))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title("Fake vs Real News Percentage")
    plt.show()

tk.Button(
    root,
    text="Show Percentage Graph",
    font=("Arial", 12, "bold"),
    bg="#3498db",
    fg="white",
    command=show_graph
).pack(pady=10)

# ---------------- CUSTOM INPUT ----------------
tk.Label(
    root,
    text="Check Your Own News:",
    font=("Arial", 14, "bold"),
    bg="#f2f3f4"
).pack(pady=10)

custom_text = tk.Text(root, height=4, width=90)
custom_text.pack(pady=5)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 16, "bold"),
    bg="#f2f3f4"
)
result_label.pack(pady=10)

def predict_custom_news():
    news = custom_text.get("1.0", tk.END).strip()
    if not news:
        result_label.config(text="⚠ Please enter news text", fg="orange")
        return

    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]

    if prediction == "FAKE":
        result_label.config(text="🛑 FAKE NEWS", fg="red")
    else:
        result_label.config(text="✅ REAL NEWS", fg="green")

tk.Button(
    root,
    text="Predict News",
    font=("Arial", 12, "bold"),
    bg="#2ecc71",
    fg="white",
    command=predict_custom_news
).pack(pady=10)

root.mainloop()
