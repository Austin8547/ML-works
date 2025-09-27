# naive-bayes-classifier
# 📩 SMS Spam Detection using Naive Bayes

A simple and effective spam classifier built using the **Multinomial Naive Bayes** algorithm and **CountVectorizer** for feature extraction. This project classifies SMS messages as either **Spam** or **Ham (not spam)**.

---

## 📂 Dataset Used

**SMS Spam Collection Dataset**  
- 📁 Total Messages: 5,574  
- 🏷️ Labels: `ham`, `spam`  
- 📄 Features:
  - `v1`: Message label (ham/spam)
  - `v2`: Message content

📌 Source: [Kaggle - Spam and Ham Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 🔍 Project Workflow

1. **Import Required Libraries**
2. **Load the Dataset**
3. **Clean and Prepare the Data**
   - Drop unnecessary columns
   - Map labels: `ham → 0`, `spam → 1`
4. **Text Vectorization**
   - Convert SMS text to numerical features using **CountVectorizer**
5. **Model Training**
   - Train a **Multinomial Naive Bayes** classifier
6. **Model Evaluation**
   - Use **accuracy, precision, recall, F1-score**
   - Visualize results with a **confusion matrix (heatmap)**
7. **Custom Testing Functions**
   - `check_message(message)` – predict for a single SMS
   - `check_spam(messages)` – predict for multiple SMS messages

---

## 📊 Evaluation Metrics

- ✅ **Accuracy**: Measures overall correctness  
- 🎯 **Precision**: How many predicted spams were actually spam  
- 🔁 **Recall**: How many actual spams were correctly predicted  
- 🧮 **F1 Score**: Balance between precision and recall  

---


---

## 📐 Behind the Math

### 🧠 1. Multinomial Naive Bayes

Naive Bayes is a **probabilistic classifier** based on **Bayes' Theorem**:

\[
P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}
\]

Where:
- \( Y \) is the label (spam or ham)
- \( X \) is the message (represented as a vector of word counts or TF-IDF)
- \( P(Y \mid X) \) is the probability of class \( Y \) given the message \( X \)

Since we're using the **Multinomial version**, we assume:
- Words are independent of each other (naive assumption)
- Word frequency matters

The prediction is made by calculating:

\[
\hat{y} = \arg\max_y \left( \log P(y) + \sum_{i=1}^{n} x_i \cdot \log P(w_i \mid y) \right)
\]

Where:
- \( x_i \) = frequency of word \( w_i \) in the message
- \( P(w_i \mid y) \) = probability of word \( w_i \) occurring in class \( y \)

---

### 🧮 2. CountVectorizer (Bag-of-Words)

Before applying Naive Bayes, we convert text into numbers using **CountVectorizer**:

- It builds a vocabulary of all unique words in the training set
- Each message becomes a vector showing **how many times** each word appears
- Resulting matrix = **(number of messages) × (number of unique words)**

> Example:
>
> `"Free offer now"` → `[1, 1, 1, 0, 0, ...]`

This gives the input required by the Naive Bayes model.

---

### 🤔 Why It Works Well

- Naive Bayes is simple but powerful for **text classification**
- Works well even with **small datasets**
- CountVectorizer or TF-IDF makes the model **numerically tractable**


