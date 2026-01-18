# Machine Learning Foundations: Regression, Classification & NLP

This project is a comprehensive Machine Learning suite developed for a university-level Machine Learning course. It implements fundamental ML algorithms—Polynomial Regression, k-Nearest Neighbors, and Multinomial Naive Bayes—from scratch using NumPy and TensorFlow to solve real-world data problems.

The project is divided into four main sections, covering everything from theoretical NLP concepts to practical implementation and model optimization.

---

## 🛠 Tech Stack

*   **Language:** Python 3.x
*   **Deep Learning Framework:** TensorFlow (Manual optimization via GradientTape)
*   **Data Analysis:** Pandas, NumPy
*   **Natural Language Processing:** NLTK (Tokenization, Porter Stemmer, Stopwords)
*   **Visualization:** Matplotlib, Seaborn

---

## 🚀 Project Components

### 1. Research & Theory
A study of core concepts in modern Machine Learning:
*   **Word2Vec:** Exploring the semantic meaning of words represented as feature vectors.
*   **Naive Bayes Methodologies:** Analysis of Gaussian (continuous), Multinomial (discrete frequencies), and Bernoulli (binary features) classifiers.
*   **Linear Separability:** Determining if classes in the Iris dataset can be separated by linear boundaries.

### 2. Seawater Temperature Prediction (Regression)
**Dataset:** `bottle.csv` (Oceanographic data) available [here](https://www.kaggle.com/datasets/sohier/calcofi?select=bottle.csv)
*   **Approach:** Polynomial Regression (Degrees 1 to 6) used to model the relationship between Salinity and Temperature.
*   **Regularization:** Manual implementation of L2 (Ridge) Regularization. 
*   **Analysis:** Testing a set of lambda values $\{0, 0.001, 0.01, 0.1, 1, 10, 100\}$ to observe the impact on the model's complexity and final cost function.

### 3. Biological Classification (k-Nearest Neighbors)
**Dataset:** `iris.csv` available [here](/data/iris.csv)
*   **Algorithm:** Custom implementation of a non-weighted k-NN classifier.
*   **Optimization:** Experimental testing of k-values from 1 to 15.
*   **Comparison:** Analyzing model accuracy using a subset of features (2D sepal dimensions) versus using the full feature set (4D).

### 4. Disaster Tweet Analysis (NLP & Naive Bayes)
**Dataset:** `disaster-tweets.csv` available [here](https://www.kaggle.com/competitions/nlp-getting-started/data?select=train.csv)
*   **Cleaning Pipeline:** Implemented text tokenization, Porter Stemming, and noise removal using regular expressions for links, emojis, and punctuation.
*   **Feature Engineering:** Built frequency vectors using Bag-of-Words (BoW).
*   **Model:** A Multinomial Naive Bayes classifier built from scratch.
*   **Analytics:** Using Likelihood Ratio (LR) to identify the top predictive keywords for "Disaster" vs. "Normal" tweet contexts.

---

## 📦 Installation & Usage

### 1. Requirements
Ensure you have all necessary libraries installed:
```bash
pip install tensorflow pandas numpy nltk matplotlib seaborn scikit-learn
```

### 2. Running the Notebooks
Each task is organized in a standalone notebook to maintain clarity and reproducibility:
```bash
jupyter notebook 2a.ipynb   # Polynomial Regression Complexity
jupyter notebook 3a.ipynb   # k-NN Implementation
jupyter notebook 4.ipynb    # NLP Preprocessing & Classification
```

---

## 📊 Summary of Findings

*   **Optimal Fit:** The project demonstrates that while higher polynomial degrees can reduce training error, L2 regularization is essential to maintain model flexibility and prevent overfitting.
*   **Feature Importance:** The NLP section proves that text cleaning (stemming and stopword removal) significantly boosts accuracy, with the Multinomial Naive Bayes model reaching over 80% precision on unseen data.
*   **The Likelihood Ratio:** Calculating LR provides intuitive insight into why specific words trigger a "Disaster" classification, highlighting the "explainability" of simpler probabilistic models.

---

## 📂 Project Structure

```text
├── 1.txt            # Theoretical Questions & NLP Research
├── 2a.ipynb         # Regression Complexity (Polynom Degrees)
├── 2b.ipynb         # Regression Regularization (L2 / Lambda)
├── 3a.ipynb         # k-NN 2D Class boundaries
├── 3b/c.ipynb       # Hyperparameter optimization (k-values)
├── 4.ipynb          # NLP Preprocessing & Naive Bayes model
└── data/            # Datasets (Iris, Bottle, Tweets)
```
