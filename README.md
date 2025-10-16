# 🎓 Student Performance Prediction using R

This project predicts students' **academic grades** based on various input factors such as study habits, attendance, parental education, and other features.  
It uses **machine learning models** (Decision Tree, Random Forest, and Multinomial Logistic Regression) implemented in **R** to analyze and classify student performance.

---

## 📂 Project Overview

The goal of this project is to predict the **Grade** of a student using behavioral and academic features from a dataset.  
The model helps teachers and institutions identify students who may need additional support.

---

## ⚙️ Technologies Used

- **Language:** R  
- **Libraries:**  
  `tidyverse`, `caret`, `randomForest`, `nnet`, `rpart`, `e1071`, `janitor`, `pROC`, `GGally`  

---

## 📊 Dataset

The dataset used is `Students Performance Dataset.csv`, which contains student-related information such as:
- Gender, Parental Education, Study Time, Attendance, etc.  
- The target variable is **Grade** (e.g., A, B, C, D, F).  

> Note: The column `Total_Score` was removed from training to prevent **data leakage** and make the model more realistic.

---

## 🤖 Models Used

1. **Decision Tree (rpart)**  
2. **Random Forest (rf)**  
3. **Multinomial Logistic Regression (multinom)**  

Among these, the **Random Forest model** showed the highest accuracy (~90%).  

---

## 🚀 Steps Performed

1. Load and clean the dataset  
2. Convert categorical columns to factors  
3. Remove missing or unnecessary columns (like IDs, names, total_score)  
4. Split the dataset (80% training, 20% testing)  
5. Train models using **caret** package with cross-validation  
6. Evaluate models using confusion matrix and accuracy metrics  
7. Save the best model (`rf_grade_model.rds`) for future prediction  

---

## 📈 Model Evaluation

| Model | Accuracy |
|-------|-----------|
| Decision Tree | ~0.82 |
| Random Forest | ~0.90 |
| Multinomial Logistic Regression | ~0.84 |

> Accuracy values are between **0 and 1**, where **1 means 100% accuracy**.

---

## 🧩 Example Prediction

The trained Random Forest model can predict the **Grade** of a new student record:

```r
new_student <- test[1, ] %>% select(-Grade)
predict(model_rf, new_student)
predict(model_rf, new_student, type = "prob")
