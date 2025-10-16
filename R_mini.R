# Install only missing packages
packages <- c("tidyverse", "caret", "randomForest", "nnet", "rpart", "e1071", "janitor", "pROC", "GGally")
to_install <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(to_install)) install.packages(to_install)

# Load libraries
library(tidyverse)
library(caret)
library(randomForest)
library(nnet)    # multinom
library(rpart)
library(e1071)   # required by caret::confusionMatrix
library(janitor) # clean names
library(pROC)
library(GGally)

setwd("C:/Program Files/RStudio")  # uncomment + edit if needed

data <- read_csv("Students Performance Dataset.csv")
glimpse(data)
# show first rows & column names
head(data)
names(data)


# Correct way: pass the whole dataframe
data <- janitor::clean_names(data)

# Now check column names
names(data)

# Drop obvious PII columns (adjust names if different)
possible_pii <- c("student_id", "first_name", "last_name", "email")
existing_pii <- intersect(possible_pii, names(data))
if(length(existing_pii)) data <- data %>% select(-all_of(existing_pii))

# Identify target column (case-insensitive)
target_idx <- which(tolower(names(data)) == "grade")
if(length(target_idx) != 1) stop("Could not find a single 'Grade' column. Check column names: ", paste(names(data), collapse=", "))
target_col <- names(data)[target_idx]
cat("Using target column:", target_col, "\n")

# Convert character columns to factors where appropriate
data <- data %>%
  mutate(across(where(is.character), as.factor))

# Ensure target is factor
data[[target_col]] <- as.factor(data[[target_col]])

# Quick check for missing values
sum_na <- sum(is.na(data))
cat("Total missing values:", sum_na, "\n")
if(sum_na > 0) {
  # Simple approach: remove rows with NA (if few); otherwise you'll need specific imputation
  cat("Dropping rows with missing values (consider imputation if many missing).\n")
  data <- na.omit(data)
}


# Frequency of grades
table(data[[target_col]])

# Basic numeric summary
data %>% select(where(is.numeric)) %>% summarise_all(list(~mean(., na.rm=TRUE), ~sd(., na.rm=TRUE))) %>% print()

# Visual: grade count
ggplot(data, aes(x = !!sym(target_col))) +
  geom_bar() + labs(title = "Grade distribution", x = "Grade", y = "Count")

# Pairwise plot for a few numeric columns (adjust columns as needed)
num_cols <- names(data)[sapply(data, is.numeric)]
if(length(num_cols) >= 4) {
  ggpairs(data %>% select(all_of(num_cols[1:4])))
}

data <- data %>% select(-total_score)

set.seed(123)
train_index <- createDataPartition(data[[target_col]], p = 0.8, list = FALSE)
train <- data[train_index, ]
test  <- data[-train_index, ]

cat("Train rows:", nrow(train), "Test rows:", nrow(test), "\n")
prop.table(table(train[[target_col]]))
prop.table(table(test[[target_col]]))


# train control with 5-fold CV and class probabilities
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, savePredictions = "final")

formula_text <- paste(target_col, "~ .")
fmla <- as.formula(formula_text)

# 1) Decision tree (rpart)
set.seed(123)
model_rpart <- train(fmla, data = train, method = "rpart", trControl = ctrl)
model_rpart

# 2) Random Forest
set.seed(123)
model_rf <- train(fmla, data = train, method = "rf", trControl = ctrl, importance = TRUE)
model_rf

# 3) Multinomial logistic regression (from nnet)
set.seed(123)
model_multinom <- train(fmla, data = train, method = "multinom", trControl = ctrl, trace = FALSE)
model_multinom

# Helper to evaluate
eval_model <- function(mod, test_df, target_col) {
  preds <- predict(mod, newdata = test_df)
  cm <- confusionMatrix(preds, test_df[[target_col]])
  list(model = mod, preds = preds, confusion = cm)
}

res_rpart <- eval_model(model_rpart, test, target_col)
res_rf    <- eval_model(model_rf, test, target_col)
res_multi <- eval_model(model_multinom, test, target_col)

# Print accuracies
cat("Decision Tree Accuracy:", res_rpart$confusion$overall["Accuracy"], "\n")
cat("Random Forest Accuracy: ", res_rf$confusion$overall["Accuracy"], "\n")
cat("Multinomial Accuracy:   ", res_multi$confusion$overall["Accuracy"], "\n")

# Show confusion matrix for best model (choose based on accuracy)
print(res_rf$confusion)   # example: print RF confusion matrix


# Variable importance
rf_imp <- varImp(model_rf)
print(rf_imp)
plot(rf_imp, top = 20)  # top features

# Pick model_rf as example best model (if it's actually best for you)
saveRDS(model_rf, file = "rf_grade_model.rds")

# To load later:
# loaded_model <- readRDS("rf_grade_model.rds")

# Predict probability & class for a new student (example: take first test row)
new_student <- test[1, ] %>% select(-all_of(target_col))
predict(model_rf, new_student)              # predicted class
predict(model_rf, new_student, type = "prob") # class probabilities

getwd()
setwd("C:/Users/Yoges/Documents/RProjects")
saveRDS(model_rf, file = "C:/Users/Yogeswaran/Desktop/rf_grade_model.rds")



