############################################################
# Deep Learning in Health (BTC1889H)
# Final Project: Deep Learning Models for Text Data
#
# Team Members:
# - Owais Amir
# - Lamees Anjum
# - Anna Fan
# - Anthony Pede
# - Ann Zhang
############################################################

rm(list = ls())

###############################################################################
# SECTION A — DATA PREPROCESSING
###############################################################################

# ============================================================
# 1) Load packages
# ============================================================
library(stringr)
library(keras3)

# ============================================================
# 2) Read training and test datasets
# ============================================================
train <- read.csv("data/Corona_NLP_train.csv", stringsAsFactors = FALSE)
test  <- read.csv("data/Corona_NLP_test.csv", stringsAsFactors = FALSE)

# ============================================================
# 3) Explore raw data structure
# ============================================================
dim(train)
dim(test)

names(train)
names(test)

str(train)
str(test)

head(train)
head(test)

# ============================================================
# 4) Keep only required columns
# OriginalTweet = input text
# Sentiment     = target label
# ============================================================
train <- train[, c("OriginalTweet", "Sentiment")]
test  <- test[, c("OriginalTweet", "Sentiment")]

# ============================================================
# 5) Save raw tweet text before cleaning
# ============================================================
train_tweets_raw <- train$OriginalTweet
test_tweets_raw  <- test$OriginalTweet

# ============================================================
# 6) Check for missing values and empty labels
# ============================================================
colSums(is.na(train))
colSums(is.na(test))

sum(trimws(train$Sentiment) == "", na.rm = TRUE)
sum(trimws(test$Sentiment) == "", na.rm = TRUE)

# ============================================================
# 7) Inspect the sentiment labels
# Expected classes:
# Extremely Negative, Negative, Neutral, Positive, Extremely Positive
# ============================================================
table(train$Sentiment, useNA = "ifany")
table(test$Sentiment, useNA = "ifany")

unique(train$Sentiment)
unique(test$Sentiment)

# ============================================================
# 8) Define a function to clean the tweets
# ============================================================
# This function removes Twitter-specific noise while preserving the main text.
clean_text <- function(x) {
  x <- iconv(x, from = "", to = "UTF-8", sub = "")
  x <- gsub("\u0092", "'", x, fixed = TRUE)
  x <- gsub("\u0093", "\"", x, fixed = TRUE)
  x <- gsub("\u0094", "\"", x, fixed = TRUE)
  x <- gsub("\u0095", " ", x, fixed = TRUE)
  x <- gsub("\u0096", " ", x, fixed = TRUE)
  x <- gsub("\u0097", " ", x, fixed = TRUE)
  
  x <- gsub("\\\\u0092", "'", x)
  x <- gsub("\\\\u0093|\\\\u0094", "\"", x)
  x <- gsub("\\\\u0095|\\\\u0096|\\\\u0097", " ", x)
  x <- gsub("\\\\xc2", "", x)
  
  x <- gsub("http\\S+|www\\S+", " ", x)
  x <- gsub("@\\w+", " ", x)
  x <- gsub("\\bRT\\b", " ", x)
  x <- gsub("&amp;", "and", x)
  x <- gsub("\\\\n", " ", x)
  x <- gsub("\n", " ", x)
  x <- gsub("\\s+", " ", x)
  x <- trimws(x)
  
  return(x)
}

# ============================================================
# 9) Apply the cleaning function
# ============================================================
train$OriginalTweet <- clean_text(train$OriginalTweet)
test$OriginalTweet  <- clean_text(test$OriginalTweet)

# ============================================================
# 10) Compare raw vs cleaned tweets for changed rows
# ============================================================
changed_rows <- which(train_tweets_raw != train$OriginalTweet)

length(changed_rows)
length(changed_rows) / length(train_tweets_raw)

data.frame(
  row    = head(changed_rows, 10),
  before = train_tweets_raw[head(changed_rows, 10)],
  after  = train$OriginalTweet[head(changed_rows, 10)]
)

# ============================================================
# 11) Check for empty tweets after cleaning
# ============================================================
sum(trimws(train$OriginalTweet) == "", na.rm = TRUE)
sum(trimws(test$OriginalTweet) == "", na.rm = TRUE)

# ============================================================
# 12) Count words in each cleaned tweet
# \S+ counts sequences of non-space characters
# ============================================================
tweet_lengths <- str_count(train$OriginalTweet, "\\S+")
test_lengths  <- str_count(test$OriginalTweet, "\\S+")

# ============================================================
# 13) Inspect rows that became empty after cleaning
# ============================================================
zero_rows <- which(tweet_lengths == 0)

zero_rows

data.frame(
  row    = zero_rows,
  before = train_tweets_raw[zero_rows],
  after  = train$OriginalTweet[zero_rows]
)

# ============================================================
# 14) Remove tweets with 0 words after cleaning
# ============================================================
train <- train[tweet_lengths > 0, ]
test  <- test[test_lengths > 0, ]

# ============================================================
# 15) Recalculate tweet lengths after removing empty rows
# ============================================================
tweet_lengths <- str_count(train$OriginalTweet, "\\S+")
test_lengths  <- str_count(test$OriginalTweet, "\\S+")

# ============================================================
# 16) Choose preprocessing parameters
# maxlen = number of tokens allowed per tweet
# ============================================================
summary(tweet_lengths)
quantile(tweet_lengths, probs = c(0.50, 0.75, 0.90, 0.95, 0.99))

# Most tweets are under 50 words
num_words <- 10000
maxlen    <- 50

# ============================================================
# 17) Create text vectorization layer
# ============================================================
vec <- layer_text_vectorization(
  max_tokens             = num_words,
  standardize            = "lower_and_strip_punctuation",
  split                  = "whitespace",
  output_mode            = "int",
  output_sequence_length = maxlen
)

# ============================================================
# 18) Build vocabulary from training text only
# ============================================================
adapt(vec, train$OriginalTweet)

# ============================================================
# 19) Inspect the learned vocabulary
# ============================================================
vocab <- vec$get_vocabulary()

vocab
cat("Found", length(vocab), "tokens in learned vocabulary.\n")
head(vocab, 50)

# ============================================================
# 20) Create word-to-index mapping
# ============================================================
word_index <- stats::setNames(seq_along(vocab), vocab)
head(word_index, 30)

# ============================================================
# 21) Convert tweets into integer sequences
# ============================================================
seq_tensor_train <- vec(train$OriginalTweet)
seq_tensor_test  <- vec(test$OriginalTweet)

class(seq_tensor_train)

x_train <- as.array(seq_tensor_train)
x_test  <- as.array(seq_tensor_test)

dim(x_train)
dim(x_test)

x_train[1:2, 1:20]

# ============================================================
# 22) Encode sentiment labels
# ============================================================
label_levels <- c(
  "Extremely Negative",
  "Negative",
  "Neutral",
  "Positive",
  "Extremely Positive"
)

train$Sentiment <- factor(train$Sentiment, levels = label_levels)
test$Sentiment  <- factor(test$Sentiment, levels = label_levels)

y_train <- as.integer(train$Sentiment) - 1
y_test  <- as.integer(test$Sentiment) - 1

# ============================================================
# 23) Check encoded label distributions
# ============================================================
table(y_train)
table(y_test)

# ============================================================
# 24) Define custom ordinal MAE metric and weighted kappa metric
# ============================================================
metric_ordinal_mae <- custom_metric(
  "ordinal_mae",
  function(y_true, y_pred) {
    
    # Take argmax of softmax output to get predicted class
    y_pred_class <- op_cast(op_argmax(y_pred, axis = -1L), "float32")
    
    # Keep original version of y_true handling
    y_true_class <- op_cast(y_true, "float32")
    
    # Mean absolute difference between class indices
    op_mean(op_abs(y_true_class - y_pred_class))
  }
)

# Compute quadratic weighted kappa from predicted classes
compute_weighted_kappa <- function(actual, predicted, k = 5) {
  cm <- matrix(0L, nrow = k, ncol = k)
  
  for (i in seq_along(actual)) {
    cm[actual[i] + 1L, predicted[i] + 1L] <-
      cm[actual[i] + 1L, predicted[i] + 1L] + 1L
  }
  
  n <- sum(cm)
  w <- outer(0:(k - 1), 0:(k - 1),
             function(i, j) (i - j)^2 / (k - 1)^2)
  E <- outer(rowSums(cm), colSums(cm)) / n
  
  round(1 - sum(w * cm) / sum(w * E), 4)
}

# ============================================================
# 25) Set global model parameters
# ============================================================
embedding_dim <- 64
lstm_units    <- 64
num_classes   <- 5
batch_size    <- 32
epochs        <- 10
val_split     <- 0.2

# ============================================================
# 26) Define helper functions
# ============================================================

# Compile all models with the same settings
compile_model <- function(model) {
  model |> compile(
    optimizer = "adam",
    loss      = "sparse_categorical_crossentropy",
    metrics   = list("sparse_categorical_accuracy", metric_ordinal_mae)
  )
}


# Extract the final epoch value of a validation metric
final_val <- function(history, metric) {
  vals <- history$metrics[[metric]]
  if (is.null(vals)) return(NA_real_)
  round(tail(vals, 1), 4)
}

# ============================================================
# 27) Create one global stratified validation split
# ============================================================
# This validation set will be reused for FF, RNN, and LSTM so that all
# model families are compared on the exact same observations.

set.seed(123)

# For each class, sample 20% of indices into the validation set
val_idx <- unlist(lapply(split(seq_along(y_train), y_train), function(idx) {
  sample(idx, size = floor(length(idx) * val_split))
}))

# Keep indices sorted for easier inspection/reproducibility
val_idx <- sort(val_idx)

# Training indices are everything not in validation
train_idx <- setdiff(seq_len(nrow(x_train)), val_idx)

# Final stratified split
x_train_model <- x_train[train_idx, , drop = FALSE]
y_train_model <- y_train[train_idx]

x_val <- x_train[val_idx, , drop = FALSE]
y_val <- y_train[val_idx]

# Check class distributions
cat("Training set class distribution:\n")
print(prop.table(table(y_train_model)))

cat("\nValidation set class distribution:\n")
print(prop.table(table(y_val)))

###############################################################################
# SECTION B — FEED-FORWARD MODEL
###############################################################################

# ============================================================
# 1) Add feed-forward model code here
# ============================================================
# Keep the same subsection structure for the FF model:
# 1) Set model parameters
# 2) Define model builder functions
# 3) Define helper functions
# 4) Print model summaries
# 5) Train candidate models
# 6) Compare validation performance
# 7) Select the best model
# 8) Retrain the best model
# 9) Evaluate on the test set
# 10) Save results


###############################################################################
# SECTION C — RNN MODEL
###############################################################################

# ============================================================
# 1) Add RNN model code here
# ============================================================
# Keep the same subsection structure for the RNN model:
# 1) Set model parameters
# 2) Define model builder functions
# 3) Define helper functions
# 4) Print model summaries
# 5) Train candidate models
# 6) Compare validation performance
# 7) Select the best model
# 8) Retrain the best model
# 9) Evaluate on the test set
# 10) Save results


###############################################################################
# SECTION D — LSTM MODEL
###############################################################################

# ============================================================
# 1) Define model builder functions
# ============================================================

# Baseline:
# Embedding -> LSTM -> Dense(softmax)
build_baseline <- function() {
  keras_model_sequential(name = "baseline") |>
    layer_embedding(
      input_dim    = num_words,
      output_dim   = embedding_dim,
      input_length = maxlen
    ) |>
    layer_lstm(units = lstm_units) |>
    layer_dense(units = num_classes, activation = "softmax")
}

# Baseline + Dropout:
# Embedding -> LSTM(dropout) -> Dropout -> Dense(softmax)
build_baseline_dropout <- function() {
  keras_model_sequential(name = "baseline_dropout") |>
    layer_embedding(
      input_dim    = num_words,
      output_dim   = embedding_dim,
      input_length = maxlen
    ) |>
    layer_lstm(
      units             = lstm_units,
      dropout           = 0.3,
      recurrent_dropout = 0.2
    ) |>
    layer_dropout(rate = 0.3) |>
    layer_dense(units = num_classes, activation = "softmax")
}

# Stacked:
# Embedding -> LSTM(return_sequences = TRUE) -> LSTM -> Dense(softmax)
build_stacked <- function() {
  keras_model_sequential(name = "stacked") |>
    layer_embedding(
      input_dim    = num_words,
      output_dim   = embedding_dim,
      input_length = maxlen
    ) |>
    layer_lstm(
      units            = lstm_units,
      return_sequences = TRUE
    ) |>
    layer_lstm(units = lstm_units) |>
    layer_dense(units = num_classes, activation = "softmax")
}

# Stacked + Dropout:
# Embedding -> LSTM(dropout, recurrent_dropout, return_sequences = TRUE)
#           -> LSTM(dropout, recurrent_dropout)
#           -> Dropout
#           -> Dense(softmax)
build_stacked_dropout <- function() {
  keras_model_sequential(name = "stacked_dropout") |>
    layer_embedding(
      input_dim    = num_words,
      output_dim   = embedding_dim,
      input_length = maxlen
    ) |>
    layer_lstm(
      units             = lstm_units,
      dropout           = 0.3,
      recurrent_dropout = 0.2,
      return_sequences  = TRUE
    ) |>
    layer_lstm(
      units             = lstm_units,
      dropout           = 0.3,
      recurrent_dropout = 0.2
    ) |>
    layer_dropout(rate = 0.3) |>
    layer_dense(units = num_classes, activation = "softmax")
}

builders <- list(
  baseline         = build_baseline,
  baseline_dropout = build_baseline_dropout,
  stacked          = build_stacked,
  stacked_dropout  = build_stacked_dropout
)

# ============================================================
# 2) Run full LSTM workflow or load saved outputs
# ============================================================
# Set to TRUE only when you want to rerun all LSTM modelling steps.
if (F) {
  
  # ============================================================
  # 2a) Print model summaries
  # ============================================================
  cat("========== Model Architecture Summaries ==========\n")
  
  param_table <- lapply(names(builders), function(nm) {
    model <- builders[[nm]]()
    compile_model(model)
    
    # Force model to build before printing summary
    dummy <- matrix(1L, nrow = 1, ncol = maxlen)
    invisible(model(dummy))
    
    cat("\n---", nm, "---\n")
    summary(model)
    
    data.frame(
      Model        = nm,
      Total_Params = as.integer(count_params(model)),
      stringsAsFactors = FALSE
    )
  })
  
  param_table <- do.call(rbind, param_table)
  
  cat("\n========== Parameter Count Summary ==========\n")
  print(param_table, row.names = FALSE)
  
  # ============================================================
  # 2b) Train all four models
  # ============================================================
  # All models use the same global stratified validation set.
  
  histories <- list()
  models    <- list()
  
  for (nm in names(builders)) {
    cat("\n--- Training:", nm, "---\n")
    
    model <- builders[[nm]]()
    compile_model(model)
    
    histories[[nm]] <- model |> fit(
      x_train_model, y_train_model,
      epochs          = epochs,
      batch_size      = batch_size,
      validation_data = list(x_val, y_val),
      verbose         = 1
    )
    
    models[[nm]] <- model
  }
  
  # ============================================================
  # 2c) Compare validation performance
  # ============================================================
  # Models are ranked using:
  # 1. highest validation weighted kappa
  # 2. lowest validation ordinal MAE
  # 3. highest validation accuracy
  
  val_results <- do.call(rbind, lapply(names(models), function(nm) {
    val_probs <- models[[nm]] |> predict(x_val, verbose = 0)
    val_preds <- apply(val_probs, 1, which.max) - 1L
    
    data.frame(
      Model      = nm,
      Val_Kappa  = compute_weighted_kappa(y_val, val_preds),
      Val_OrdMAE = final_val(histories[[nm]], "val_ordinal_mae"),
      Val_Acc    = final_val(histories[[nm]], "val_sparse_categorical_accuracy"),
      stringsAsFactors = FALSE
    )
  }))
  
  cat("\n========== Validation Comparison ==========\n")
  print(val_results, row.names = FALSE)
  cat("===========================================\n")
  
  # ============================================================
  # 2d) Select the best model
  # ============================================================
  val_results$rank_kappa <- rank(-val_results$Val_Kappa,  ties.method = "first")
  val_results$rank_mae   <- rank( val_results$Val_OrdMAE, ties.method = "first")
  val_results$rank_acc   <- rank(-val_results$Val_Acc,    ties.method = "first")
  
  val_results <- val_results[
    order(val_results$rank_kappa, val_results$rank_mae, val_results$rank_acc), ]
  
  best_name <- val_results$Model[1]
  
  cat("\nBest model:", best_name, "\n")
  cat("  Validation Kappa  :", val_results$Val_Kappa[1],  "\n")
  cat("  Validation OrdMAE :", val_results$Val_OrdMAE[1], "\n")
  cat("  Validation Acc    :", val_results$Val_Acc[1],    "\n")
  
  # Keep only the main comparison columns
  val_results <- val_results[, c("Model", "Val_Kappa", "Val_OrdMAE", "Val_Acc")]
  
  # ============================================================
  # 2e) Retrain the best model on the full training set
  # ============================================================
  cat("\n--- Retraining best model (", best_name, ") on full training set ---\n")
  
  best_model <- builders[[best_name]]()
  compile_model(best_model)
  
  history_best <- best_model |> fit(
    x_train, y_train,
    epochs     = epochs,
    batch_size = batch_size,
    verbose    = 1
  )
  
  save_model(best_model, "best_lstm_model.keras")
  cat("Best model saved to best_lstm_model.keras\n")
  
  # ============================================================
  # 2f) Evaluate the best model on the test set
  # ============================================================
  cat("\n--- Test Set Evaluation:", best_name, "---\n")
  
  test_eval  <- best_model |> evaluate(x_test, y_test, verbose = 0)
  test_probs <- best_model |> predict(x_test, verbose = 0)
  test_preds <- apply(test_probs, 1, which.max) - 1L
  
  test_acc   <- round(as.numeric(test_eval[["sparse_categorical_accuracy"]]), 4)
  test_mae   <- round(as.numeric(test_eval[["ordinal_mae"]]), 4)
  test_kappa <- compute_weighted_kappa(y_test, test_preds)
  
  cat("  Accuracy       :", test_acc,   "\n")
  cat("  Ordinal MAE    :", test_mae,   "\n")
  cat("  Weighted Kappa :", test_kappa, "\n")
  
  test_cm <- table(Actual = y_test, Predicted = test_preds)
  
  cat("\nConfusion Matrix:\n")
  print(test_cm)
  
  cat("\nPer-class accuracy:\n")
  print(round(diag(test_cm) / rowSums(test_cm), 3))
  
  # ============================================================
  # 2g) Save LSTM results
  # ============================================================
  save(
    param_table,
    histories,
    val_results,
    best_name,
    history_best,
    test_eval,
    test_probs,
    test_preds,
    test_cm,
    test_acc,
    test_mae,
    test_kappa,
    file = "lstm_results.RData"
  )
  
  cat("\nResults saved to lstm_results.RData\n")
  
} else {
  
  # ============================================================
  # 3a) Load saved model and results
  # ============================================================
  cat("\n--- Loading saved LSTM model and results ---\n")
  
  best_model <- load_model(
    "best_lstm_model.keras",
    custom_objects = list(ordinal_mae = metric_ordinal_mae)
  )
  load("lstm_results.RData")
  
  cat("Loaded best model:", best_name, "\n")
  cat("Loaded results from lstm_results.RData\n")
  
  # ============================================================
  # 3b) Reprint saved outputs
  # ============================================================
  cat("\n========== Parameter Count Summary ==========\n")
  print(param_table, row.names = FALSE)
  
  cat("\n========== Validation Comparison ==========\n")
  print(val_results, row.names = FALSE)
  cat("===========================================\n")
  
  cat("\n--- Saved Test Results ---\n")
  cat("  Accuracy       :", test_acc,   "\n")
  cat("  Ordinal MAE    :", test_mae,   "\n")
  cat("  Weighted Kappa :", test_kappa, "\n")
  
  cat("\nConfusion Matrix:\n")
  print(test_cm)
  
  cat("\nPer-class accuracy:\n")
  print(round(diag(test_cm) / rowSums(test_cm), 3))
}

# ============================================================
# 4) Generate plots and tables
# ============================================================

library(ggplot2)
library(patchwork)

# Use the validation history from the selected model
history_plot <- histories[[best_name]]

# Build data frame
df <- data.frame(
  epoch    = 1:length(history_plot$metrics$loss),
  loss     = history_plot$metrics$loss,
  val_loss = history_plot$metrics$val_loss,
  acc      = history_plot$metrics$sparse_categorical_accuracy,
  val_acc  = history_plot$metrics$val_sparse_categorical_accuracy,
  mae      = history_plot$metrics$ordinal_mae,
  val_mae  = history_plot$metrics$val_ordinal_mae
)

# Common theme
pub_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title      = element_text(face = "bold", size = 12, hjust = 0.5),
    axis.title      = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title    = element_blank(),
    panel.grid.minor = element_blank(),
    strip.text      = element_text(face = "bold")
  )

# Accuracy panel
p_acc <- ggplot(df, aes(x = epoch)) +
  geom_line(aes(y = acc, color = "Training"), linewidth = 1) +
  geom_line(aes(y = val_acc, color = "Validation"), linewidth = 1) +
  labs(
    title = "Accuracy",
    x = "Epoch",
    y = "Accuracy"
  ) +
  scale_color_manual(values = c("Training" = "#1f77b4", "Validation" = "#d62728")) +
  pub_theme

# Loss panel
p_loss <- ggplot(df, aes(x = epoch)) +
  geom_line(aes(y = loss, color = "Training"), linewidth = 1) +
  geom_line(aes(y = val_loss, color = "Validation"), linewidth = 1) +
  labs(
    title = "Loss",
    x = "Epoch",
    y = "Loss"
  ) +
  scale_color_manual(values = c("Training" = "#1f77b4", "Validation" = "#d62728")) +
  pub_theme

# Ordinal MAE panel
p_mae <- ggplot(df, aes(x = epoch)) +
  geom_line(aes(y = mae, color = "Training"), linewidth = 1) +
  geom_line(aes(y = val_mae, color = "Validation"), linewidth = 1) +
  labs(
    title = "Ordinal MAE",
    x = "Epoch",
    y = "Ordinal MAE"
  ) +
  scale_color_manual(values = c("Training" = "#1f77b4", "Validation" = "#d62728")) +
  pub_theme

# Combine into one publication-ready figure
final_fig <- (p_acc / p_loss / p_mae) +
  plot_annotation(
    title = "Training and Validation Performance of the Selected Stacked LSTM Model",
    theme = theme(
      plot.title = element_text(face = "bold", size = 14, hjust = 0.5)
    )
  )

final_fig

ggsave(
  filename = "lstm_training_curves.png",
  plot = final_fig,
  width = 12,
  height = 12,
  dpi = 300
)
