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
# ============================================================
# 1) Load packages
# ============================================================
library(stringr)
library(keras3)

# ============================================================
# 2) Read training and test datasets
# ============================================================
train <- read.csv("Corona_NLP_train.csv", stringsAsFactors = FALSE)
test  <- read.csv("Corona_NLP_test.csv", stringsAsFactors = FALSE)

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
# 4) Keep only the columns:
# OriginalTweet = input text
# Sentiment = label / target
# ============================================================
train <- train[, c("OriginalTweet", "Sentiment")]
test  <- test[, c("OriginalTweet", "Sentiment")]


# ============================================================
# 5) Save raw tweet text before cleaning
# ============================================================
train_tweets_raw <- train$OriginalTweet
test_tweets_raw  <- test$OriginalTweet

# ============================================================
# 6) Check for missing values & empty strings
# check both NA and ""
# ============================================================
colSums(is.na(train))
colSums(is.na(test))

sum(trimws(train$Sentiment) == "", na.rm = TRUE)
sum(trimws(test$Sentiment) == "", na.rm = TRUE)

# ============================================================
# 7) Inspect the sentiment labels
#  should have the classes:
# Extremely Negative, Negative, Neutral, Positive, Extremely Positive
# ============================================================
table(train$Sentiment, useNA = "ifany")
table(test$Sentiment, useNA = "ifany")

unique(train$Sentiment)
unique(test$Sentiment)

# ============================================================
# 8) Define a function to clean the tweets
# This function removes Twitter-specific noise while preserving the main text based on 
# common practices

# The function cleans the following: 
  # fix encoding issues found in EDA
  # remove URLs
  # remove @mentions
  # removes RT retweet marker
  # fixes HTML ampersands
  # removes line breaks
  # collapses extra spaces
  # removes other symbols found in the data
# ============================================================
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
# 9) Apply the cleaning function to both train and test
# ============================================================
train$OriginalTweet <- clean_text(train$OriginalTweet)
test$OriginalTweet  <- clean_text(test$OriginalTweet)

# ============================================================
# 10) Compare raw vs cleaned tweets for rows that changed
# ============================================================
changed_rows <- which(train_tweets_raw != train$OriginalTweet)

length(changed_rows)
length(changed_rows) / length(train_tweets_raw)

data.frame(
  row = head(changed_rows, 10),
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
#     \\S+ counts sequences of non-space characters
# ============================================================
tweet_lengths <- str_count(train$OriginalTweet, "\\S+")
test_lengths  <- str_count(test$OriginalTweet, "\\S+")


# ============================================================
# 13) Inspect rows that became empty after cleaning
#     This confirms whether they were just mentions/noise
# ============================================================
zero_rows <- which(tweet_lengths == 0)

zero_rows

data.frame(
  row = zero_rows,
  before = train_tweets_raw[zero_rows],
  after  = train$OriginalTweet[zero_rows]
)

# ============================================================
# 14) Remove tweets with 0 words after cleaning (these rows do not provide meaningful text)
# ============================================================
train <- train[tweet_lengths > 0, ]
test  <- test[test_lengths > 0, ]


# ============================================================
# 15) Recalculate tweet lengths AFTER removing empty rows
# ============================================================
tweet_lengths <- str_count(train$OriginalTweet, "\\S+")
test_lengths  <- str_count(test$OriginalTweet, "\\S+")


# ============================================================
# 16) Choose a reasonable maxlen value
# maxlen = number of tokens allowed per tweet
# summarize tweet lengths in order to pick a value
# ============================================================
summary(tweet_lengths)
quantile(tweet_lengths, probs = c(0.50, 0.75, 0.90, 0.95, 0.99))
# most tweets are <50 words

# ============================================================
# 17) Choose preprocessing parameters
# ideal maxlen is 50 as almost all tweets are preserved almost completely from our data
# numwords = 10,000 keeps frequent informative words while still filtering
# ============================================================
num_words <- 10000
maxlen <- 50

# ============================================================
# 18) Create vectorization layer
# ============================================================

vec <- layer_text_vectorization(
  max_tokens = num_words,
  standardize = "lower_and_strip_punctuation",
  split = "whitespace",
  output_mode = "int",
  output_sequence_length = maxlen
)

# ============================================================
# 19) Build vocabulary from train only
# ============================================================
adapt(vec, train$OriginalTweet)

# ============================================================
# 20) Inspect the learned vocabulary
# ============================================================
vocab <- vec$get_vocabulary()
vocab
cat("Found", length(vocab), "tokens in learned vocabulary.\n")
head(vocab, 50)

# ============================================================
# 21) Create word to index mapping 
# ============================================================
word_index <- stats::setNames(seq_along(vocab), vocab)
head(word_index, 30)

# ============================================================
# 22) Convert tweets into integer sequences
# ============================================================
seq_tensor_train <- vec(train$OriginalTweet)
seq_tensor_test  <- vec(test$OriginalTweet)

class(seq_tensor_train)

x_train <- as.array(seq_tensor_train)
x_test  <- as.array(seq_tensor_test)

# evaluate
dim(x_train)
dim(x_test)

x_train[1:2, 1:20]

# ============================================================
# 23) Encode sentiment labels
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
# 24) Check encoded label distributions
# ============================================================
table(y_train)
table(y_test)