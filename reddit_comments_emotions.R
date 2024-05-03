install.packages("tidytext")
install.packages("tidyverse")
install.packages("textdata")
install.packages("textstem")
install.packages("GGally") #Scatter plot metrics
install.packages("tidylo") #term frequency, to find most characteristic words
install.packages("wordcloud")
install.packages("dplyr")
install.packages("ggplot2")
install.packages('topicmodels')
install.packages('naivebayes')
install.packages('stopwords')

require(tidytext)
require(tidyverse)
require(textdata)
require(textstem)
require(GGally)
require(tidylo)
require(wordcloud)
require(dplyr)
require(ggplot2)
require(topicmodels)
require(naivebayes)
require(stopwords)

# Read data
goemotions_1 <- read_csv("goemotions_1.csv")
reddit_comments <- goemotions_1

emotions <- readLines("emotions.txt")

# Data inspection
glimpse(reddit_comments)  # Check structure of the data
head(reddit_comments)     # View first few rows

# Summary statistics for numerical features
summary(reddit_comments)

#-------------------------------------------------------------------------------
# Filter out neutral emotion
emotion_to_filter <- c("neutral")
filtered_emotions <- emotions[!emotions %in% emotion_to_filter]

# Subset reddit_comments to include only common columns
mapped_columns_data <- reddit_comments[, filtered_emotions]

# Sum the occurrences of emotions in each common column
occurrences <- colSums(mapped_columns_data == 1)

# Sort the emotions by occurrences in descending order
sorted_emotions <- sort(occurrences, decreasing = TRUE)

# Create a data frame for plotting
plot_data <- data.frame(Emotion = names(sorted_emotions),
                        Occurrence = sorted_emotions)

# Create the horizontal bar plot using ggplot2
ggplot(plot_data, aes(x = Occurrence, y = fct_reorder(Emotion, -Occurrence))) +
  geom_bar(stat = "identity", fill = "#56B4E9") +
  labs(title = "Occurrence of Emotions",
       x = "Number of occurences",
       y = "Emotions") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 35, hjust = 1)) +  # Rotate x-axis labels vertically
  coord_flip()  # Flip coordinates to create horizontal plot

# Check if emotions belong to multiple comments
# Sum the values in each row of emotion columns
reddit_comments$multiple_emotions <- rowSums(reddit_comments[, filtered_emotions])

# Check if any comment belongs to multiple emotions
comments_with_multiple_emotions <- reddit_comments[reddit_comments$multiple_emotions > 1, ]

# View comments with multiple emotions
print(comments_with_multiple_emotions)

#-------------------------------------------------------------------------------
# Word frequencies for textual features
word_freq <- reddit_comments %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  count(word, sort = TRUE)

# Word cloud
wordcloud(words = word_freq$word, freq = word_freq$n, max.words = 100, colors = brewer.pal(8, "Dark2"))

#-------------------------------------------------------------------------------
# The most characteristic words - bigrams
reddit_comments_unnested <- reddit_comments %>%
  select(text) %>%
  mutate(text = tolower(text)) %>%
  unnest_tokens(output = bigram, input = text, token = 'ngrams', n = 2) %>%
  group_by(bigram) %>% 
  count() %>%
  ungroup() %>%
  mutate(total_count = sum(n)) %>%
  arrange(desc(n)) %>%
  mutate(log_odds_weighted = log(n / (total_count - n)))

# Select top 10 bigrams by log odds ratio
top_bigrams <- reddit_comments_unnested %>%
  slice_max(log_odds_weighted, n = 10)

top_bigrams

# Plot the top bigrams
ggplot(top_bigrams, aes(x = log_odds_weighted, y = reorder(bigram, log_odds_weighted), fill = 'red')) +
  geom_bar(stat = 'identity') +
  labs(x = "Log Odds Ratio", y = "Bigram", title = "Top Bigrams by Log Odds Ratio") +
  theme_minimal() +
  theme(legend.position = 'none') +
  coord_flip()  # Flip coordinates to create horizontal plot

#-------------------------------------------------------------------------------
# Convert from epoch to human-readable date and place after "created_utc" column
reddit_comments <- reddit_comments %>%
  mutate(created_readable = as.POSIXct(created_utc, origin = "1970-01-01", tz = "GMT"), .after = "created_utc")

# Remove the time from date
reddit_comments <- reddit_comments %>%
  mutate(created_date = as.Date(created_readable))

# plot timeseries chart
reddit_comments %>%
  group_by(created_date) %>%
  count() %>%
  ggplot(aes(x = created_date, y = n)) +
  geom_line(color = "#D55E00") +
  labs(title = "Comment Activity Over Time",
       x = 'Created date') +
  theme_minimal() +
  theme(legend.position = 'top')

# Extract hour and store it in a new column called "created_hour"
reddit_comments <- reddit_comments %>% mutate(created_hour = format(created_readable, format = "%H"))

# Count comments per hour
hourly_counts <- reddit_comments %>%
  group_by(created_hour) %>%
  summarise(comment_count = n())

# The hourly distribution of comments bar chart
ggplot(hourly_counts, aes(x = created_hour, y = comment_count)) +
  geom_bar(stat = "identity", fill = "#D55E00") +
  labs(x = "Hour of the Day", y = "Number of Comments", title = "Hourly Distribution of Comments") + 
  theme_minimal()

#-------------------------------------------------------------------------------
# Sentiment analysis
reddit_comments_sentiments <- reddit_comments %>%
  unnest_tokens(word, text) %>%
  filter(!word %in% stop_words$word) %>%
  inner_join(get_sentiments("nrc"), by = "word") %>%
  count(sentiment, name = "n")

ggplot(reddit_comments_sentiments, aes(x = reorder(sentiment, -n), y = n)) +
  geom_bar(stat = "identity", fill = "#56B4E9") +
  labs(title = "Sentiment Distribution",
       x = "Sentiment",
       y = "Frequency") +
  theme_minimal()

#-------------------------------------------------------------------------------
# Topic Modelling
# Unnest & count the words in the documents
word_count<-reddit_comments_class%>%select(id,text)%>%
  unnest_tokens(input = text, 
                output = word)%>%
  filter(!word%in%stop_words$word)%>%
  group_by(id,word)%>%count()

# Cast into dtm-object
reddit_comments_dtm_object<-word_count%>%cast_dtm(id,word,n)

reddit_comments_dtm_object

# Estimate the LDA topic model
reddit_comments_lda<-LDA(reddit_comments_dtm_object,
                         k=4,
                         control=list(seed=1234))

reddit_comments_topic_words<-tidy(reddit_comments_lda,matrix='beta') # most probable words for each topics

reddit_comments_topic_words%>%
  group_by(topic)%>%
  slice_max(beta,n=10)%>%ungroup()%>%
  ggplot(aes(x=beta,y=term,fill=as.factor(topic)))+
  geom_col(show.legend = FALSE)+
  facet_wrap(vars(topic),scales = 'free')

document_topic_table<-tidy(reddit_comments_lda,matrix='gamma')

document_topic_table%>%filter(topic==4)

# Extract topic-word probabilities
reddit_comments_topic_words <- tidy(reddit_comments_lda, matrix = "beta")

# Filter the top words for each topic
top_words_per_topic <- reddit_comments_topic_words %>%
  group_by(topic) %>%
  top_n(n = 10, wt = beta) %>%
  arrange(topic, -beta)

# Plot the top words for each topic
ggplot(top_words_per_topic, aes(x = reorder(term, beta), y = beta, fill = factor(topic))) +
  geom_bar(stat = "identity") +
  labs(x = "Word", y = "Probability", title = "Top Words for Each Topic") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  facet_wrap(~ topic, scales = "free") +
  scale_fill_brewer(palette = "Set1")
