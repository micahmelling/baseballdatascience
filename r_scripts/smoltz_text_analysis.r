# HoF text mining
# Load libraries
library(quanteda)
library(tm)
library(data.table)
library(topicmodels)
library(ggplot2)

#############Text Analysis#############
# Read in data
smoltz <- read.csv('smoltz_speech.csv')

#Convert the tweet text into a corpus, the object R needs for text analysis
masterCorpus <- Corpus(VectorSource(smoltz$text))
masterCorpus

#Remove punctuation and numbers from the text; convert all text to lower case
masterCorpus <- tm_map(masterCorpus, removePunctuation)
masterCorpus <- tm_map(masterCorpus, removeNumbers)
masterCorpus <- tm_map(masterCorpus, tolower)

#Remove common stop words as well as bespoke list of words
masterCorpus <- tm_map(masterCorpus, removeWords, stopwords("english"))
masterCorpus <- tm_map(masterCorpus, removeWords, c("how", "why", "be", "here", "there", "via", 
                                                    "amp", "there", "will", "can", "see", "new", 
                                                    "sap", "help", "find", "get", "make", "watch", 
                                                    "take", "learn", "need", "one", "now", "just",
                                                    "like", "cant", "got", "much", "say", "way",
                                                    "going", "dont", "said", "ever", "doesnt",
                                                    "come", "since","wont","saying","didnt",
                                                    "every", "hes","youre", "still", "ive", 
                                                    "use", "even", "u", "ut"))

#Stem the document to have R read items like "analyze" and "analyzed" as one term
#Stripe white space created by removed words and treat the corpus as plain text
masterCorpus <- tm_map(masterCorpus, stemDocument)
masterCorpus <- tm_map(masterCorpus, stripWhitespace)
masterCorpus <- tm_map(masterCorpus, PlainTextDocument)

#create document term matrix, which tells us which words were used in each document (i.e. a tweet)
dtm <- DocumentTermMatrix(masterCorpus)
dtm

#Sum the number of times each word was used
freq <- colSums(as.matrix(dtm))

#Order the number of times by how often they were used
ord <- order(freq)
freq[tail(ord)]

#Convert the list of words to a dataframe, which will be easier to work with
wf <- data.frame(word=names(freq), freq=freq) 
write.csv(wf, file = "smoltz_unigrams.csv")

# Create chart of most used words
p <- ggplot(subset(wf, freq > 10), aes(word, freq, fill = "blue"))    
p <- p + geom_bar(stat="identity") + ggtitle('Smoltz Most Used Words')   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))+
  theme(plot.title = element_text(hjust = 0.5)) + theme(legend.position = "none")   
p   

#Create a Quanteda corpus from the TM corpus
smoltz$text <- as.character(smoltz$text)
corpus <- corpus(smoltz$text)

#Create a document term matrix of bi-grams, groups of two words, and sort the results by popularity
dfm.bi <- dfm(corpus, ignoredFeatures = stopwords("english"), stem = TRUE, ngrams = 2, verbose = FALSE)
dfm.bi.freq <- colSums(dfm.bi)
dfm.bi.freq <- sort(dfm.bi.freq, decreasing=TRUE) 

#Take out bi-grams used fewer than ten times
dfm.bi.freq.prune <- as.numeric()
for (i in 1:length(dfm.bi.freq)) { 
  if (dfm.bi.freq[i] > 2) {
    dfm.bi.freq.prune  <- c(dfm.bi.freq.prune, dfm.bi.freq[i]) }
}

#Convert results to a data frame
bigrams <- data.frame(dfm.bi.freq.prune)

#Change rownames to a column and rename columns
setDT(bigrams, keep.rownames = TRUE)[]
names(bigrams)[1] <- "word"
names(bigrams)[2] <- "freq"

write.csv(bigrams, file = "smoltz_bigrams.csv")

p <- ggplot(subset(bigrams, freq > 3), aes(word, freq, fill = "blue"))    
p <- p + geom_bar(stat="identity") + ggtitle('Smoltz Most Used Bigramss')   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))+
  theme(plot.title = element_text(hjust = 0.5)) + theme(legend.position = "none")   
p 

#Remove sparse terms from the document term matrix
dtms <- removeSparseTerms(dtm, 0.98)  
dtms

#Cluster words that often appear together
d <- dist(t(dtms), method="euclidian")   
kfit <- kmeans(d, 10)   
clusplot(as.matrix(d), kfit$cluster, color=T, shade=T, labels=2, lines=0) 
clusters <- data.frame(kfit$cluster)
print(clusters)

#Create a document term matrix for topic modeling
dfm.uni <- dfm(corpus, ignoredFeatures = stopwords("english"), stem = TRUE, verbose = FALSE)

#Run LDA topic model
burnin <- 4000
iter <- 2000
thin <- 500
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE
k <- 5

ldaOut <-LDA(dtm,k, method='Gibbs', control=list(nstart=nstart, 
                                                 seed = seed, 
                                                 best=best, 
                                                 burnin = burnin, 
                                                 iter = iter, thin=thin))

ldaOut.topics <- as.matrix(terms(ldaOut, 6))

