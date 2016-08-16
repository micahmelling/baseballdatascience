##Read in CSV and inspect data
hosmer <- read.csv("hosmer_stats_updated.csv")
summary(hosmer)

##Bar charts of selected variables
library(ggplot2)
qplot(factor(pitch_type), data=hosmer, geom="bar", fill=factor(pitch_type)) +
  ggtitle("Types of Pitches Hit in Play")

qplot(factor(events), data=hosmer, geom="bar", fill=factor(events)) +
  ggtitle("Result of Balls in Play") + coord_flip()

##Pitch type analysis
#Let's run a chi-squared test to see if hit location depends on pitch type
table1 <- table(hosmer$pitch_type, hosmer$hit_location)
table1
chisq.test(table1)

##Let's look at hit distance, speed, and angle
#First, let's inspect means and standard deviations by pitch type for hit distance, speed, and angle
tapply(hosmer$hit_distance_sc, hosmer$pitch_type, mean)
tapply(hosmer$hit_distance_sc, hosmer$pitch_type, sd)
tapply(hosmer$hit_speed, hosmer$pitch_type, mean)
tapply(hosmer$hit_speed, hosmer$pitch_type, sd)
tapply(hosmer$hit_angle, hosmer$pitch_type, mean)
tapply(hosmer$hit_angle, hosmer$pitch_type, sd)

#Plot lowess lines to explore the relationship between variables
ggplot(hosmer, aes(x=hit_speed, y=hit_angle)) +
  geom_point() +    
  geom_smooth() + ggtitle("Relationship Between Hit Angle and Hit Speed")

ggplot(hosmer, aes(x=hit_angle, y=hit_distance_sc)) +
  geom_point(shape=19) +    
  geom_smooth() + ggtitle("Relationship Between Hit Distance and Hit Angle")

ggplot(hosmer, aes(x=hit_speed, y=hit_distance_sc)) +
  geom_point(shape=19) +    
  geom_smooth() + ggtitle("Relationship Between Hit Distance and Hit Speed")

#Let's look at different styles of density plots for the result of balls put in play
ggplot(hosmer, aes(hit_speed, colour = description)) +
  geom_density() + ggtitle("Density by Result of Hit")

ggplot(hosmer, aes(hit_speed, fill = description)) +
  geom_density(position="stack") + ggtitle("Density by Result of Hit")

ggplot(hosmer, aes(hit_speed, fill = description)) +
  geom_density(position="fill") + ggtitle("Density by Result of Hit")

#Viloin plot of hit speed by type of pitch
g<-ggplot(hosmer, aes(x=pitch_type, y=hit_speed))
g + geom_violin(alpha=0.5, color="gray")+geom_jitter(alpha=0.5, aes(color=pitch_type),
                                                     position = position_jitter(width = 0.1))+ 
  coord_flip() + ggtitle("Hit Speed by Pitch Type")

##Segment pitch break length into quartiles and see how Hosmer handles pitches with greater break
library(data.table)
setDT(hosmer)
hosmer[,quartile:=cut(break_length,
                      breaks=quantile(break_length,probs=seq(0,1,by=1/4)),
                      labels=1:4,right=F)]

#Hit speed histograms faceted by break length
ggplot(hosmer, aes(hit_speed, fill = quartile)) +
  geom_histogram(binwidth = 10) + facet_wrap(~ quartile) +
  ggtitle("Hit Speed by Quartile of Pitch Break Length")

#Let's look at some specific scenarios
aggregate(hit_distance_sc ~ pitch_type + inning, data = hosmer, mean)
aggregate(hit_distance_sc ~ pitch_type + inning, data = hosmer, length)

hosmer$outs_when_up <- as.factor(as.numeric(hosmer$outs_when_up))
aggregate(hit_distance_sc ~ pitch_type + inning + outs_when_up, data = hosmer, mean)
aggregate(hit_distance_sc ~ pitch_type + inning + outs_when_up, data = hosmer, length)

#Let's look at the impact of the count
hosmer$balls <- as.factor(as.integer(hosmer$balls))
hosmer$strikes <- as.factor(as.integer(hosmer$strikes))

ggplot(hosmer, aes(hit_speed, fill = balls)) +
  geom_density(position="fill") + ggtitle("Hit Speed by Number of Balls")

ggplot(hosmer, aes(hit_speed, fill = strikes)) +
  geom_density(position="fill") + ggtitle("Hit Speed by Number of Strikes")

ggplot(hosmer, aes(hit_angle, fill = balls)) +
  geom_density(position="fill") + ggtitle("Hit Angle by Number of Balls")

ggplot(hosmer, aes(hit_angle, fill = strikes)) +
  geom_density(position="fill") + ggtitle("Hit Angle by Number of Strikes")

ggplot(hosmer, aes(hit_distance_sc, fill = balls)) +
  geom_density(position="fill") + ggtitle("Hit Distance by Number of Balls")

ggplot(hosmer, aes(hit_distance_sc, fill = strikes)) +
  geom_density(position="fill") + ggtitle("Hit Distance by Number of Strikes")

aggregate(hit_distance_sc ~ strikes + inning + outs_when_up, data = hosmer, mean)
aggregate(hit_distance_sc ~ strikes + inning + outs_when_up, data = hosmer, length)

#Lastly, let's do a k-means cluster of hit distance, speed, and angle
#Start by selecting the desired columns and scaling the data
library(dplyr)
hosmer1 <- subset(hosmer, select = c(54, 55, 56))

hosmer2 <- scale(hosmer1)
head(hosmer2)

#Elbow plot to determine the number of clusters
wss <- (nrow(hosmer2)-1)*sum(apply(hosmer2,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(hosmer2,
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares", main = "Elbow Plot for No. of Clusters")

#K-Means cluster with k=3
set.seed(600)
fit1 <- kmeans(hosmer2, 3) 
hosmer3 <- data.frame(hosmer1, fit1$cluster) 
hosmer <- data.frame(hosmer, fit1$cluster) 
head(hosmer3, 5)

#Look at appended clusters in full and stripped down datasets
cluster1 <- hosmer3[which(hosmer3$fit1.cluster=='1'),]
cluster2 <- hosmer3[which(hosmer3$fit1.cluster=='2'),]   
cluster3 <- hosmer3[which(hosmer3$fit1.cluster=='3'),]
summary(cluster1)
summary(cluster2)
summary(cluster3)

cluster1a <- hosmer[which(hosmer$fit1.cluster=='1'),]
cluster2a <- hosmer[which(hosmer$fit1.cluster=='2'),]   
cluster3a <- hosmer[which(hosmer$fit1.cluster=='3'),]
summary(cluster1a)
summary(cluster2a)
summary(cluster3a)

#PCA plot of clusters
library(cluster)
set.seed(500)
clusplot(hosmer2, fit1$cluster, color=TRUE, shade=TRUE,
         labels=2, lines=0, main = "PCA Plot of K-Means Cluster")
