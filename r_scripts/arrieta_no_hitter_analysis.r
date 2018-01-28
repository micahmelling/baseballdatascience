#Create simple web scraper to retrieve data from brooksbaseball.net
library(XML)
arrieta <-htmlParse("http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=453562&game=gid_2016_04_21_chnmlb_cinmlb_1/&s_type=3&h_size=700&v_size=50")
arrieta.tab<-readHTMLTable(arrieta, stringsAsFactors=FALSE)
arrieta.df<-arrieta.tab[[1]]
write.csv(arrieta.df, file = "arrieta.csv")

#Inspect the data
str(arrieta.df)
#unfortunately, everything has been read in as characters

#Instead, let's read in the CSV of the data, which should eliminate this issue
arrieta <- read.csv("arrieta.csv")
summary(arrieta)

#Let's start by looking at Arrietta's pitch speed during the night
plot(ecdf(arrieta$start_speed),
     main = "Cumulative Distribution of Pitch Speed",
     ylab = "Cumulative Proportion",
     xlab = "Pitch Speed",
     yaxt = "n")
axis (side=2, at=seq(0, 1, by=0.1), las=1, labels=paste(seq(0, 100, by=10),
                                                            "%", sep=" "))
abline(h=0.9, lty=3)
abline(v=quantile(arrieta$start_speed, pr=0.9), lty=3)

#Let's now take a look at horitzontal and vertical movement of his pitches
library(ggplot2)
p <- ggplot(arrieta, aes(x=pfx_x, y=pfx_z))
p + geom_point() + stat_density2d() + ggtitle("Density of Vertical and Hortizontal Pitch Movement")

#Let's visualize the difference in Arrietta's pitches based on pitch velocity, 
#movement,and spin 
ggplot(arrieta, aes(start_speed, fill = mlbam_pitch_name)) +
  geom_histogram(binwidth = 1) + facet_wrap(~ mlbam_pitch_name) +
  ggtitle("Pitch Speed Histogram by Pitch Type")

ggplot(arrieta, aes(x=pfx_x, y=pfx_z)) +
  geom_point(shape=19) + facet_wrap(~ mlbam_pitch_name) +  
  geom_smooth() + ggtitle("Vertical and Horizontal Movement by Pitch")

#Delete rows containing CH
without_ch <- arrieta[-c(69, 90), ]
ggplot(without_ch, aes(spin, fill = mlbam_pitch_name)) +
  geom_density() + facet_wrap(~ mlbam_pitch_name) +
  ggtitle("Spin by Pitch Type")

#Subset data by different play result to see if we can deduce any insights
ball <- subset(arrieta, pdes == "Ball")
called_strike <- subset(arrieta, pdes == "Called Strike")
foul <- subset(arrieta, pdes == "Foul")
in_play_outs <- subset(arrieta, pdes == "In play, out(s)")
swinging_strike <- subset(arrieta, pdes == "Swinging Strike")

summary(ball)
summary(called_strike)
summary(foul)
summary(in_play_outs)
summary(swinging_strike)

#Look at Arrieta's performance throughout the game
mean(arrieta$start_speed)
aggregate(start_speed ~ inning + mlbam_pitch_name, data = arrieta, mean)

p1 <- ggplot(arrieta, aes(x=inning, y=mlbam_pitch_name, fill=start_speed))
p1 + geom_tile() + scale_fill_gradient2(midpoint=92, low="blue", high="red") +
  ggtitle("Pitch Speed by Inning") + 
  scale_x_continuous(breaks = c(1,2,3,4,5,6,7,8,9))

library(dplyr)
early <- filter(arrieta, inning == 1 | inning == 2 | inning == 3 )
mid <- filter(arrieta, inning == 4 | inning == 5 | inning == 6 )
late <- filter(arrieta, inning == 7 | inning == 8 | inning == 9 )

summary(early)
prop.table(table(early$mlbam_pitch_name))
mean(early$start_speed)

summary(mid)
prop.table(table(mid$mlbam_pitch_name))
mean(mid$start_speed)

summary(late)
summary(mid)
prop.table(table(late$mlbam_pitch_name))
mean(late$start_speed)

#Can we predict which pitch Arrieta will throw next?

#Read in new dataset
prediction <- read.csv("arrieta_prediction.csv")
summary(prediction)
str(prediction)

prediction$strikes <- as.factor(prediction$strikes)
prediction$balls <- as.factor(prediction$balls)

#Create training and test sets
library(caret)

inTrain <- createDataPartition(y=prediction$pitch_type,p=0.75, list=FALSE)
training <- prediction[inTrain,]
testing <- prediction[-inTrain,]

#Decision Tree and Random Forest
library(rpart)
library(rattle)
tree <- rpart(pitch_type ~ ., method="class", data=training)
printcp(tree)
print(tree)
fancyRpartPlot(tree)

library(randomForest)
fit_rf <- randomForest(pitch_type ~ ., data=training)
print(fit_rf)

head(fit_rf$votes)
importance(fit_rf)
barplot(fit_rf$importance[ , 1 ], main="Importance of Variables in Random Forest", 
        cex.names =0.5) 

tree_predictions <- predict(tree, testing, type="class")
table <- data.frame(tree_predictions, testing$pitch_type)
table
confusionMatrix(tree_predictions, testing$pitch_type)

rf_predictions <- predict(fit_rf, testing, type="class")
table2 <- data.frame(rf_predictions, testing$pitch_type)
table2
confusionMatrix(rf_predictions, testing$pitch_type)

rf_predictions2 <- predict(fit_rf, testing, type="prob")
table3 <- data.frame(rf_predictions2, testing$pitch_type)
table3
