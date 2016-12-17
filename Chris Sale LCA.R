#Can Latent Class Analysis identify Chris Sale's pitches?

#Libraries
library(XML)
library(ggplot2)
library(poLCA)
library(dplyr)
library(MASS)
library(colorRamps)

#Create simple web scraper to retrieve data from brooksbaseball.net
#Let's scrape the last five games of the 2016 regular season

#Game 1
game1 <-htmlParse("http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=519242&game=gid_2016_10_02_minmlb_chamlb_1/&s_type=3&h_size=700&v_size=500")
game1.tab <-readHTMLTable(game1, stringsAsFactors=FALSE)
game1.df <- game1.tab[[1]]

#Game 2
game2 <-htmlParse("http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=519242&game=gid_2016_09_27_tbamlb_chamlb_1/&s_type=3&h_size=700&v_size=500")
game2.tab <-readHTMLTable(game2, stringsAsFactors=FALSE)
game2.df <- game2.tab[[1]]

#Game 3
game3 <-htmlParse("http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=519242&game=gid_2016_09_21_chamlb_phimlb_1/&s_type=3&h_size=700&v_size=500")
game3.tab <-readHTMLTable(game3, stringsAsFactors=FALSE)
game3.df <- game3.tab[[1]]

#Game 4
game4 <-htmlParse("http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=519242&game=gid_2016_09_16_chamlb_kcamlb_1/&s_type=3&h_size=700&v_size=500")
game4.tab <-readHTMLTable(game4, stringsAsFactors=FALSE)
game4.df <- game4.tab[[1]]

#Game 5
game5 <-htmlParse("http://www.brooksbaseball.net/pfxVB/tabdel_expanded.php?pitchSel=519242&game=gid_2016_09_11_kcamlb_chamlb_1/&s_type=3&h_size=700&v_size=500")
game5.tab <-readHTMLTable(game5, stringsAsFactors=FALSE)
game5.df <- game5.tab[[1]]

#Combine all the dataframes
sale <- rbind(game1.df, game2.df, game3.df, game4.df, game5.df)

#All the columns were read in as characters, which isn't what we want
#Let's write the file to a CSV and read it back in, which should fix the issue
write.csv(sale, "sale.csv")
sale <- read.csv("sale.csv")

#Alright, it looks like everything read in correctly

#Let's create a pie chart of Sale's different pitches
ggplot(sale, aes(x = factor(1), fill = factor(mlbam_pitch_name))) +
  geom_bar(width = 1) + coord_polar(theta = "y") + 
  ggtitle("Pitches Thrown in Final 5 Games of 2016") + ylab(" ") +
  xlab(" ") + scale_y_continuous(breaks = sale$mlbam_pitch_names, 
                                 labels=sale$mlbam_pitch_name)

#Let's run multiple latent class models to see how they perform
#We'll use the following variables in the model:
#spin, pfx_x, pfx_z, vx0, vy0, vz0, ax, ay, az, start_speed
#https://fastballs.wordpress.com/2007/08/02/glossary-of-the-gameday-pitch-fields/

#First, though, we need to convert the numeric data into 
#categorical data; let's split the variables into quartiles
sale.sub <- sale[c("spin", "pfx_x", "pfx_z", "vx0", "vy0", "vz0", 
                   "ax", "ay", "az", "start_speed")]

quartile <- function(x) {
  ntile(x, 4)
}

sale.sub <- apply(sale.sub, 2, quartile)
sale.sub <- data.frame(sale.sub)
sale.sub <- data.frame(sapply(sale.sub, as.factor))
summary(sale.sub)

#Sale throws three pitches, so we're most interested in the lc3 model
f <- cbind(spin, pfx_x, pfx_z, vx0, vy0, vz0, ax, ay, az, start_speed)~1
set.seed(200)
lc2 <- poLCA(f, sale.sub, nclass=2, graph = TRUE)
lc3 <- poLCA(f, sale.sub, nclass=3, graph = TRUE)
lc4 <- poLCA(f, sale.sub, nclass=4, graph = TRUE)

#Since we know Sale throws three pitches, it doesn't make sense to
#run more models, though the lc4 had a better AIC than the lc3
#Let's dive deeper in the lc3 results

#Look at predictions and probabilities for each observation
probs <- lc3$posterior
head(probs)

preds <- lc3$predclass
head(preds)

#Create a dataframe of predictions and probabilities assigned to each observation
prediction_frame <- data.frame(preds, probs)

#Create a data frame of the the original numeric data and the declared pitch type
sale.original <- sale[c("spin", "pfx_x", "pfx_z", "vx0", "vy0", "vz0", 
                   "ax", "ay", "az", "start_speed", "mlbam_pitch_name")]

#Bind the data frames
sale.final <- data.frame(sale.original, prediction_frame)

#Clean the names of the columns from the prediction frame
names(sale.final)[12] <- "Predicted_Class"
names(sale.final)[13] <- "Class1_Prob"
names(sale.final)[14] <- "Class2_Prob"
names(sale.final)[15] <- "Class3_Prob"

#Change the predicted class to a factor
sale.final$Predicted_Class <- as.factor(sale.final$Predicted_Class)

#Get summaries of the data
sale_lca1 <- subset(sale.final, Predicted_Class=="1")
summary(sale_lca1)

sale_lca2 <- subset(sale.final, Predicted_Class=="2")
summary(sale_lca2)

sale_lca3 <- subset(sale.final, Predicted_Class=="3")
summary(sale_lca3)

#Parallel coordinates plot to view the classes
r <- (sale.final$Predicted_Class)
parcoord(sale.final[1:10], col=r)
