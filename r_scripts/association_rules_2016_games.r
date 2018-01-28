##Load libraries##
library(XML)
library(ggplot2)
library(arules)
library(arulesViz)
library(plyr)
library(dplyr)
library(stringr)
library(matrixStats)
library(data.table)
library(Hmisc)
library(gridExtra)
library(knitr)
library(rgl)

##Read in and inspect data##

#Scrape a dataset that includes each team's abbreviation
#We'll isolate the names and put them in a list, which will be used to scrape
#the data we really want
abbreviations <- htmlParse("http://www.baseball-reference.com/leagues/MLB/2016.shtml")
abbreviations.tab <- readHTMLTable(abbreviations, stringsAsFactors=FALSE)
abbreviations.df <- abbreviations.tab[[2]]

#Create list of teams
teams <- list(abbreviations.df$Tm)
teams <- sapply(teams, "[", c(1:30))

#Let's now scrape per-game offensive statistics for each team
fetch_offense <- function(team) {
  url <- paste0("http://www.baseball-reference.com/teams/tgl.cgi?team=", team, "&t=b&year=2016.com")
  data <- readHTMLTable(url, stringsAsFactors = FALSE)
  data <- data[[1]]
  data$team <- team
  data
}

offensive_stats <- ldply("teams", fetch_offense, .progress="text")

#Hmmmm, I'm getting a subscript out of bounds error
#Let's try a loop and see if that works
results <- data.frame()
for (i in "teams") {
  results <- c(results, fetch_offense(i))
  }

#Still getting the same error; looks like we may have to manually 
#insert each team
ARI_offense <- ldply("ARI", fetch_offense, .progress="text")
ATL_offense <- ldply("ATL", fetch_offense, .progress="text")
BAL_offense <- ldply("BAL", fetch_offense, .progress="text")
BOS_offense <- ldply("BOS", fetch_offense, .progress="text")
CHC_offense <- ldply("CHC", fetch_offense, .progress="text")
CHW_offense <- ldply("CHW", fetch_offense, .progress="text")
CIN_offense <- ldply("CIN", fetch_offense, .progress="text")
CLE_offense <- ldply("CLE", fetch_offense, .progress="text")
COL_offense <- ldply("COL", fetch_offense, .progress="text")
DET_offense <- ldply("DET", fetch_offense, .progress="text")
HOU_offense <- ldply("HOU", fetch_offense, .progress="text")
KCR_offense <- ldply("KCR", fetch_offense, .progress="text")
LAA_offense <- ldply("LAA", fetch_offense, .progress="text")
LAD_offense <- ldply("LAD", fetch_offense, .progress="text")
MIA_offense <- ldply("MIA", fetch_offense, .progress="text")
MIL_offense <- ldply("MIL", fetch_offense, .progress="text")
MIN_offense <- ldply("MIN", fetch_offense, .progress="text")
NYM_offense <- ldply("NYM", fetch_offense, .progress="text")
NYY_offense <- ldply("NYY", fetch_offense, .progress="text")
OAK_offense <- ldply("OAK", fetch_offense, .progress="text")
PHI_offense <- ldply("PHI", fetch_offense, .progress="text")
PIT_offense <- ldply("PIT", fetch_offense, .progress="text")
SDP_offense <- ldply("SDP", fetch_offense, .progress="text")
SEA_offense <- ldply("SEA", fetch_offense, .progress="text")
SFG_offense <- ldply("SFG", fetch_offense, .progress="text")
STL_offense <- ldply("STL", fetch_offense, .progress="text")
TBR_offense <- ldply("TBR", fetch_offense, .progress="text")
TEX_offense <- ldply("TEX", fetch_offense, .progress="text")
TOR_offense <- ldply("TOR", fetch_offense, .progress="text")
WSN_offense <- ldply("WSN", fetch_offense, .progress="text")

#Bind the data frames
offensive_complete <- rbind(ARI_offense, ATL_offense, BAL_offense, BOS_offense, CHC_offense,
CHW_offense, CIN_offense, CLE_offense, COL_offense, DET_offense, HOU_offense, KCR_offense,
LAA_offense, LAD_offense, MIA_offense, MIL_offense, MIN_offense, NYM_offense, NYY_offense,
OAK_offense, PHI_offense, PIT_offense, SDP_offense, SEA_offense, SFG_offense, STL_offense,
TBR_offense, TEX_offense, TOR_offense, WSN_offense)

#OK, now that we have offensive stats, let's scrape pitching stats
fetch_pitching <- function(team) {
  url <- paste0("http://www.baseball-reference.com/teams/tgl.cgi?team=", team, "&t=p&year=2016.com")
  data <- readHTMLTable(url, stringsAsFactors = FALSE)
  data <- data[[1]]
  data$team <- team
  data
}

pitching_stats <- ldply("teams", fetch_pitching, .progress="text")

#Still getting a subscript out of bounds error, so we'll have to 
#manually insert each team
ARI_pitching <- ldply("ARI", fetch_pitching, .progress="text")
ATL_pitching <- ldply("ATL", fetch_pitching, .progress="text")
BAL_pitching <- ldply("BAL", fetch_pitching, .progress="text")
BOS_pitching <- ldply("BOS", fetch_pitching, .progress="text")
CHC_pitching <- ldply("CHC", fetch_pitching, .progress="text")
CHW_pitching <- ldply("CHW", fetch_pitching, .progress="text")
CIN_pitching <- ldply("CIN", fetch_pitching, .progress="text")
CLE_pitching <- ldply("CLE", fetch_pitching, .progress="text")
COL_pitching <- ldply("COL", fetch_pitching, .progress="text")
DET_pitching <- ldply("DET", fetch_pitching, .progress="text")
HOU_pitching <- ldply("HOU", fetch_pitching, .progress="text")
KCR_pitching <- ldply("KCR", fetch_pitching, .progress="text")
LAA_pitching <- ldply("LAA", fetch_pitching, .progress="text")
LAD_pitching <- ldply("LAD", fetch_pitching, .progress="text")
MIA_pitching <- ldply("MIA", fetch_pitching, .progress="text")
MIL_pitching <- ldply("MIL", fetch_pitching, .progress="text")
MIN_pitching <- ldply("MIN", fetch_pitching, .progress="text")
NYM_pitching <- ldply("NYM", fetch_pitching, .progress="text")
NYY_pitching <- ldply("NYY", fetch_pitching, .progress="text")
OAK_pitching <- ldply("OAK", fetch_pitching, .progress="text")
PHI_pitching <- ldply("PHI", fetch_pitching, .progress="text")
PIT_pitching <- ldply("PIT", fetch_pitching, .progress="text")
SDP_pitching <- ldply("SDP", fetch_pitching, .progress="text")
SEA_pitching <- ldply("SEA", fetch_pitching, .progress="text")
SFG_pitching <- ldply("SFG", fetch_pitching, .progress="text")
STL_pitching <- ldply("STL", fetch_pitching, .progress="text")
TBR_pitching <- ldply("TBR", fetch_pitching, .progress="text")
TEX_pitching <- ldply("TEX", fetch_pitching, .progress="text")
TOR_pitching <- ldply("TOR", fetch_pitching, .progress="text")
WSN_pitching <- ldply("WSN", fetch_pitching, .progress="text")

#Bind the data frames
pitching_complete <- rbind(ARI_pitching, ATL_pitching, BAL_pitching, BOS_pitching, CHC_pitching,
CHW_pitching, CIN_pitching, CLE_pitching, COL_pitching, DET_pitching, HOU_pitching, KCR_pitching,
LAA_pitching, LAD_pitching, MIA_pitching, MIL_pitching, MIN_pitching, NYM_pitching, NYY_pitching,
OAK_pitching, PHI_pitching, PIT_pitching, SDP_pitching, SEA_pitching, SFG_pitching, STL_pitching,
TBR_pitching, TEX_pitching, TOR_pitching, WSN_pitching)

#The pitching data was read in an odd wa, so let's fix that
fix_date <- function(x) {
  x <- str_replace_all(x, "Â", "") 
}

pitching_complete$Date <- fix_date(pitching_complete$Date)

#Merge data frames on team and Gtm, which is essentially a unique ID 
#for each game
game_logs <- merge(offensive_complete, pitching_complete, by=c("Gtm", "team"))

#Drop the first 750 rows, which are essentially the headings
game_logs <- game_logs[-c(1:750), ] 

#All of the columns were read in as characters
#Let's convert each column to the correct data type
#We'll drop a few columns in the process
cat_columns <- subset(game_logs, select = c(1:6, 32:33, 65))
game_logs1 <- data.frame(sapply(cat_columns, as.factor))
summary(game_logs1)

num_columns <- subset(game_logs, select = c(8:31, 39:64))
game_logs2 <- data.frame(sapply(num_columns, as.numeric))

game_logs3 <- subset(game_logs, select = c(7, 66))

games_final <- data.frame(game_logs1, game_logs2, game_logs3)
summary(games_final)

write.csv(games_final, file = "2016_game_logs.csv")

##Create exploratory visualizations##
#Look at distributions of key variables

#Runs scored
p1 <- ggplot(games_final, aes(R.x)) +
  geom_density() + ggtitle("Distribution of Runs Scored") + xlab("Runs Scored") +
  ylab(" ")

#Stolen Bases
p2 <- ggplot(games_final, aes(SB.x)) +
  geom_density() + ggtitle("Distribution of Stolen Bases") + xlab("Stolen Bases") +
  ylab(" ")

#Earned Runs
p3 <- ggplot(games_final, aes(ER)) +
  geom_density() + ggtitle("Distribution of Earned Runs") + xlab("Earned Runs") +
  ylab(" ")

#Pitches
p4 <- ggplot(games_final, aes(Pit)) +
  geom_density() + ggtitle("Distribution of Pitches") + xlab("Pitches") +
  ylab(" ")

grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2)

#Make 3D scatter plots of runs, hits, and home runs
interleave <- function(v1, v2) as.vector(rbind(v1,v2))

plot3d(games_final$R.x, games_final$HR.x, games_final$H.x, xlab = "Runs",
ylab = "Home Runs", zlab = "Hits", type = "s", size = 0.75, lit = FALSE)

segments3d(interleave(games_final$R.x, games_final$R.x),
           interleave(games_final$HR.x, games_final$HR.x),
           interleave(games_final$H.x, min(games_final$H.x)),
           alpha = 0.4, col = "blue")

#Create color-coded scatter plot of pitches thrown and walks issued
p5 <- ggplot(games_final, aes(x=Pit, y=BB.y, colour=R.y)) +
  geom_point(size=3) + scale_color_gradientn(colours = c("darkred",
  "orange", "yellow")) + xlab("Pitches Thrown") + ylab("Walks Issued") +
  ggtitle("Relationship between Walks and Pitches \n Colored by Runs Allowed")

p5

#Let's dive into some stats by umpire
umps <- table(games_final$Umpire)
umps <- as.data.frame(umps)
mean(umps$Freq)
sd(umps$Freq)

#Inspect hits by umpire
mean_hits <- aggregate(H.x ~ Umpire, data = games_final, mean)
sd_hits <- aggregate(H.x ~ Umpire, data = games_final, sd)
ump_hits <- merge(mean_hits, sd_hits, by = "Umpire")
names(ump_hits)[2]<-"Mean"
names(ump_hits)[3]<-"Standard Deviation"
ump_hits$Umpire <- sub("^$", "Unknown", ump_hits$Umpire)
ump_hits1 <- head(ump_hits[order(-ump_hits$Mean),], 15)

data.m <- melt(ump_hits1, id.vars='Umpire')
ggplot(data.m, aes(Umpire, value)) +   
  geom_bar(aes(fill = variable), position = "dodge", stat="identity") + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + xlab("") +
  ggtitle("Mean and Standard Deviations for Hits \n in Games Segmented by Home Plate Umpire")
  
#Inspect runs by umpire
mean_runs <- aggregate(R.x ~ Umpire, data = games_final, mean)
sd_runs <- aggregate(R.x ~ Umpire, data = games_final, sd)
ump_runs <- merge(mean_runs, sd_runs, by = "Umpire")
names(ump_runs)[2]<-"Mean"
names(ump_runs)[3]<-"Standard Deviation"
ump_runs$Umpire <- sub("^$", "Unknown", ump_runs$Umpire)
ump_runs1 <- head(ump_runs[order(-ump_runs$Mean),], 15)

data.m <- melt(ump_runs1, id.vars='Umpire')
ggplot(data.m, aes(Umpire, value)) +   
  geom_bar(aes(fill = variable), position = "dodge", stat="identity") + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + xlab("") +
  ggtitle("Mean and Standard Deviations for Runs \n in Games Segmented by Home Plate Umpire")

#Inspect walks by umpire
mean_walks <- aggregate(BB.y ~ Umpire, data = games_final, mean)
sd_walks <- aggregate(BB.y ~ Umpire, data = games_final, sd)
ump_walks <- merge(mean_walks, sd_walks, by = "Umpire")
names(ump_walks)[2]<-"Mean"
names(ump_walks)[3]<-"Standard Deviation"
ump_walks$Umpire <- sub("^$", "Unknown", ump_walks$Umpire)
ump_walks1 <- head(ump_walks[order(-ump_walks$Mean),], 15)

data.m <- melt(ump_walks1, id.vars='Umpire')
ggplot(data.m, aes(Umpire, value)) +   
  geom_bar(aes(fill = variable), position = "dodge", stat="identity") + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + xlab("") +
  ggtitle("Mean and Standard Deviations for Walks \n in Games Segmented by Home Plate Umpire")

#Strikeouts by Umpire
mean_strikeouts <- aggregate(SO.y ~ Umpire, data = games_final, mean)
sd_strikeouts <- aggregate(SO.x ~ Umpire, data = games_final, sd)
ump_strikeouts <- merge(mean_strikeouts, sd_strikeouts, by = "Umpire")
names(ump_strikeouts)[2]<-"Mean"
names(ump_strikeouts)[3]<-"Standard Deviation"
ump_strikeouts$Umpire <- sub("^$", "Unknown", ump_strikeouts$Umpire)
ump_strikeouts1 <- head(ump_strikeouts[order(-ump_strikeouts$Mean),], 15)

data.m <- melt(ump_strikeouts1, id.vars='Umpire')
ggplot(data.m, aes(Umpire, value)) +   
  geom_bar(aes(fill = variable), position = "dodge", stat="identity") + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + xlab("") +
  ggtitle("Mean and Standard Deviations for Strikeouts \n in Games Segmented by Home Plate Umpire")

#Pitches by Umpire
mean_pitches <- aggregate(Pit ~ Umpire, data = games_final, mean)
sd_pitches <- aggregate(Pit ~ Umpire, data = games_final, sd)
ump_pitches <- merge(mean_pitches, sd_pitches, by = "Umpire")
names(ump_pitches)[2]<-"Mean"
names(ump_pitches)[3]<-"Standard Deviation"
ump_pitches$Umpire <- sub("^$", "Unknown", ump_pitches$Umpire)
ump_pitches1 <- head(ump_pitches[order(-ump_pitches$Mean),], 15)

data.m <- melt(ump_pitches1, id.vars='Umpire')
ggplot(data.m, aes(Umpire, value)) +   
  geom_bar(aes(fill = variable), position = "dodge", stat="identity") + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + xlab("") +
  ggtitle("Mean and Standard Deviations for Pitches \n in Games Segmented by Home Plate Umpire")

#Strikes by Umpire
mean_strikes <- aggregate(Str ~ Umpire, data = games_final, mean)
sd_strikes <- aggregate(Str ~ Umpire, data = games_final, sd)
ump_strikes <- merge(mean_strikes, sd_strikes, by = "Umpire")
names(ump_strikes)[2]<-"Mean"
names(ump_strikes)[3]<-"Standard Deviation"
ump_strikes$Umpire <- sub("^$", "Unknown", ump_strikes$Umpire)
ump_strikes1 <- head(ump_strikes[order(-ump_strikes$Mean),], 15)

data.m <- melt(ump_strikes1, id.vars='Umpire')
ggplot(data.m, aes(Umpire, value)) +   
  geom_bar(aes(fill = variable), position = "dodge", stat="identity") + 
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) + xlab("") +
  ggtitle("Mean and Standard Deviations for Strikes \n in Games Segmented by Home Plate Umpire")

#Let's see what variables are most correlated
correlations <- cor(games_final[10:59])
write.csv(correlations, file = "game_stat_correlations.csv")

##Cut each variable into quartiles
game_num <- games_final[10:59]

quartile3 <- function(x) {
  ntile(x, 4)
}

cut_data <- apply(game_num, 2, quartile3)
cut_data <- data.frame(cut_data)
cut_data <- data.frame(sapply(cut_data, as.factor))
summary(cut_data)

##Conduct association rules mining##

#Tell R to treat the data as transactions
game.trans <- as(cut_data, "transactions") 
summary(game.trans)

#Calculate, summarize, and plot rules
seg.rules <- apriori(game.trans, parameter = list(support=0.1, conf=0.3, target="rules"))
summary(seg.rules)
plot(seg.rules)

#Inspect 50 rules with the highest lift
seg.hi <- head(sort(seg.rules, by = "lift"), 50)
inspect(seg.hi)
plot(seg.hi, method = "graph", control = list(type="items"))

#Let's drop a few variables that probably should not be in the analysis
myvars <- names(cut_data) %in% c("BA", "OBP", "SLG", "OPS", "IR", "IS")
cut_data1 <- cut_data[!myvars]

##Re-conduct association rules mining##
#Tell R to treat the data as transactions
game.trans1 <- as(cut_data1, "transactions") 
summary(game.trans1)

#Calculate, summarize, and plot rules
seg.rules1 <- apriori(game.trans1, parameter = list(support=0.1, conf=0.3, target="rules"))
summary(seg.rules1)
plot(seg.rules1)

#Inspect 50 rules with the highest lift
seg.hi1 <- head(sort(seg.rules1, by = "lift"), 500)
inspect(seg.hi1)
plot(seg.hi1, method = "graph", control = list(type="items"))
