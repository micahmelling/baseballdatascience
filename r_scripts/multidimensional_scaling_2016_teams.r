#Multidimensional scaling and hierarchical clustering on 2016 offensive statistics

#Load libraries
library(XML)
library(plyr)
library(dplyr)
library(stringr)
library(Hmisc)
library(MASS)
library(cluster)

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

#I get a script out of bounds error when I run a loop; looks like we may have to manually 
#insert each team
ARI_offense <- ldply("ARI", fetch_offense, .progress="text")
ATL_offense <- ldply("ATL", fetch_offense, .progress="text")
BAL_offense <- ldply("BAL", fetch_offense, .progress="text")
CHW_offense <- ldply("CHW", fetch_offense, .progress="text")
CIN_offense <- ldply("CIN", fetch_offense, .progress="text")
COL_offense <- ldply("COL", fetch_offense, .progress="text")
DET_offense <- ldply("DET", fetch_offense, .progress="text")
HOU_offense <- ldply("HOU", fetch_offense, .progress="text")
KCR_offense <- ldply("KCR", fetch_offense, .progress="text")
LAA_offense <- ldply("LAA", fetch_offense, .progress="text")
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
TOR_offense <- ldply("TOR", fetch_offense, .progress="text")

#The scraper did not work on the following teams for some reason,
#so I created a slightly different scraper for just these ones
fetch_offense2 <- function(team) {
  url <- paste0("http://www.baseball-reference.com/teams/tgl.cgi?team=", team, "&t=b&year=2016.com")
  data <- readHTMLTable(url, stringsAsFactors = FALSE)
  data <- data[[2]]
  data$team <- team
  data
}

TEX_offense <- ldply("TEX", fetch_offense2, .progress="text")
WSN_offense <- ldply("WSN", fetch_offense2, .progress="text")
LAD_offense <- ldply("LAD", fetch_offense2, .progress="text")
CLE_offense <- ldply("CLE", fetch_offense2, .progress="text")
BOS_offense <- ldply("BOS", fetch_offense2, .progress="text")
CHC_offense <- ldply("CHC", fetch_offense2, .progress="text")

#Bind the data frames
offense <- rbind(ARI_offense, ATL_offense, BAL_offense, BOS_offense, CHC_offense,
                            CHW_offense, CIN_offense, CLE_offense, COL_offense, DET_offense, HOU_offense, KCR_offense,
                            LAA_offense, LAD_offense, MIA_offense, MIL_offense, MIN_offense, NYM_offense, NYY_offense,
                            OAK_offense, PHI_offense, PIT_offense, SDP_offense, SEA_offense, SFG_offense, STL_offense,
                            TBR_offense, TEX_offense, TOR_offense, WSN_offense)

#Remove monthly headers in the dataframe
offense <- offense[!grepl("PA", offense$Opp),]

#Select only the columns we need and convert to the correct data type
offense_num <- subset(offense, select = c(7:24, 29))
offense_cat <- offense[33]

offense_cat$team <- as.factor(offense_cat$team)
offense_num <- data.frame(sapply(offense_num, as.numeric))

offense_sub <- cbind(offense_cat, offense_num)

#Aggregate the stats by team
totals <- aggregate(. ~ team, offense_sub, sum)

#Calculate, BA, OBP, and SLG for each team
totals$BA <- totals$H / totals$AB

totals$OBP <- (totals$H + totals$BB + totals$HBP) / (totals$AB + totals$BB + totals$HBP + 
                                                       totals$SF)

totals$X1B <- totals$H - (totals$X2B + totals$X3B + totals$HR)
totals$SLG <- (totals$X1B + totals$X2B * 2 + totals$X3B * 3 + totals$HR * 4) / totals$AB

#Apply multidimensional scaling to the data
rownames(totals) <- totals[, 1]
totals <- totals[2:24]
totals <- scale(totals)
  
team.dist <- dist(totals)
team.mds <- cmdscale(team.dist)

plot(team.mds, type = "n")
text(team.mds, row.names(team.mds))

#Apply hierarchical clustering to the data
set.seed(100)
teams_hclust <- totals
dm = dist(teams_hclust,method="euclidean")
hclust_teams <- hclust(dm, method="complete")
plot(hclust_teams)
