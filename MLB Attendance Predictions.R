#Identify links from which to scrape the data
#We'll scrape data from 1995-2015; I want the data to be post-strike
#attendance: http://www.baseball-reference.com/leagues/MLB/1990-misc.shtml
#standings: http://www.baseball-reference.com/leagues/MLB/1990-standings.shtml
#pitching: http://www.baseball-reference.com/leagues/MLB/1990-standard-pitching.shtml
#fielding: http://www.baseball-reference.com/leagues/MLB/1990-standard-fielding.shtml
#batting: http://www.baseball-reference.com/leagues/MLB/1990-standard-batting.shtml

#Load libraries
library(XML)
library(ggplot2)
library(plyr)
library(dplyr)
library(car)

#Scrape attendance data
fetch_attendance <- function(year) {
  url <- paste0("http://www.baseball-reference.com/leagues/MLB/", year, "-misc.shtml")
  data <- readHTMLTable(url, stringsAsFactors = FALSE)
  data <- data[[1]]
  data$year <- year
  data
}

attendance <- ldply(1995:2015, fetch_attendance, .progress="text")

#Scrape standings data
fetch_standings <- function(year1) {
  url1 <- paste0("http://www.baseball-reference.com/leagues/MLB/", year1, "-standings.shtml")
  data1 <- readHTMLTable(url1, stringsAsFactors = FALSE)
  data1 <- data1[[2]]
  data1$year1 <- year1
  data1
}

standings <- ldply(1995:2015, fetch_standings, .progress="text")

#Scrape pitching data
fetch_pitching <- function(year2) {
  url2 <- paste0("http://www.baseball-reference.com/leagues/MLB/", year2, "-standard-pitching.shtml")
  data2 <- readHTMLTable(url2, stringsAsFactors = FALSE)
  data2 <- data2[[1]]
  data2$year2 <- year2
  data2
}

pitching <- ldply(1995:2015, fetch_pitching, .progress="text")


#Scrape fielding data
fetch_fielding <- function(year3) {
  url3 <- paste0("http://www.baseball-reference.com/leagues/MLB/", year3, "-standard-fielding.shtml")
  data3 <- readHTMLTable(url3, stringsAsFactors = FALSE)
  data3 <- data3[[1]]
  data3$year3 <- year3
  data3
}

fielding <- ldply(1995:2015, fetch_fielding, .progress="text")

#Scrape batting data
fetch_batting <- function(year4) {
  url4 <- paste0("http://www.baseball-reference.com/leagues/MLB/", year4, "-standard-batting.shtml")
  data4 <- readHTMLTable(url4, stringsAsFactors = FALSE)
  data4 <- data4[[1]]
  data4$year4 <- year4
  data4
}

batting <- ldply(1995:2015, fetch_batting, .progress="text")

#Now that we've scraped the data, we need to munge the data frames
#We'll merge the data frames on team name and year
#First, though, we need to clean up the year columns

#Change column names
names(standings)[24]<-"year"
names(pitching)[37]<-"year"
names(fielding)[17]<-"year"
names(batting)[30]<-"year"

#Merge the five data frames on team name and year
#We can only merge two data frames at a time, so we'll have to repeat the process a few times
teams_data <- merge(standings, attendance, by=c("Tm", "year"))
teams_data2 <- merge(teams_data, batting, by=c("Tm", "year"))
teams_data3 <- merge(teams_data2, fielding, by=c("Tm", "year"))
teams_data4 <- merge(teams_data3, pitching, by=c("Tm", "year"))

#It's possible that team names have changed over time
#Let's inspect the data frame to see if that's the case
#A "correct" team should have 21 records
teams_counts <- aggregate(year ~ Tm, data = teams_data4, length)
print(teams_counts)
teams_counts[order(teams_counts$year),]

#It looks like we might have issues with 9 records
#ARI came into the NL in 1998, so their data is fine as is
#Tampa Bay should also only have 18 seasons of data

#Duplicated columns will prevent us from running the next commands,
#so let's delete those here
teams_data4 <- teams_data4[c(-71,-92)]

#Combine MON and WSN
teams_data5 <- mutate(teams_data4, Tm = recode(Tm, "'MON'='WSN'"))

#Combine FLA and MIA
teams_data5 <- mutate(teams_data5, Tm = recode(Tm, "'FLA'='MIA'"))

#Combine TBD and TBR 
teams_data5 <- mutate(teams_data5, Tm = recode(Tm, "'TBD'='TBR'"))

#Combine CAL, ANA, and LAA
teams_data5 <- mutate(teams_data5, Tm = recode(Tm, "'CAL'='LAA'"))
teams_data5 <- mutate(teams_data5, Tm = recode(Tm, "'ANA'='LAA'"))

#Check the data frame to see if all now looks OK
teams_counts1 <- aggregate(year ~ Tm, data = teams_data5, length)
print(teams_counts1)

#All looks good, so let's create a final data frame with the columns we might be
#most interested in investigating

#Rad viz

#Parellel coordinates

#Cleveland dot plot

#Alluvial plots

#Ridge regression

#Lasso regression

#Elastic net




