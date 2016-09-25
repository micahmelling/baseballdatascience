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
library(data.table)
library(stringr)
library(alluvial)
library (glmnet)

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
#We can only merge two data frames at a time
#So we'll have to repeat the process a few times
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

#Let's inspect the data to see if it all looks good
str(teams_data5)

#We need to remove the commas from the attendance column, since R will 
#have trouble reading it
remove_commas <- function(x) {
  x <- str_replace_all(x, ",", "") 
}

teams_data5$Attendance <- remove_commas(teams_data5$Attendance)

#All the colums have been read as characters
#Let's identify the numeric columns and make them numeric
#Let's also rename the data frame while we're at it
columns <- subset(teams_data5, select = c(5:13, 25:33, 40:81, 84:117))
teams <- data.frame(sapply(columns, as.numeric))
teams_other <- subset(teams_data5, select = c(-5:-13, -25:-33, -40:-81, -84:-117))
teams_final <- data.frame(teams, teams_other)

#Let's add a column to the data frame that identifies which quartile
#each team falls under in terms of wins
setDT(teams_final)
teams_final[,wins_quartile:=cut(W.x,
                      breaks=quantile(W.x,probs=seq(0,1,by=1/4)),
                      labels=1:4,right=F)]

str(teams_final$wins_quartile)

#I want to create rad viz plots but found it quite cumbersome in R
#So, let's write CSVs of the data we want to visualize and import to
#Python, which makes rad viz pretty easy using Pandas
offensive_stats <- subset(teams_final, select = c(118,25,26,29,31,34))
write.csv(offensive_stats, file = "offensive_stats.csv")

pitching_stats <- subset(teams_final, select = c(118,56,67,79,81))
write.csv(pitching_stats, file = "pitching_stats.csv")

#Develop cleveland dot plots
#Wins by Season
ggplot(teams_final)+ geom_point(aes(x=Tm, y=W.x), colour = "blue") +
  coord_flip() + ggtitle("Single Season Win Totals") + 
  xlab("Team") + ylab("Wins")

#Total Wins
wins_total <- aggregate(W.x ~ Tm, data = teams_final, sum) 
wins_total1$Tm <-factor(wins_total$Tm, levels=wins_total[order(wins_total$W.x), "Tm"])

ggplot(wins_total1)+ geom_point(aes(x=Tm, y=W.x), colour = "blue") +
  coord_flip() + ggtitle("Win Totals from 1995-2015") + 
  xlab("Team") + ylab("Wins")

#Total Home Runs
hr_total <- aggregate(HR.x ~ Tm, data = teams_final, sum) 
hr_total$Tm <-factor(hr_total$Tm, levels=hr_total[order(hr_total$HR.x), "Tm"])

ggplot(hr_total)+ geom_point(aes(x=Tm, y=HR.x), colour = "blue") +
  coord_flip() + ggtitle("Home Run Totals from 1995-2015") + 
  xlab("Team") + ylab("Home Runs")

#Total Stolen Bases
sb_total <- aggregate(SB ~ Tm, data = teams_final, sum) 
sb_total$Tm <-factor(sb_total$Tm, levels=sb_total[order(sb_total$SB), "Tm"])

ggplot(sb_total)+ geom_point(aes(x=Tm, y=SB), colour = "blue") +
  coord_flip() + ggtitle("Stolen Base Totals from 1995-2015") + 
  xlab("Team") + ylab("Stolen Bases")

#Total Errors
e_total <- aggregate(E ~ Tm, data = teams_final, sum) 
e_total$Tm <-factor(e_total$Tm, levels=e_total[order(e_total$E), "Tm"])

ggplot(e_total)+ geom_point(aes(x=Tm, y=E), colour = "blue") +
  coord_flip() + ggtitle("Error Totals from 1995-2015") + 
  xlab("Team") + ylab("Errors")

#Total Earned Runs
er_total <- aggregate(ER ~ Tm, data = teams_final, sum) 
er_total$Tm <-factor(er_total$Tm, levels=er_total[order(er_total$ER), "Tm"])

ggplot(er_total)+ geom_point(aes(x=Tm, y=ER), colour = "blue") +
  coord_flip() + ggtitle("Earned Run Totals from 1995-2015") + 
  xlab("Team") + ylab("Earned Runs")

#Alluvial plot
wins_over_time <- subset(teams_final, select = c(95,96,2))
selected_teams <- filter(wins_over_time, Tm == "NYY" | Tm == "ATL" | Tm == "STL" 
                         | Tm == "SFG" | Tm == "BOS" | Tm == "TEX")

alluvial_ts(selected_teams, title = "Wins over Time")

#Now that we've inspected the data, let's do some prediction
#What seems to impact attendance?
#Let's start with subsetting our data frame to only include the columns we
#propose are predictive
teams_subset <- subset(teams_final, select = c(10,2,12,13,17,25,34,56,73,77,95))
summary(teams_subset)
teams_subset$Tm <- as.factor(as.character(teams_subset$Tm))

#Let's also sum the attendance column for context
sum(teams_subset$Attendance)
#1,492,344,734

#Split data into training and test sets
set.seed(10)
train = sample(1: nrow(x), nrow(x)/2)
test = (-train )
y.test = y[test]

#Ordinary least squares
#I'm interested in seeing what the coefficients will look like if we feed the 
#model the entire dataset
pairs(teams_subset)
ols_model <- lm(Attendance ~ 0 + W.x + BatAge.x + PAge.x + X.A.S +
                  R.y + SO.x + E + SV + ER + Tm, data=teams_subset)
plot(ols_model)
summary(ols_model)
vif(ols_model)

#Looks like we have some issues with multi-collinearity, so let's pivot to
#ridge and lasso

#Ridge regression
#Create matrices needed for the glmnet package
x <- model.matrix (Attendance ~.,teams_subset )[,-1]
y=teams_subset$Attendance

#Run the ridge regression model
ridge_model = glmnet(x[train,], y[train], alpha = 0)

#Use cross-validation to determine the best value for lambda
cv.out = cv.glmnet(x[train,], y[train], alpha = 0)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam

#Run the ridge regression on the full dataset using the optimal lambda value
#View the coefficients of the model
out = glmnet(x,y,alpha =0)
ridge.coef = predict(out, type ="coefficients", s=bestlam)
ridge.coef

#Lasso regression
#Follow the same process as above, except we need to set alpha = 1

#Run the lasso regression model
lasso_model = glmnet(x[train,], y[train], alpha = 1)

#Use cross-validation to determine the best lambda
cv.out1 = cv.glmnet(x[train,], y[train], alpha = 1)
plot(cv.out1)
bestlam1 = cv.out$lambda.min
bestlam1

#Run the lasso regression on the full dataset using the optimal lambda value
#View the coefficients of the model
out1 = glmnet(x,y,alpha = 1)
lasso.coef = predict(out1, type ="coefficients", s=bestlam1)
lasso.coef

#Lastly, let's look at the MSE of each model
#MSE of the OLS model
#OLS is simply the same as setting lambda equal to zero
ols_pred = predict(ridge.mod ,s=0, newx=x[test,], exact=T)
ols_mse <- mean((ols_pred -y.test)^2)
print(ols_mse)
sqrt(ols_mse)
#MSE: 197,876,169,581
#RMSE:444,832

#MSE of the ridge regression model
ridge.pred = predict(ridge_model,s=bestlam ,newx=x[test ,])
ridge_mse <- mean((ridge.pred-y.test)^2)
print(ridge_mse)
sqrt(ridge_mse)
#MSE: 189,415,000,000
#RMSE: 435,218

#MSE of the lasso regression model
lasso.pred = predict(lasso_model,s=bestlam ,newx=x[test ,])
lasso_mse <- mean((lasso.pred-y.test)^2)
print(lasso_mse)
sqrt(lasso_mse)
#MSE: 220,407,311,134
#RMSE: 469,475



