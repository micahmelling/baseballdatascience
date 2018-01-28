#Load libraries and read in data
library(AnomalyDetection)
library(ggplot2)
library(XML)

royals.parse <-htmlParse("http://www.baseball-reference.com/teams/tgl.cgi?team=KCR&t=b&year=2016")
royals.tab<-readHTMLTable(royals.parse, stringsAsFactors=FALSE)
royals.df<-royals.tab[[1]]
write.csv(royals.df, file = "royals2016.csv")
str(royals.df)

#The dataframe includes a couple of month headers; let's delete those
#They are rows 24, 53, 81, 108, 138, 
royals <- royals.df[-c(24, 53, 81, 108, 138), ]

#R read in all columns as characters; we'll need to covert the 
#appropriate columns to numeric values
cols.num <- royals[c(1:2, 7:30)] 
royals1 <- data.frame(sapply(cols.num, as.numeric))

royals_factors <- data.frame(cbind(royals$Date, royals$X, royals$Opp, 
                      royals$Rslt,royals$Thr))

royals_final <- data.frame(royals1, royals_factors)
names(royals_final)[27]<-"Date"
names(royals_final)[28]<-"Opponent"
names(royals_final)[29]<-"Result"
names(royals_final)[30]<-"Pitcher_Throws"
str(royals_final)

#Anomaly detection on different offensive statistics

#Runs
anomr = AnomalyDetectionVec(royals_final$R, max_anoms=0.02, direction="both", 
                            period = 7, plot = TRUE)
anomr$plot
anomr

#Home runs
anomhr = AnomalyDetectionVec(royals_final$HR, max_anoms=0.02, direction="both", 
                            period = 7, plot = TRUE)
anomhr$plot
anomhr

#Hits
anomh = AnomalyDetectionVec(royals_final$H, max_anoms=0.02, direction="both", 
                             period = 7, plot = TRUE)
anomh$plot
anomh

#Base on Balls
anombb = AnomalyDetectionVec(royals_final$BB, max_anoms=0.02, direction="both", 
                            period = 7, plot = TRUE)
anombb$plot
anombb

#Strikeouts
anomso = AnomalyDetectionVec(royals_final$SO, max_anoms=0.02, direction="both", 
                             period = 7, plot = TRUE)
anomso$plot
anomso

#Stolen Bases
anomsb = AnomalyDetectionVec(royals_final$SB, max_anoms=0.02, direction="both", 
                             period = 7, plot = TRUE)
anomsb$plot
anomsb


#Left on base
anomlob = AnomalyDetectionVec(royals_final$LOB, max_anoms=0.02, direction="both", 
                              period = 7, plot = TRUE)
anomlob$plot
anomlob

#Ground into double play
anomgdp = AnomalyDetectionVec(royals_final$GDP, max_anoms=0.02, direction="both", 
                              period = 7, plot = TRUE)
anomgdp$plot
anomgdp

#Hit by pitch
anomhbp = AnomalyDetectionVec(royals_final$HBP, max_anoms=0.02, direction="both", 
                              period = 7, plot = TRUE)
anomhbp$plot
anomhbp





