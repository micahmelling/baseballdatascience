# Career Length Survival Analysis
setwd("C:/Users/Micah/Desktop/Baseball Data Science/Career Length Survival Analysis")

# Import libraries
library(reshape)
library(dplyr)
library(tidyverse)
library(survival)
library(survminer)

# Read in data
df <- read.csv('player_data_for_survival_analysis.csv')

# Survival Analysis
s <- Surv(df$time_in_mlb, df$status)
class(s)

# Survival analysis that doesn't consider any groupings
survfit(s~1)
survfit(Surv(time_in_mlb, status)~1, data=df)
sfit <- survfit(Surv(time_in_mlb, status)~1, data=df)
ggsurvplot(sfit)

# Survival analysis with hits
sfit1 <- survfit(Surv(time_in_mlb, status) ~ hits,
                   data=df)

summary(sfit1)
summary(sfit1, times=seq(0, 8000, 500))
plot(sfit1)
ggsurvplot(sfit1)

# Survival analysis with hits and throws
sfit2 <- survfit(Surv(time_in_mlb, status) ~ hits + throws,
                 data=df)

ggsurvplot(sfit2)

# Survival analysis with birth_country
sfit3 <- survfit(Surv(time_in_mlb, status) ~ birth_country,
                 data=df)

ggsurvplot(sfit3)

# Survival analysis with height
sfit4 <- survfit(Surv(time_in_mlb, status) ~ height,
                 data=df)

ggsurvplot(sfit4)

# Cox regression
fit <- coxph(Surv(time_in_mlb, status) ~ average_salary + 
               birth_country + weight + height + hits +
               throws + age_at_debut, data = df)

fit


