#Read in and inspect data
library(Lahman) #https://cran.r-project.org/web/packages/Lahman/Lahman.pdf 
data(Teams) #http://rpackages.ianhowson.com/rforge/Lahman/man/Teams.html 
summary(Teams)

teams_subset <- Teams[c(7, 15:23, 27:28, 30:38)]

#Isolate selected numeric variables
#We'll also create a reference data frame, that will just have team name and year
#The reference set will be handy later
teams_subset <- Teams[c(7, 15:23, 27:28, 30:32, 34:38)] 
teams_reference <- Teams[c(1, 4, 7, 15:23, 27:28, 30:32, 33:38)] 

#Remove teams with missing values
teams_subset <- na.omit(teams_subset)
teams_reference <- na.omit(teams_reference)

#On the reference set, drop everything but year and franchise IDs
teams_reference <- teams_reference[c(1:2)]

#Make each variable per-game rather than an aggregate
teams_final <- sweep(teams_subset,1,unlist(teams_subset[,1]),"/")
summary(teams_final)

#Drop the games column
teams_final <- teams_final[(-1)]

#Create a visual of correlations among variables
library(corrplot)
library(gplots)

correlations <- cor(teams_final)
correlations <- round(correlations, digits=2)

corrplot(correlations)
corrplot(correlations, method="shade", shade.col=NA, tl.col="black")

#Look at relationships between selected variables
#To avoid over-plotting, we'll use the hexbin package
library(hexbin)
library(ggplot2)
p <- ggplot(teams_final, aes(x=E, y=RA))
p + stat_binhex() +
  scale_fill_gradient(low="lightblue", high="red") +
  ggtitle("Relationship Between Errors Per Game \n and Runs Allowed Per Game")

p1 <- ggplot(teams_final, aes(x=HR, y=R))
p1 + stat_binhex() +
  scale_fill_gradient(low="lightblue", high="red") +
  ggtitle("Relationship Between HR Per Game \n and Runs Per Game")

#Develop parallel coordinates plot of variables
library(MASS)
library(colorRamps)

c <- blue2red(100)
r <- cut(teams_final$SHO, 100)
parcoord(teams_final, col=c[as.numeric(r)])

h <- cut(teams_final$HR, 100)
parcoord(teams_final, col=c[as.numeric(h)])

#Conduct k-means cluster
teams_scaled <- scale(teams_final)
wss <- (nrow(teams_scaled)-1)*sum(apply(teams_scaled,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(teams_scaled,
                                     centers=i)$withinss)
plot(1:15, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares", main = "Elbow Plot for No. of Clusters")

set.seed(500)
fit1 <- kmeans(teams_final, 6, nstart=25) 

library(cluster)
set.seed(500)
clusplot(teams_scaled, fit1$cluster, color=TRUE, shade=TRUE,
         labels=2, lines=0, main = "PCA Plot of K-Means Cluster")

teams_final <- data.frame(teams_final, fit1$cluster)
cluster1 <- teams_final[which(teams_final$fit1.cluster=='1'),]
cluster2 <- teams_final[which(teams_final$fit1.cluster=='2'),]
cluster3 <- teams_final[which(teams_final$fit1.cluster=='3'),]
cluster4 <- teams_final[which(teams_final$fit1.cluster=='4'),]
cluster5 <- teams_final[which(teams_final$fit1.cluster=='5'),]
cluster6 <- teams_final[which(teams_final$fit1.cluster=='6'),]

summary(cluster1)
summary(cluster2)
summary(cluster3)
summary(cluster4)
summary(cluster5)
summary(cluster6)

#Merge clusters with reference dataset of teams and years
teams_reference <- data.frame(teams_reference, fit1$cluster)
teams1 <- teams_reference[which(teams_reference$fit1.cluster=='1'),]
teams2 <- teams_reference[which(teams_reference$fit1.cluster=='2'),]
teams3 <- teams_reference[which(teams_reference$fit1.cluster=='3'),]
teams4 <- teams_reference[which(teams_reference$fit1.cluster=='4'),]
teams5 <- teams_reference[which(teams_reference$fit1.cluster=='5'),]
teams6 <- teams_reference[which(teams_reference$fit1.cluster=='6'),]
team_clusters <- rbind(teams1, teams2, teams3, teams4, teams5, teams6)
write.csv(team_clusters, file = "Historical Team Clustering Results.csv")

print(teams1)
teams1$yearID <- as.factor(as.integer(teams1$yearID))
summary(teams1, 50)

print(teams2)
teams2$yearID <- as.factor(as.integer(teams2$yearID))
summary(teams2, 50)

print(teams3)
teams3$yearID <- as.factor(as.integer(teams3$yearID))
summary(teams3, 50)

print(teams4)
teams4$yearID <- as.factor(as.integer(teams4$yearID))
summary(teams4, 50)

print(teams5)
teams5$yearID <- as.factor(as.integer(teams5$yearID))
summary(teams5, 50)

print(teams6)
teams6$yearID <- as.factor(as.integer(teams6$yearID))
summary(teams6, 50)

#Conduct hierarchical cluster
library(cluster)
set.seed(100)
teams_hclust <- teams_scaled
dm = dist(teams_hclust,method="euclidean")
hclust_teams <- hclust(dm, method="complete")
plot(hclust_teams)

plot(cut(as.dendrogram(hclust_teams), h=8)$lower[[2]])
teams_hclust[c(307, 324), ]
teams_reference[c(307, 324), ]
teams_hclust[c(307, 304), ]
teams_reference[c(307, 304), ]

plot(cut(as.dendrogram(hclust_teams), h=4)$lower[[30]])
teams_hclust[c(768, 944), ]
teams_reference[c(768, 944), ]
teams_hclust[c(944, 771), ]
teams_reference[c(944, 771), ]

#Conduct PCA on the data
library("factoextra")
library("FactoMineR")
teams_final2 <- teams_final[c(-20:-21)]

teams_pca <- prcomp(teams_final2, scale = TRUE)
summary(teams_pca)
fviz_screeplot(teams_pca, ncp=10)

pca.var <- get_pca_var(teams_pca)
pca.var
pca.var$contrib
pca.var$coord

fviz_contrib(teams_pca, choice = "var", axes = 1)
fviz_contrib(teams_pca, choice = "var", axes = 2)

fviz_pca_var(teams_pca)
fviz_pca_var(teams_pca, col.var="contrib")




 









