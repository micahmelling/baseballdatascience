# Library imports

library(ggplot2)
library(MASS)
library(colorRamps)
library(FactoMineR)
library(factoextra)
library(gplots)

# Parallel coordinates for Pedro
pedro <- read.csv('pedro_sims.csv')
pedro <- pedro[-c(1)]

c <- blue2red(15)
r <- cut(pedro$Counter, 15)
parcoord(pedro, col=c[as.numeric(r)])

# Jitter plot for Clemens
clemens <- read.csv('clemens1997.csv')
clemens <- clemens[order(clemens$Similarity),] 

ggplot(clemens, aes(x=Pitcher_and_Year, y=Similarity)) + 
  geom_jitter(alpha=0.5, position = position_jitter(width = 0.1)) + 
  ggtitle("Pitchers Most Similar to 1997 Roger Clemens") + 
  labs(y="Similarity", x='Player and Year') +
  theme(axis.text.x = element_text(angle = 60, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5))

# Circular bar chart for Johnson
johnson <- read.csv('johnson2002.csv')
johnson <- johnson[order(johnson$Similarity),] 

ggplot(johnson, aes(x = Pitcher_and_Year, y = Similarity,
                    fill = Pitcher_and_Year)) + 
  geom_bar(width = 0.85, stat="identity") +    
  
  # To use a polar plot and not a basic barplot
  coord_polar(theta = "y") +    
  
  #Remove useless labels of axis
  xlab("") + ylab("") +
  
  #Increase ylim to avoid having a complete circle
  ylim(c(0,1.5)) + 
  
  #Add group labels close to the bars :
  geom_text(data = johnson, hjust = 1, size = 3,
            aes(x = Pitcher_and_Year, y = 0, label = Pitcher_and_Year)) +
  
  #Remove useless legend, y axis ticks and y axis text
  theme(legend.position = "none", axis.text.y = element_blank(), 
        axis.ticks = element_blank())

# Balloon plot for Greinke
greinke <- read.csv('greinke_sims.csv')
rownames(greinke) <- greinke[,1]
greinke <- greinke[-c(1)]

dt <- as.table(as.matrix(greinke))
balloonplot(t(dt), main='Pitching Stats', xlab="", ylab="",
            label=TRUE, show.margins=FALSE)
