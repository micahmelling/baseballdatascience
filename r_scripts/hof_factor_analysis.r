# Factor analysis on offensive statistics for HoFers

# Library imports
library(nFactors)
library(gplots)
library(RColorBrewer)
library(semPlot)

# Read in data
hof <- read.csv('hof_hitting_stats.csv')
  
# Determine number of factors
nScree(hof)
eigen(cor(hof))

# Looks like three factors will be optimal
# Run factor analysis using three factors
fa <- factanal(hof, factors = 3)
print(fa)

# Create a heatmap of the loadings
heatmap.2(fa$loadings, col = brewer.pal(9, "Greens"), trace = "none",
          key = FALSE, dend = 'none', Colv = FALSE, cexCol = 1.2,
          main = "\n\n\n\n\nFactor Loadings for HoF Hitting Stats")

# Create SEM plot of the factors
semPaths(fa, what = "est", residuals = FALSE, cut = 0.4,
         posCol = c("white", "darkgreen"), 
         negCol = c("white", "red"),
         edge.label.cex = 0.60, nCharNodes = 7)

###########################################################
# Lets do the same for pitching data

# Read in data
hof_pitch <- read.csv('hof_pitching_stats.csv')

# Determine number of factors
nScree(hof_pitch)
eigen(cor(hof_pitch))

# Looks like two factors will be optimal
# Run factor analysis using two factors
# However, the algorithm isn't working on this data
# with two factors, so we'll switch to 3
fa2 <- factanal(hof_pitch, factors = 3)
print(fa2)

# Create a heatmap of the loadings
heatmap.2(fa2$loadings, col = brewer.pal(9, "Greens"), trace = "none",
          key = FALSE, dend = 'none', Colv = FALSE, cexCol = 1.2,
          main = "\n\n\n\n\nFactor Loadings for HoF Pitching Stats")

# Create SEM plot of the factors
semPaths(fa2, what = "est", residuals = FALSE, cut = 0.4,
         posCol = c("white", "darkgreen"), 
         negCol = c("white", "red"),
         edge.label.cex = 0.60, nCharNodes = 7)
