# Library imports
library(ggplot2)
library(ggpubr)

# Read in data
transitions <- read.csv('scherzer_transitions.csv')
pitch_counts <- read.csv('pitches_in_counts.csv')

# Transition Probabilities Plot
ggdotchart(transitions, x = "Transition", y = "Probability", 
           color = "Probability",
           sorting = "descending",                       
           rotate = TRUE,                                
           dot.size = 2,                                 
           y.text.col = TRUE,                            
           ggtheme = theme_pubr()
)+
  theme_cleveland() 

# Pitch Counts Plot
ggdotchart(pitch_counts, x = "pitch", y = "pitch_percentage",
           color = "count", 
           palette = c("#FF0000", "#00FF00", "#0000FF",
                       "#00FFFF", "#800080", "#FF7F50",
                       "#FF8C00", "#008000", "#00FA9A",
                       "#4169E1", "#00FFFF", "#FFD700"),

           sorting = "ascending",                        
           add = "segments",                             
           ggtheme = theme_pubr()                        
)





