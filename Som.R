library(kohonen)
library(tidyverse)
library(caret)

# Charger les données
df <- read.csv("marketing_campaign.csv", sep=";")

# Statistiques descriptives
summary(df)

# Sélection des variables pour segmentation
features <- c("Income", "Kidhome", "Teenhome", "MntWines", "MntFruits", 
              "MntMeatProducts", "MntFishProducts", "NumWebPurchases", 
              "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth")

# Normalisation des données
df_clean <- df %>% select(all_of(features)) %>% na.omit()
df_scaled <- as.data.frame(scale(df_clean))

# SOM pour segmentation
som_grid <- somgrid(xdim = 10, ydim = 10, topo = "hexagonal")
som_model <- som(as.matrix(df_scaled), grid = som_grid, rlen = 100)

# Remplacement de clusterSOM par une approche basée sur hclust
codes <- getCodes(som_model)  # Extraction des codes des neurones
dist_matrix <- dist(codes)    # Matrice de distances entre les codes
hc <- hclust(dist_matrix, method = "ward.D2") # Clustering hiérarchique
som_cluster <- cutree(hc, k = 5) # Définition du nombre de clusters

# Palette de couleurs
pretty_palette <- rainbow(length(unique(som_cluster)))

# Visualisation des clusters SOM
plot(som_model, type="mapping", bgcol = pretty_palette[som_cluster], main = "Clusters SOM")
add.cluster.boundaries(som_model, som_cluster)

# Autres visualisations SOM
plot(som_model, type="count", main="Nombre d'observations par cluster")
plot(som_model, type="codes", main="Clusters SOM avec les attributs")

# Attribution des clusters aux données d'origine
df_clean$SOM_Cluster <- som_cluster[som_model$unit.classif]

# Affichage des clusters
print(table(df_clean$SOM_Cluster))
