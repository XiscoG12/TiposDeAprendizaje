# ==============================================================================
# PROGRAMACIÓN PARA BIG DATA / CIENCIA DE DATOS
# ALGORITMOS DE APRENDIZAJE NO SUPERVISADO EN R
# ==============================================================================
# Integrantes: Erick Emmanuel Arizabalo Cepeda, Francisco Gabriel Reyna Montaño
# Docente: Jorge Peralta Escobar
# Periodo Escolar: Enero – Julio 2026
# ==============================================================================

# ------------------------------------------------------------------------------
# PREPARACIÓN DE ENTORNO Y INSTALACIÓN/CARGA DE LIBRERÍAS
# ------------------------------------------------------------------------------
# Descomentar si es necesario instalar los paquetes requeridos
# install.packages(c("tidyverse", "cluster", "factoextra", "gridExtra"))

library(tidyverse)    # Manipulación de datos y visualización elegante (ggplot2)
library(cluster)      # Funciones avanzadas de agrupamiento (silhouette, pam)
library(factoextra)   # Extracción y visualización interactiva de análisis multivariante
library(gridExtra)    # Organización de gráficos en cuadrículas

# ------------------------------------------------------------------------------
# CARGA Y PREPROCESAMIENTO DE DATOS (ANÁLISIS EXPLORATORIO - EDA)
# ------------------------------------------------------------------------------
# Carga del dataset WineQT
if (file.exists("WineQT.csv")) {
  datos_raw <- read.csv("WineQT.csv")
} else if (file.exists("WineQT.csv.csv")) {
  datos_raw <- read.csv("WineQT.csv.csv")
} else {
  stop("El archivo WineQT.csv no ha sido encontrado en el directorio actual.")
}

# Eliminación de columna identificadora 'Id'
datos_limpios <- datos_raw %>% select(-Id)

# Verificación de estructura inicial del dataset
cat("=== ESTRUCTURA DEL DATASET DE VINOS ===\n")
print(str(datos_limpios))

# Separación de características químicas y variable explicativa 'quality'
# Guardamos la variable quality para validación cruzada cualitativa
calidad_original <- datos_limpios$quality
caracteristicas_quimicas <- datos_limpios %>% select(-quality)

# Estandarización de variables (z-score scaling)
# Crucial para evitar sesgos por escalas numéricas dispares en el cálculo de distancias
datos_escalados <- as.data.frame(scale(caracteristicas_quimicas))

cat("\n=== DATOS QUÍMICOS ESCALADOS (MUESTRA) ===\n")
print(head(datos_escalados, 3))

# Fijación de semilla de aleatoriedad para asegurar reproducibilidad
set.seed(42)

# ------------------------------------------------------------------------------
# DETERMINACIÓN DEL NÚMERO ÓPTIMO DE CLÚSTERES (K)
# ------------------------------------------------------------------------------
# Determinación de K por el Método del Codo (WSS - Within Cluster Sum of Squares)
grafico_codo <- fviz_nbclust(datos_escalados, kmeans, method = "wss") +
  labs(title = "Método del Codo (WSS)",
       subtitle = "Selección de K basado en la reducción de inercia",
       x = "Número de Clústeres (K)", y = "Suma de Cuadrados Intra-Clúster") +
  theme_minimal()

# Determinación de K por el Método de la Silueta (Silhouette Coefficient)
grafico_silueta <- fviz_nbclust(datos_escalados, kmeans, method = "silhouette") +
  labs(title = "Método de la Silueta",
       subtitle = "Selección de K basada en la cohesión y separación media",
       x = "Número de Clústeres (K)", y = "Ancho de Silueta Promedio") +
  theme_minimal()

# Guardar ambos gráficos comparativos en un archivo PNG
cat("\n[Guardando gráfico del Método del Codo y Silueta en 'metodo_codo_silueta.png'...]\n")
png("metodo_codo_silueta.png", width = 1000, height = 500, res = 120)
grid.arrange(grafico_codo, grafico_silueta, ncol = 2)
dev.off()

# ------------------------------------------------------------------------------
# ALGORITMO 1: K-MEANS CLUSTERING (K-MEDIOS)
# ------------------------------------------------------------------------------
# Basándonos en el análisis previo, seleccionamos un K óptimo de 3 clústeres
k_optimo <- 3
cat(paste("\n--- Ejecutando K-Means con K =", k_optimo, "---\n"))

modelo_kmeans <- kmeans(datos_escalados, centers = k_optimo, nstart = 25)

# Adición de los resultados de clustering al dataframe original
datos_agrupados <- datos_limpios %>%
  mutate(Cluster_KMeans = factor(modelo_kmeans$cluster))

# Análisis de centroides (perfiles químicos promedio por clúster)
cat("\n=== PERFIL DE CENTROIDES QUÍMICOS POR CLÚSTER ===\n")
centroides <- datos_agrupados %>%
  group_by(Cluster_KMeans) %>%
  summarise(across(everything(), mean))
print(t(centroides))

# Visualización bi-dimensional de los clústeres de K-Means y guardar a archivo
plot_kmeans_clusters <- fviz_cluster(modelo_kmeans, data = datos_escalados,
             geom = "point",
             ellipse.type = "convex",
             ggtheme = theme_bw()) +
  labs(title = "Visualización de Clústeres de K-Means",
       subtitle = "Proyección sobre los dos componentes principales de mayor varianza")
cat("[Guardando visualización de clústeres en 'kmeans_clusters.png'...]\n")
ggsave("kmeans_clusters.png", plot = plot_kmeans_clusters, width = 8, height = 6, dpi = 150)

# ------------------------------------------------------------------------------
# ALGORITMO 2: AGRUPAMIENTO JERÁRQUICO (HIERARCHICAL CLUSTERING)
# ------------------------------------------------------------------------------
cat("\n--- Ejecutando Agrupamiento Jerárquico Aglomerativo (Método de Ward) ---\n")

# Cálculo de la matriz de distancias euclidianas
matriz_distancias <- dist(datos_escalados, method = "euclidean")

# Ajuste del modelo jerárquico mediante el método de Ward
modelo_jerarquico <- hclust(matriz_distancias, method = "ward.D2")

# Visualización del Dendrograma y guardar a archivo
# Para una visualización clara del dendrograma, graficaremos una muestra aleatoria de 150 elementos
set.seed(42)
muestra_indices <- sample(1:nrow(datos_escalados), 150)
dist_muestra <- dist(datos_escalados[muestra_indices, ], method = "euclidean")
hclust_muestra <- hclust(dist_muestra, method = "ward.D2")

cat("[Guardando Dendrograma en 'dendrograma_jerarquico.png'...]\n")
png("dendrograma_jerarquico.png", width = 900, height = 600, res = 120)
plot(hclust_muestra, hang = -1, cex = 0.6, main = "Dendrograma de Agrupamiento Jerárquico (Muestra de 150 Vinos)",
     xlab = "Muestras de Vinos", ylab = "Distancia de Fusión (Método de Ward)", sub = "")
# Dibujar rectángulos rojos que delimitan los 3 grupos sugeridos
rect.hclust(hclust_muestra, k = 3, border = "red")
dev.off()

# Cortar el árbol completo para obtener 3 clústeres jerárquicos
cluster_jerarquico_asignados <- cutree(modelo_jerarquico, k = 3)
datos_agrupados$Cluster_Jerarquico <- factor(cluster_jerarquico_asignados)

# ------------------------------------------------------------------------------
# ALGORITMO 3: ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
# ------------------------------------------------------------------------------
cat("\n--- Ejecutando Análisis de Componentes Principales (PCA) ---\n")

modelo_pca <- prcomp(caracteristicas_quimicas, scale. = TRUE)

# Resumen de varianza explicada por cada componente principal
cat("\n=== RESUMEN DE COMPONENTES PRINCIPALES ===\n")
print(summary(modelo_pca))

# Visualización del Scree Plot (Gráfico de Sedimentación) y guardar a archivo
scree_plot <- fviz_eig(modelo_pca, addlabels = TRUE, ylim = c(0, 40)) +
  labs(title = "Gráfico de Sedimentación (Scree Plot)",
       x = "Componentes Principales", y = "Porcentaje de Varianza Explicada") +
  theme_minimal()
cat("[Guardando gráfico de sedimentación en 'pca_scree_plot.png'...]\n")
ggsave("pca_scree_plot.png", plot = scree_plot, width = 7, height = 5, dpi = 150)

# Visualización del Biplot de PCA y guardar a archivo
# Coloreamos las observaciones basándonos en los clústeres de K-Means calculados
biplot_pca <- fviz_pca_biplot(modelo_pca,
                              label = "var", # Mostrar etiquetas solo de variables químicas
                              col.ind = datos_agrupados$Cluster_KMeans, # Color según clúster
                              palette = "jco",
                              addEllipses = TRUE,
                              ellipse.level = 0.95,
                              legend.title = "Clúster K-Means",
                              ggtheme = theme_minimal()) +
  labs(title = "Biplot Multivariado del PCA",
       subtitle = "Variables químicas (vectores de carga) y observaciones del vino")
cat("[Guardando Biplot del PCA en 'pca_biplot.png'...]\n")
ggsave("pca_biplot.png", plot = biplot_pca, width = 8, height = 6, dpi = 150)

# ------------------------------------------------------------------------------
# EVALUACIÓN Y VALIDACIÓN CRUZADA CON LA CALIDAD ORIGINAL (WINE QUALITY)
# ------------------------------------------------------------------------------
# Evaluamos estadísticamente si los clústeres químicos coinciden con la percepción sensorial
tabla_validacion <- table(KMeans = datos_agrupados$Cluster_KMeans, CalidadOriginal = calidad_original)
cat("\n=== RELACIÓN CRUZADA ENTRE CLÚSTERES QUÍMICOS Y CALIDAD DEL VINO ===\n")
print(tabla_validacion)

# Interpretación rápida del rendimiento
# Calculamos la calidad promedio por clúster químico
calidad_promedio_cluster <- datos_agrupados %>%
  group_by(Cluster_KMeans) %>%
  summarise(Calidad_Promedio = mean(quality), Total_Vinos = n())
print(calidad_promedio_cluster)

cat("\n[Proceso de Aprendizaje No Supervisado Completado Exitosamente]\n")
# ==============================================================================