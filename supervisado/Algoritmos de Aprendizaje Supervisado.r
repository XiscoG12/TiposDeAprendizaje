# ==============================================================================
# PREPARACIÓN INICIAL Y CARGA DE DATOS
# ==============================================================================

# Instalar librerías necesarias si no se tienen
#install.packages(c("tidyverse", "caret", "rpart", "rpart.plot", "e1071", "class", "randomForest", "xgboost", "nnet"))

library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(class)
library(randomForest)
library(xgboost)
library(nnet)

# Cargar el dataset (Asegúrate de ajustar la ruta de tu archivo)
datos <- read.csv("Downloads/WineQT.csv")
# Eliminar la columna 'Id' ya que no aporta información predictiva
datos <- datos %>% select(-Id)

# Fijar semilla para que los resultados sean reproducibles
set.seed(123)

# Crear partición de entrenamiento (80%) y prueba (20%)
indice_entrenamiento <- createDataPartition(datos$quality, p = 0.8, list = FALSE)
datos_entrenamiento  <- datos[indice_entrenamiento, ]
datos_prueba         <- datos[-indice_entrenamiento, ]


# ==============================================================================
# 1. REGRESIÓN LINEAL (Linear Regression)
# ==============================================================================
# EN QUÉ DESTACA: Es el modelo base por excelencia para variables continuas. 
# Destaca por su extrema simplicidad, velocidad de cómputo y alta interpretabilidad.
# DIFERENCIAS Y LIMITACIONES: Asume una relación estrictamente en línea recta entre 
# las variables y la meta. Si los datos tienen curvas o interacciones complejas, falla.

modelo_lineal <- lm(quality ~ ., data = datos_entrenamiento)

# Predicción y Evaluación (Raíz del Error Cuadrático Medio - RMSE)
pred_lineal <- predict(modelo_lineal, newdata = datos_prueba)
rmse_lineal <- RMSE(pred_lineal, datos_prueba$quality)
cat("RMSE Regresión Lineal:", rmse_lineal, "\n")


# ==============================================================================
# TRANSFORMACIÓN PARA CLASIFICACIÓN
# ==============================================================================
# Para los siguientes algoritmos de clasificación, convertiremos la calidad en una 
# variable binaria: "Malo" (calidad <= 5) y "Bueno" (calidad >= 6).
convertir_clase <- function(df) {
  df %>% 
    mutate(calidad_binaria = ifelse(quality >= 6, "Bueno", "Malo")) %>%
    mutate(calidad_binaria = as.factor(calidad_binaria)) %>%
    select(-quality)
}

train_clasif <- convertir_clase(datos_entrenamiento)
test_clasif  <- convertir_clase(datos_prueba)


# ==============================================================================
# 2. REGRESIÓN LOGÍSTICA (Logistic Regression)
# ==============================================================================
# EN QUÉ DESTACA: Es el estándar de oro para clasificación binaria cuando buscas 
# entender el "por qué". Te da las probabilidades exactas de pertenecer a una clase.
# MEJORAS SOBRE R. LINEAL: En lugar de predecir una línea infinita (que daría valores 
# imposibles como calidad de vino de -5 o 100), aplica la función sigmoide para 
# acotar los resultados estrictamente entre 0 y 1 (probabilidad).

modelo_logistico <- glm(calidad_binaria ~ ., data = train_clasif, family = "binomial")

# Predicciones (Devuelve probabilidades, usamos umbral de 0.5)
prob_logistica <- predict(modelo_logistico, newdata = test_clasif, type = "response")
pred_logistica <- factor(ifelse(prob_logistica > 0.5, "Bueno", "Malo"), levels = c("Bueno", "Malo"))

# Evaluación
matriz_logistica <- confusionMatrix(pred_logistica, test_clasif$calidad_binaria)
cat("Precisión (Accuracy) Regresión Logística:", matriz_logistica$overall["Accuracy"], "\n")


# ==============================================================================
# 3. ÁRBOLES DE DECISIÓN (Decision Trees)
# ==============================================================================
# EN QUÉ DESTACA: Excelente para capturar reglas de negocio ("Si el alcohol es > 10.5 
# y la acidez es baja, entonces el vino es Bueno"). No requiere que los datos estén escalados.
# MEJORAS SOBRE LAS REGRESIONES: Puede modelar relaciones no lineales y cortes abruptos 
# en los datos de forma nativa sin necesidad de fórmulas complejas. Es totalmente visual.

modelo_arbol <- rpart(calidad_binaria ~ ., data = train_clasif, method = "class")

# Dibujar el árbol para ver las reglas de decisión
rpart.plot(modelo_arbol, main = "Árbol de Decisión - Calidad del Vino")

# Predicción y Evaluación
pred_arbol <- predict(modelo_arbol, newdata = test_clasif, type = "class")
matriz_arbol <- confusionMatrix(pred_arbol, test_clasif$calidad_binaria)
cat("Precisión Árbol de Decisión:", matriz_arbol$overall["Accuracy"], "\n")


# ==============================================================================
# 4. MÁQUINAS DE VECTORES DE SOPORTE (SVM)
# ==============================================================================
# EN QUÉ DESTACA: Es sumamente efectivo en espacios de alta dimensionalidad (muchas variables).
# MEJORAS SOBRE ÁRBOLES Y REGRESIONES: Introduce el "Truco del Kernel" (aquí usamos 'radial'). 
# Si los vinos buenos y malos están mezclados de forma compleja, SVM proyecta los datos a una 
# dimensión matemática superior donde sí pueda trazar una línea o frontera limpia para separarlos.

modelo_svm <- svm(calidad_binaria ~ ., data = train_clasif, kernel = "radial", probability = TRUE)

# Predicción y Evaluación
pred_svm <- predict(modelo_svm, newdata = test_clasif)
matriz_svm <- confusionMatrix(pred_svm, test_clasif$calidad_binaria)
cat("Precisión SVM:", matriz_svm$overall["Accuracy"], "\n")


# ==============================================================================
# 5. k-VECINOS MÁS CERCANOS (k-NN)
# ==============================================================================
# EN QUÉ DESTACA: Es un algoritmo intuitivo basado en analogías ("Dime con quién andas y te 
# diré quién eres"). No asume ninguna distribución matemática de los datos.
# DIFERENCIAS CLAVE: A diferencia de todos los anteriores, k-NN NO aprende reglas ni 
# coeficientes durante el entrenamiento. Solo guarda los datos y calcula distancias geométricas.
# NOTA DE DESARROLLO: Requiere obligatoriamente normalizar/escalar las variables.

# Escalamos los datos numéricos (excluyendo la variable objetivo)
escalador <- preProcess(train_clasif[, -12], method = c("center", "scale"))
train_escalado <- predict(escalador, train_clasif[, -12])
test_escalado  <- predict(escalador, test_clasif[, -12])

# Ejecución de KNN (usando k=5 vecinos como ejemplo)
pred_knn <- knn(train = train_escalado, test = test_escalado, 
                cl = train_clasif$calidad_binaria, k = 5)

# Evaluación
matriz_knn <- confusionMatrix(pred_knn, test_clasif$calidad_binaria)
cat("Precisión k-NN:", matriz_knn$overall["Accuracy"], "\n")


# ==============================================================================
# 6. NAIVE BAYES
# ==============================================================================
# EN QUÉ DESTACA: Es increíblemente veloz y requiere muy pocos datos para entrenar. 
# Trabaja mediante probabilidades condicionales combinadas.
# DIFERENCIAS Y LIMITACIONES: Asume "ingenuamente" que todas las características 
# químicas del vino son 100% independientes entre sí (ej. asume que el pH no tiene relación 
# con la acidez), lo cual es falso, pero a pesar de eso suele dar buenos resultados basales.

modelo_nb <- naiveBayes(calidad_binaria ~ ., data = train_clasif)

# Predicción y Evaluación
pred_nb <- predict(modelo_nb, newdata = test_clasif)
matriz_nb <- confusionMatrix(pred_nb, test_clasif$calidad_binaria)
cat("Precisión Naive Bayes:", matriz_nb$overall["Accuracy"], "\n")


# ==============================================================================
# 7. BOSQUES ALEATORIOS (Random Forest) - Ensamble en Paralelo
# ==============================================================================
# EN QUÉ DESTACA: Es uno de los algoritmos más robustos y estables de la industria. 
# MEJORAS SOBRE UN ÁRBOL ÚNICO: Un solo árbol de decisión tiende a sobreajustarse (memorizar) 
# los datos. Random Forest soluciona esto creando cientos de árboles (ej. 500) donde cada uno 
# ve datos y variables diferentes. Al final, hacen una votación democrática. Reduce el error drásticamente.

modelo_rf <- randomForest(calidad_binaria ~ ., data = train_clasif, ntree = 500)

# Predicción y Evaluación
pred_rf <- predict(modelo_rf, newdata = test_clasif)
matriz_rf <- confusionMatrix(pred_rf, test_clasif$calidad_binaria)
cat("Precisión Random Forest:", matriz_rf$overall["Accuracy"], "\n")


# ==============================================================================
# 8. GRADIENT BOOSTING (XGBoost) - Ensamble Secuencial
# ==============================================================================
# EN QUÉ DESTACA: Es el algoritmo rey en competencias de ciencia de datos debido a su altísima precisión.
# MEJORAS SOBRE RANDOM FOREST: En lugar de crear árboles independientes al mismo tiempo, 
# XGBoost crea los árboles de uno en uno, de forma secuencial. Cada nuevo árbol se enfoca 
# exclusivamente en corregir los errores cometidos por el árbol anterior (mediante Descenso del Gradiente).
# NOTA DE DESARROLLO: Requiere transformar los datos a matrices numéricas estrictas y etiquetas 0/1.

# Preparación específica para XGBoost
X_train <- as.matrix(train_clasif[, -12])
y_train <- ifelse(train_clasif$calidad_binaria == "Bueno", 1, 0)
X_test  <- as.matrix(test_clasif[, -12])

modelo_xgb <- xgboost(data = X_train, label = y_train, nrounds = 50, 
                      objective = "binary:logistic", verbose = 0)

# Predicción y Evaluación (Umbral 0.5)
prob_xgb <- predict(modelo_xgb, newdata = X_test)
pred_xgb <- factor(ifelse(prob_xgb > 0.5, "Bueno", "Malo"), levels = c("Bueno", "Malo"))

matriz_xgb <- confusionMatrix(pred_xgb, test_clasif$calidad_binaria)
cat("Precisión Gradient Boosting (XGBoost):", matriz_xgb$overall["Accuracy"], "\n")


# ==============================================================================
# 9. REDES NEURONALES (Multilayer Perceptron - MLP)
# ==============================================================================
# EN QUÉ DESTACA: Capacidad ilimitada para aproximar cualquier función matemática compleja. 
# Destaca cuando hay patrones ocultos extremadamente sutiles o interacciones cruzadas de variables.
# MEJORAS GENERALES: No asume linealidad, ni independencia, ni requiere estructuras de árboles. 
# Aprende ajustando los pesos de sus conexiones a través de backpropagation.
# LIMITACIONES: Es una "caja negra" (no es fácil saber por qué tomó una decisión) y requiere ajuste de parámetros.

# Entrenamos una red con una capa oculta de 5 neuronas
modelo_nn <- nnet(calidad_binaria ~ ., data = train_clasif, size = 5, maxnwts = 2000, trace = FALSE)

# Predicción y Evaluación
pred_nn <- predict(modelo_nn, newdata = test_clasif, type = "class")
pred_nn <- factor(pred_nn, levels = c("Bueno", "Malo"))

matriz_nn <- confusionMatrix(pred_nn, test_clasif$calidad_binaria)
cat("Precisión Red Neuronal:", matriz_nn$overall["Accuracy"], "\n")


# ==============================================================================
# RESUMEN COMPARATIVO FINAL DE CLASIFICACIÓN
# ==============================================================================
cat("\n--- TABLA DE RENDIMIENTO FINAL (ACCURACY) ---\n")
cat("Regresión Logística: ", matriz_logistica$overall["Accuracy"], "\n")
cat("Árbol de Decisión:   ", matriz_arbol$overall["Accuracy"], "\n")
cat("SVM:                 ", matriz_svm$overall["Accuracy"], "\n")
cat("k-NN:                ", matriz_knn$overall["Accuracy"], "\n")
cat("Naive Bayes:         ", matriz_nb$overall["Accuracy"], "\n")
cat("Random Forest:       ", matriz_rf$overall["Accuracy"], "\n")
cat("XGBoost:             ", matriz_xgb$overall["Accuracy"], "\n")
cat("Red Neuronal:        ", matriz_nn$overall["Accuracy"], "\n")