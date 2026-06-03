# ==============================================================================
# PROGRAMACIÓN PARA BIG DATA / CIENCIA DE DATOS
# IMPLEMENTACIÓN DE APRENDIZAJE POR REFUERZO: Q-LEARNING DESDE CERO
# ==============================================================================
# Integrantes: Erick Emmanuel Arizabalo Cepeda, Francisco Gabriel Reyna Montaño
# Docente: Jorge Peralta Escobar
# Periodo Escolar: Enero – Julio 2026
# ==============================================================================

library(tidyverse)

# ------------------------------------------------------------------------------
# CARGA DE DATOS Y CONSTRUCCIÓN DEL ENTORNO BASADO EN WINEQT
# ------------------------------------------------------------------------------
if (file.exists("WineQT.csv")) {
  datos_raw <- read.csv("WineQT.csv")
} else if (file.exists("WineQT.csv.csv")) {
  datos_raw <- read.csv("WineQT.csv.csv")
} else {
  stop("El archivo WineQT.csv no ha sido encontrado en el directorio actual.")
}

# Limpieza básica
datos_env <- datos_raw %>% select(residual.sugar, pH, quality)

# Discretizamos las variables para construir un espacio de estados finito (Grid World Químico)
# Clasificamos Azúcar Residual y pH en 3 categorías discretas (1: Bajo, 2: Medio, 3: Alto)
datos_discretos <- datos_env %>%
  mutate(
    sugar_cat = as.numeric(cut(residual.sugar, breaks = 3, labels = c(1, 2, 3))),
    ph_cat = as.numeric(cut(pH, breaks = 3, labels = c(1, 2, 3)))
  )

# Calculamos la calidad promedio real en WineQT para cada combinación de categorías químicas
# Esto definirá la función de recompensa empírica del entorno
tabla_recompensa_empirica <- datos_discretos %>%
  group_by(sugar_cat, ph_cat) %>%
  summarise(Calidad_Media = mean(quality), .groups = 'drop')

cat("=== RECOMPENSA EMPÍRICA (CALIDAD HISTÓRICA POR ESTADO) ===\n")
print(tabla_recompensa_empirica)

# Definimos los 9 Estados (Combinaciones de Azúcar y pH)
# Estado = (sugar_cat - 1) * 3 + ph_cat
mapear_estado <- function(sugar, ph) {
  return((sugar - 1) * 3 + ph)
}

# Reconstruimos la recompensa por estado en un vector plano de tamaño 9
recompensa_base_estados <- numeric(9)
for (i in 1:9) {
  sugar <- ceiling(i / 3)
  ph <- ((i - 1) %% 3) + 1
  valor_medio <- tabla_recompensa_empirica %>%
    filter(sugar_cat == sugar, ph_cat == ph) %>%
    pull(Calidad_Media)

  # Si un estado no tiene muestras, asignamos una calidad por defecto baja
  if (length(valor_medio) == 0) {
    recompensa_base_estados[i] <- 4.5
  } else {
    recompensa_base_estados[i] <- valor_medio
  }
}

# ------------------------------------------------------------------------------
# DEFINICIÓN DE PARÁMETROS DEL MDP Y EL AGENTE
# ------------------------------------------------------------------------------
# Número de Estados: 9
# Número de Acciones: 3 (1: Mantener, 2: Desacidificar (pH+1), 3: Endulzar (Sugar+1))
num_estados <- 9
num_acciones <- 3

# Costos asociados a cada acción para evitar optimizaciones triviales
costos_acciones <- c(0.00, 0.20, 0.15)

# Función de Transición del Entorno (Simulador de mezcla y aditivos químicos)
paso_entorno <- function(estado_actual, accion) {
  sugar <- ceiling(estado_actual / 3)
  ph <- ((estado_actual - 1) %% 3) + 1

  costo <- costos_acciones[accion]
  penalizacion_invalido <- 0

  if (accion == 1) {
    # Mantener: El estado no se altera
  } else if (accion == 2) {
    # Desacidificar: Incrementa pH (disminuye acidez)
    if (ph < 3) {
      ph <- ph + 1
    } else {
      penalizacion_invalido <- -1.5 # Penalización por desbordar límites físicos
    }
  } else if (accion == 3) {
    # Endulzar: Incrementa azúcar residual
    if (sugar < 3) {
      sugar <- sugar + 1
    } else {
      penalizacion_invalido <- -1.5 # Penalización
    }
  }

  nuevo_estado <- mapear_estado(sugar, ph)

  # Recompensa = Calidad del nuevo estado - costo de acción + penalización
  recompensa <- recompensa_base_estados[nuevo_estado] - costo + penalizacion_invalido

  return(list(s_siguiente = nuevo_estado, r = recompensa))
}

# ------------------------------------------------------------------------------
# ALGORITMO Q-LEARNING DESDE CERO
# ------------------------------------------------------------------------------
# Hiperparámetros de Aprendizaje
alfa <- 0.1         # Tasa de aprendizaje (Learning rate)
gamma <- 0.9        # Factor de descuento (Discount factor)
epsilon <- 0.3      # Probabilidad de exploración inicial (Epsilon-greedy)
decay_rate <- 0.995 # Factor de decaimiento del epsilon por episodio
num_episodios <- 1000

# Inicialización de la Tabla Q con ceros
tabla_q <- matrix(0, nrow = num_estados, ncol = num_acciones)
rownames(tabla_q) <- paste("Estado_", 1:9, sep="")
colnames(tabla_q) <- c("Mantener", "Desacidificar", "Endulzar")

# Registro para analizar curvas de rendimiento del agente
historial_recompensas <- numeric(num_episodios)

cat("\n--- Iniciando Entrenamiento del Agente Sommelier (Q-Learning) ---\n")
set.seed(123)

for (episodio in 1:num_episodios) {
  # El vino inicial inicia en un estado aleatorio
  estado_actual <- sample(1:num_estados, 1)
  recompensa_acumulada <- 0
  pasos_maximos <- 15 # Límite de acciones por lote de vino para evitar bucles

  for (paso in 1:pasos_maximos) {
    # Selección de Acción Epsilon-Greedy
    if (runif(1) < epsilon) {
      # Exploración: Selección de acción aleatoria
      accion <- sample(1:num_acciones, 1)
    } else {
      # Explotación: Selección del valor máximo conocido en la Tabla Q
      # Si hay empates, se selecciona uno aleatoriamente
      valores_estado <- tabla_q[estado_actual, ]
      accion <- which(valores_estado == max(valores_estado))
      if (length(accion) > 1) {
        accion <- sample(accion, 1)
      }
    }

    # Ejecución de la acción y observación del entorno
    resultado <- paso_entorno(estado_actual, accion)
    s_sig <- resultado$s_siguiente
    recompensa <- resultado$r

    # Regla de Actualización de Q-Learning (Ecuación de Bellman)
    mejor_q_siguiente <- max(tabla_q[s_sig, ])
    tabla_q[estado_actual, accion] <- tabla_q[estado_actual, accion] +
      alfa * (recompensa + gamma * mejor_q_siguiente - tabla_q[estado_actual, accion])

    recompensa_acumulada <- recompensa_acumulada + recompensa
    estado_actual <- s_sig
  }

  # Registrar la recompensa total del episodio
  historial_recompensas[episodio] <- recompensa_acumulada

  # Decaimiento del parámetro epsilon (hacia una mayor explotación)
  epsilon <- epsilon * decay_rate
}

cat("\n[Entrenamiento Finalizado con Éxito]\n")

# ------------------------------------------------------------------------------
# EXTRACCIÓN DE LA POLÍTICA ÓPTIMA DE TRATAMIENTO QUÍMICO
# ------------------------------------------------------------------------------
cat("\n=== TABLA Q FINAL APRENDIDA POR EL AGENTE ===\n")
print(round(tabla_q, 3))

# Extraemos la mejor acción (política óptima pi*) para cada estado
politica_optima <- apply(tabla_q, 1, which.max)
nombres_acciones <- c("Mantener", "Desacidificar", "Endulzar")

cat("\n=== POLÍTICA ÓPTIMA DE OPTIMIZACIÓN DE VINOS ===\n")
for (i in 1:num_estados) {
  sugar <- ceiling(i / 3)
  ph <- ((i - 1) %% 3) + 1
  desc_estado <- paste("Azúcar:", sugar, "| pH:", ph)
  accion_optima <- nombres_acciones[politica_optima[i]]
  cat(sprintf("Estado %d (%s) -> Acción Óptima Recomendada: %s\n", i, desc_estado, accion_optima))
}

# ------------------------------------------------------------------------------
# VISUALIZACIÓN DE LA CURVA DE APRENDIZAJE Y CONVERGENCIA
# ------------------------------------------------------------------------------
# Suavizamos los resultados mediante una media móvil para observar mejor la convergencia
df_progreso <- data.frame(
  Episodio = 1:num_episodios,
  Recompensa = historial_recompensas
) %>%
  mutate(Recompensa_Suave = stats::filter(Recompensa, rep(1/20, 20), sides = 2))

grafico_convergencia <- ggplot(df_progreso, aes(x = Episodio)) +
  geom_line(aes(y = Recompensa), alpha = 0.15, color = "darkred") +
  geom_line(aes(y = Recompensa_Suave), color = "darkred", linewidth = 1) +
  labs(title = "Curva de Convergencia del Agente de Q-Learning",
       subtitle = "Incremento del Retorno Neto Promedio mediante Suavizado de 20 Episodios",
       x = "Episodio de Entrenamiento", y = "Recompensa Acumulada") +
  theme_minimal()

cat("[Guardando curva de convergencia en 'qlearning_convergencia.png'...]\n")
ggsave("qlearning_convergencia.png", plot = grafico_convergencia, width = 8, height = 5, dpi = 150)
# ==============================================================================
