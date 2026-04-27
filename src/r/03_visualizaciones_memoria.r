Sys.setlocale("LC_ALL", "Spanish_Spain.utf8")
options(encoding = "UTF-8")

library(tidyverse)
library(ggplot2)
library(pROC)
library(scales)
library(gridExtra)

find_project_root <- function() {
    script_path <- tryCatch({
        args <- commandArgs(trailingOnly = FALSE)
        file_arg <- args[grep("--file=", args)]
        if (length(file_arg) > 0) normalizePath(sub("--file=", "", file_arg[1])) else NULL
    }, error = function(e) NULL)
    if (is.null(script_path)) {
        script_path <- tryCatch(normalizePath(sys.frame(1)$ofile), error = function(e) NULL)
    }
    if (!is.null(script_path)) {
        current <- dirname(script_path)
        for (i in 1:5) {
            if (dir.exists(file.path(current, "data"))) return(normalizePath(current))
            parent <- dirname(current)
            if (parent == current) break
            current <- parent
        }
    }
    current <- normalizePath(".")
    for (i in 1:5) {
        if (dir.exists(file.path(current, "data"))) return(current)
        parent <- dirname(current)
        if (parent == current) break
        current <- parent
    }
    stop("No se ha podido localizar la raíz del proyecto.")
}

root_dir <- find_project_root()
training_path <- file.path(root_dir, "data", "processed", "training_dataset.csv")
output_dir <- file.path(root_dir, "reports", "memoria")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

df <- read_csv(training_path, show_col_types = FALSE)

df <- df %>%
    mutate(
        Gender = factor(Gender, levels = c("F", "M")),
        Family_History = factor(Family_History, levels = c("No", "Yes")),
        Smoking_History = factor(Smoking_History, levels = c("No", "Yes")),
        Alcohol_Consumption = factor(Alcohol_Consumption, levels = c("No", "Yes")),
        Obesity_BMI = factor(Obesity_BMI, levels = c("Normal", "Overweight", "Obese")),
        Diet_Risk = factor(Diet_Risk, levels = c("Low", "Moderate", "High")),
        Physical_Activity = factor(Physical_Activity, levels = c("High", "Moderate", "Low")),
        Diabetes = factor(Diabetes, levels = c("No", "Yes")),
        Inflammatory_Bowel_Disease = factor(Inflammatory_Bowel_Disease, levels = c("No", "Yes")),
        Genetic_Mutation = factor(Genetic_Mutation, levels = c("No", "Yes"))
    )

modelo <- glm(
    Cancer ~ Age + Gender + Family_History + Smoking_History +
             Alcohol_Consumption + Obesity_BMI + Diet_Risk +
             Physical_Activity + Diabetes + Inflammatory_Bowel_Disease +
             Genetic_Mutation,
    data = df, family = binomial(link = "logit")
)

df$prob_pred <- predict(modelo, type = "response")

tema_memoria <- theme_minimal(base_size = 11) +
    theme(
        plot.title = element_text(face = "bold", size = 14, color = "#0D47A1"),
        plot.subtitle = element_text(color = "#546E7A", size = 10),
        legend.position = "top",
        panel.grid.minor = element_blank(),
        plot.caption = element_text(size = 8, color = "#78909C", face = "italic"),
        axis.title = element_text(face = "bold")
    )


# 1. ROC con bootstrap
cat("[1/5] Curva ROC con bootstrap...\n")

roc_obj <- roc(df$Cancer, df$prob_pred, levels = c(0, 1), direction = "<")
auc_val <- as.numeric(auc(roc_obj))
ci_auc <- ci.auc(roc_obj, conf.level = 0.95, method = "bootstrap", boot.n = 2000)
roc_ci <- ci.se(roc_obj, specificities = seq(0, 1, 0.05),
                conf.level = 0.95, boot.n = 500)

roc_df <- data.frame(
    fpr = 1 - as.numeric(rownames(roc_ci)),
    tpr_low = roc_ci[, 1],
    tpr_mid = roc_ci[, 2],
    tpr_high = roc_ci[, 3]
)

p_roc <- ggplot(roc_df, aes(x = fpr)) +
    geom_ribbon(aes(ymin = tpr_low, ymax = tpr_high), fill = "#1976D2", alpha = 0.2) +
    geom_line(aes(y = tpr_mid), color = "#1976D2", linewidth = 1.2) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#78909C") +
    annotate("text", x = 0.55, y = 0.25,
             label = sprintf("AUC = %.3f\nIC 95%%: [%.3f - %.3f]",
                             auc_val, ci_auc[1], ci_auc[3]),
             size = 4.5, color = "#0D47A1", fontface = "bold", hjust = 0) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    coord_fixed() +
    labs(
        title = "Curva ROC — Modelo 1 (regresión logística)",
        subtitle = "Área sombreada = IC del 95% vía bootstrap (500 réplicas)",
        x = "1 - Especificidad", y = "Sensibilidad",
        caption = paste("Dataset:", nrow(df), "muestras")
    ) +
    tema_memoria

ggsave(file.path(output_dir, "roc_con_bootstrap.png"),
       p_roc, width = 7, height = 7, dpi = 300, bg = "white")


# 2. Calibración
cat("[2/5] Curva de calibración...\n")

df_calib <- df %>%
    mutate(decil = ntile(prob_pred, 10)) %>%
    group_by(decil) %>%
    summarise(
        prob_media_predicha = mean(prob_pred),
        tasa_real = mean(Cancer),
        n = n(),
        se = sqrt(tasa_real * (1 - tasa_real) / n),
        low = pmax(0, tasa_real - 1.96 * se),
        high = pmin(1, tasa_real + 1.96 * se)
    )

p_calib <- ggplot(df_calib, aes(x = prob_media_predicha, y = tasa_real)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed",
                color = "#78909C", linewidth = 0.8) +
    geom_errorbar(aes(ymin = low, ymax = high),
                  width = 0.015, color = "#1976D2", alpha = 0.6) +
    geom_point(aes(size = n), color = "#1976D2", alpha = 0.85) +
    geom_smooth(method = "loess", se = FALSE, color = "#E53935",
                linetype = "dotted", formula = y ~ x) +
    scale_size_continuous(range = c(3, 8), guide = "none") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    coord_fixed() +
    labs(
        title = "Curva de calibración — Modelo 1",
        subtitle = "Concordancia entre probabilidad predicha y frecuencia observada (deciles)",
        x = "Probabilidad predicha (media del decil)",
        y = "Frecuencia real observada",
        caption = "Línea diagonal = calibración perfecta. Línea roja = ajuste LOESS."
    ) +
    tema_memoria

ggsave(file.path(output_dir, "calibration_plot.png"),
       p_calib, width = 7, height = 7, dpi = 300, bg = "white")


# 3. Forest plot RR literatura
cat("[3/5] Forest plot RR literatura...\n")

rr_data <- tribble(
    ~factor,                                  ~RR,   ~low,  ~high, ~fuente,
    "Mutación genética conocida",             4.00,  2.80,  5.70,  "NCI CCRAT",
    "Enfermedad inflamatoria (EII)",          2.93,  2.30,  3.73,  "Johnson 2013",
    "Antecedentes familiares 1er grado",      1.79,  1.51,  2.12,  "Butterworth 2006",
    "Obesidad (IMC ≥30)",                     1.33,  1.22,  1.45,  "Ma 2013",
    "Diabetes tipo 2",                        1.27,  1.15,  1.40,  "Johnson 2013",
    "Dieta alto riesgo",                      1.25,  1.13,  1.38,  "Johnson 2013",
    "Consumo regular de alcohol",             1.21,  1.10,  1.33,  "Johnson 2013",
    "Tabaquismo crónico",                     1.18,  1.07,  1.30,  "Johnson 2013",
    "Sobrepeso (IMC 25-29.9)",                1.10,  1.03,  1.17,  "Ma 2013",
    "Dieta moderada",                         1.10,  1.02,  1.19,  "Johnson 2013",
    "Actividad física moderada",              0.88,  0.80,  0.97,  "Johnson 2013",
    "Actividad física alta",                  0.76,  0.68,  0.85,  "Johnson 2013"
) %>%
    mutate(
        direccion = case_when(RR > 1 ~ "Riesgo", RR < 1 ~ "Protector", TRUE ~ "Neutral"),
        factor = factor(factor, levels = rev(factor))
    )

p_forest_rr <- ggplot(rr_data, aes(x = RR, y = factor, color = direccion)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "#78909C", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = low, xmax = high), height = 0.3, linewidth = 0.9) +
    geom_point(size = 4) +
    geom_text(aes(label = sprintf("RR=%.2f [%.2f-%.2f]", RR, low, high)),
              hjust = -0.12, size = 3, color = "#37474F") +
    scale_x_continuous(trans = "log2",
                       breaks = c(0.5, 0.75, 1, 1.5, 2, 3, 5),
                       limits = c(0.4, 10)) +
    scale_color_manual(values = c("Riesgo" = "#E53935", "Protector" = "#43A047", "Neutral" = "#90A4AE")) +
    labs(
        title = "Riesgos relativos del Modelo 1B (calculadora de literatura)",
        subtitle = "Valores e IC del 95% según meta-análisis publicados",
        x = "Riesgo Relativo (escala logarítmica)", y = "", color = "",
        caption = "Fuentes: Butterworth 2006, Ma 2013, Johnson 2013, NCI CCRAT"
    ) +
    tema_memoria + theme(axis.text.y = element_text(size = 10))

ggsave(file.path(output_dir, "forest_plot_rr_literatura.png"),
       p_forest_rr, width = 11, height = 7, dpi = 300, bg = "white")


# 4. Distribución de predicciones
cat("[4/5] Distribución predicciones...\n")

df_plot <- df %>%
    mutate(cohorte = factor(Cancer, levels = c(0, 1), labels = c("Control", "Cáncer")))

p_dist <- ggplot(df_plot, aes(x = prob_pred, fill = cohorte, color = cohorte)) +
    geom_density(alpha = 0.4, linewidth = 0.8) +
    geom_vline(xintercept = 0.3, linetype = "dashed", color = "#546E7A", linewidth = 0.5) +
    geom_vline(xintercept = 0.7, linetype = "dashed", color = "#546E7A", linewidth = 0.5) +
    annotate("rect", xmin = 0, xmax = 0.3, ymin = 0, ymax = Inf, fill = "#43A047", alpha = 0.05) +
    annotate("rect", xmin = 0.3, xmax = 0.7, ymin = 0, ymax = Inf, fill = "#FFA000", alpha = 0.05) +
    annotate("rect", xmin = 0.7, xmax = 1, ymin = 0, ymax = Inf, fill = "#E53935", alpha = 0.05) +
    annotate("text", x = 0.15, y = Inf, label = "BAJO", vjust = 1.5,
             color = "#43A047", fontface = "bold", size = 3.5) +
    annotate("text", x = 0.5, y = Inf, label = "MEDIO", vjust = 1.5,
             color = "#FFA000", fontface = "bold", size = 3.5) +
    annotate("text", x = 0.85, y = Inf, label = "ALTO", vjust = 1.5,
             color = "#E53935", fontface = "bold", size = 3.5) +
    scale_fill_manual(values = c("Control" = "#43A047", "Cáncer" = "#E53935")) +
    scale_color_manual(values = c("Control" = "#43A047", "Cáncer" = "#E53935")) +
    scale_x_continuous(limits = c(0, 1), labels = percent) +
    labs(
        title = "Distribución de probabilidades predichas por cohorte",
        subtitle = "Líneas punteadas = umbrales de estratificación (0.30 y 0.70)",
        x = "Probabilidad predicha de cáncer", y = "Densidad",
        fill = "Cohorte", color = "Cohorte"
    ) +
    tema_memoria

ggsave(file.path(output_dir, "distribucion_predicciones.png"),
       p_dist, width = 9, height = 6, dpi = 300, bg = "white")


# 5. Curvas de métricas vs umbral
cat("[5/5] Curvas métricas vs umbral...\n")

umbrales <- seq(0.01, 0.99, by = 0.01)
metricas <- data.frame(
    umbral = umbrales,
    sensibilidad = NA_real_, especificidad = NA_real_,
    precision = NA_real_, f1 = NA_real_
)

for (i in seq_along(umbrales)) {
    u <- umbrales[i]
    pred <- as.numeric(df$prob_pred >= u)
    tp <- sum(pred == 1 & df$Cancer == 1)
    fp <- sum(pred == 1 & df$Cancer == 0)
    fn <- sum(pred == 0 & df$Cancer == 1)
    tn <- sum(pred == 0 & df$Cancer == 0)
    sens <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
    esp  <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
    prec <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
    f1   <- ifelse((prec + sens) > 0, 2 * prec * sens / (prec + sens), 0)
    metricas$sensibilidad[i] <- sens
    metricas$especificidad[i] <- esp
    metricas$precision[i] <- prec
    metricas$f1[i] <- f1
}

metricas_long <- metricas %>%
    pivot_longer(-umbral, names_to = "metrica", values_to = "valor")

umbral_optimo <- metricas$umbral[which.max(metricas$f1)]
f1_max <- max(metricas$f1)

p_metricas <- ggplot(metricas_long, aes(x = umbral, y = valor, color = metrica)) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = umbral_optimo, linetype = "dashed",
               color = "#37474F", linewidth = 0.5) +
    annotate("text", x = umbral_optimo + 0.02, y = 0.05,
             label = sprintf("Umbral F1 máx = %.2f", umbral_optimo),
             hjust = 0, size = 3.5, color = "#37474F") +
    scale_color_manual(values = c(
        "sensibilidad"  = "#E53935",
        "especificidad" = "#1976D2",
        "precision"     = "#00897B",
        "f1"            = "#7B1FA2"
    ), labels = c(
        "sensibilidad"  = "Sensibilidad",
        "especificidad" = "Especificidad",
        "precision"     = "Precisión (VPP)",
        "f1"            = "F1-score"
    )) +
    scale_x_continuous(limits = c(0, 1), labels = percent) +
    scale_y_continuous(limits = c(0, 1), labels = percent) +
    labs(
        title = "Métricas del clasificador según umbral",
        subtitle = "Selección del punto de corte óptimo",
        x = "Umbral de clasificación", y = "Valor", color = "Métrica"
    ) +
    tema_memoria

ggsave(file.path(output_dir, "curvas_metricas.png"),
       p_metricas, width = 9, height = 6, dpi = 300, bg = "white")

cat("\n=================================================\n")
cat("  VISUALIZACIONES COMPLETADAS\n")
cat("=================================================\n")
for (f in list.files(output_dir)) cat("  -", f, "\n")
cat(sprintf("\nAUC Modelo 1: %.3f (IC 95%%: %.3f - %.3f)\n",
            auc_val, ci_auc[1], ci_auc[3]))
cat(sprintf("Umbral F1-óptimo: %.2f (F1=%.3f)\n", umbral_optimo, f1_max))