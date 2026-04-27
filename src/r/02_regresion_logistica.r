# Regresión logística del Modelo 1 con:
#   - Odds Ratios (OR) con IC del 95%
#   - Valores p de Wald
#   - Forest plot de publicación
#   - Comparación de OR con los RR de literatura
#

Sys.setlocale("LC_ALL", "Spanish_Spain.utf8")
options(encoding = "UTF-8")

library(tidyverse)
library(ggplot2)
library(broom)
library(scales)

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
output_dir <- file.path(root_dir, "reports", "regresion_logistica")
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

cat("[1/4] Ajustando regresión logística...\n")

modelo <- glm(
    Cancer ~ Age + Gender + Family_History + Smoking_History +
             Alcohol_Consumption + Obesity_BMI + Diet_Risk +
             Physical_Activity + Diabetes + Inflammatory_Bowel_Disease +
             Genetic_Mutation,
    data = df, family = binomial(link = "logit")
)

sink(file.path(output_dir, "modelo_resumen.txt"))
cat("=================================================\n")
cat("  REGRESIÓN LOGÍSTICA - MODELO 1\n")
cat("=================================================\n\n")
print(summary(modelo))
cat("\n\n=== Test de razón de verosimilitudes ===\n")
print(anova(modelo, test = "LRT"))
sink()

cat("[2/4] Extrayendo OR con IC 95%...\n")

resultados <- tidy(modelo, exponentiate = TRUE, conf.int = TRUE, conf.level = 0.95)

resultados <- resultados %>%
    filter(term != "(Intercept)") %>%
    rename(factor = term, OR = estimate, OR_low = conf.low, OR_high = conf.high, p = p.value) %>%
    mutate(
        factor_legible = case_when(
            factor == "Age"                              ~ "Edad (+1 año)",
            factor == "GenderM"                          ~ "Género: Hombre (ref: Mujer)",
            factor == "Family_HistoryYes"                ~ "Antecedentes familiares",
            factor == "Smoking_HistoryYes"               ~ "Tabaquismo",
            factor == "Alcohol_ConsumptionYes"           ~ "Alcohol regular",
            factor == "Obesity_BMIOverweight"            ~ "Sobrepeso (ref: Normal)",
            factor == "Obesity_BMIObese"                 ~ "Obesidad (ref: Normal)",
            factor == "Diet_RiskModerate"                ~ "Dieta moderada (ref: Low)",
            factor == "Diet_RiskHigh"                    ~ "Dieta alto riesgo (ref: Low)",
            factor == "Physical_ActivityModerate"        ~ "Actividad moderada (ref: Alta)",
            factor == "Physical_ActivityLow"             ~ "Sedentarismo (ref: Alta)",
            factor == "DiabetesYes"                      ~ "Diabetes",
            factor == "Inflammatory_Bowel_DiseaseYes"    ~ "EII",
            factor == "Genetic_MutationYes"              ~ "Mutación genética",
            TRUE ~ factor
        ),
        significativo = case_when(
            p < 0.001 ~ "***", p < 0.01 ~ "**", p < 0.05 ~ "*", TRUE ~ "n.s."
        ),
        direccion = case_when(
            OR > 1 & p < 0.05 ~ "Factor de riesgo",
            OR < 1 & p < 0.05 ~ "Factor protector",
            TRUE              ~ "No significativo"
        )
    ) %>%
    arrange(desc(OR))

readr::write_excel_csv(resultados, file.path(output_dir, "coeficientes_OR.csv"))

cat("[3/4] Forest plot...\n")

resultados_plot <- resultados %>%
    mutate(factor_legible = factor(factor_legible, levels = rev(factor_legible)))

forest <- ggplot(resultados_plot,
                 aes(x = OR, y = factor_legible, color = direccion)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "#78909C", linewidth = 0.5) +
    geom_errorbarh(aes(xmin = OR_low, xmax = OR_high), height = 0.25, linewidth = 0.9) +
    geom_point(aes(size = -log10(p + 1e-300)), alpha = 0.9) +
    geom_text(aes(label = sprintf("%.2f [%.2f-%.2f] %s",
                                   OR, OR_low, OR_high, significativo)),
              hjust = -0.15, vjust = -0.6, size = 3, color = "#37474F") +
    scale_x_continuous(trans = "log2",
                       breaks = c(0.5, 0.75, 1, 1.5, 2, 3, 5, 10)) +
    scale_color_manual(values = c(
        "Factor de riesgo" = "#E53935",
        "Factor protector" = "#43A047",
        "No significativo" = "#90A4AE"
    )) +
    scale_size_continuous(range = c(2, 6), guide = "none") +
    labs(
        title = "Forest plot: Odds Ratios del Modelo 1",
        subtitle = "Regresión logística multivariable. IC del 95%. Escala logarítmica.",
        x = "Odds Ratio (OR)", y = "",
        color = "Dirección del efecto",
        caption = "Significación: *** p<0.001  **p<0.01  *p<0.05  n.s.=no significativo"
    ) +
    theme_minimal(base_size = 11) +
    theme(
        plot.title = element_text(face = "bold", size = 14, color = "#0D47A1"),
        plot.subtitle = element_text(color = "#546E7A", size = 10),
        legend.position = "top",
        panel.grid.minor = element_blank(),
        plot.caption = element_text(size = 8, color = "#78909C", face = "italic"),
        axis.text.y = element_text(size = 10)
    )

ggsave(file.path(output_dir, "forest_plot_OR.png"),
       forest, width = 11, height = 7, dpi = 300, bg = "white")

cat("[4/4] Comparación OR vs RR literatura...\n")

rr_literatura <- tribble(
    ~factor_legible,                         ~RR_literatura, ~fuente,
    "Antecedentes familiares",               1.79,            "Butterworth 2006",
    "Tabaquismo",                            1.18,            "Johnson 2013",
    "Alcohol regular",                       1.21,            "Johnson 2013",
    "Sobrepeso (ref: Normal)",               1.10,            "Ma 2013",
    "Obesidad (ref: Normal)",                1.33,            "Ma 2013",
    "Dieta moderada (ref: Low)",             1.10,            "Johnson 2013",
    "Dieta alto riesgo (ref: Low)",          1.25,            "Johnson 2013",
    "Actividad moderada (ref: Alta)",        1.14,            "Johnson 2013",
    "Sedentarismo (ref: Alta)",              1.32,            "Johnson 2013",
    "Diabetes",                              1.27,            "Johnson 2013",
    "EII",                                   2.93,            "Johnson 2013",
    "Mutación genética",                     4.00,            "NCI CCRAT"
)

comparacion <- resultados %>%
    select(factor_legible, OR, OR_low, OR_high) %>%
    inner_join(rr_literatura, by = "factor_legible")

p_comp <- ggplot(comparacion, aes(x = reorder(factor_legible, OR))) +
    geom_segment(aes(xend = factor_legible, y = RR_literatura, yend = OR),
                 color = "#B0BEC5", linewidth = 0.5) +
    geom_point(aes(y = OR, color = "OR (nuestro modelo)"), size = 4) +
    geom_point(aes(y = RR_literatura, color = "RR (literatura)"), size = 4, shape = 17) +
    geom_errorbar(aes(ymin = OR_low, ymax = OR_high, color = "OR (nuestro modelo)"),
                  width = 0.2, alpha = 0.5) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "#78909C") +
    scale_color_manual(values = c(
        "OR (nuestro modelo)" = "#1976D2",
        "RR (literatura)" = "#E53935"
    )) +
    scale_y_continuous(trans = "log2", breaks = c(0.5, 1, 1.5, 2, 3, 5)) +
    coord_flip() +
    labs(
        title = "Concordancia entre OR del modelo y RR de literatura",
        subtitle = "Validación cruzada: coherencia entre datos internos y literatura publicada",
        x = "", y = "Magnitud del efecto (escala log)", color = "",
        caption = "Las líneas unen los valores del mismo factor."
    ) +
    theme_minimal(base_size = 11) +
    theme(
        plot.title = element_text(face = "bold", size = 13, color = "#0D47A1"),
        plot.subtitle = element_text(color = "#546E7A", size = 10),
        legend.position = "top",
        panel.grid.minor = element_blank(),
        plot.caption = element_text(size = 8, color = "#78909C", face = "italic")
    )

ggsave(file.path(output_dir, "comparacion_OR_vs_RR.png"),
       p_comp, width = 10, height = 6.5, dpi = 300, bg = "white")

comparacion_tabla <- comparacion %>%
    mutate(diferencia_pct = round((OR / RR_literatura - 1) * 100, 1)) %>%
    select(factor_legible, OR, OR_low, OR_high, RR_literatura, diferencia_pct, fuente)
readr::write_excel_csv(comparacion_tabla, file.path(output_dir, "comparacion_OR_vs_RR.csv"))

cat("\n=================================================\n")
cat("  REGRESIÓN LOGÍSTICA COMPLETADA\n")
cat("=================================================\n")
for (f in list.files(output_dir)) cat("  -", f, "\n")