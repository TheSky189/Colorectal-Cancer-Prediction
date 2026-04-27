# Análisis exploratorio de datos (EDA) del dataset clínico.

# Forzar UTF-8 para que los acentos se vean bien en Windows
Sys.setlocale("LC_ALL", "Spanish_Spain.utf8")
options(encoding = "UTF-8")

library(tidyverse)
library(ggplot2)
library(ggcorrplot)
library(scales)


# Resolución robusta del root del proyecto
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
    stop("No se ha podido localizar la raíz del proyecto (buscando carpeta 'data/').")
}

root_dir <- find_project_root()
training_path <- file.path(root_dir, "data", "processed", "training_dataset.csv")
output_dir <- file.path(root_dir, "reports", "eda")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

cat("Raíz del proyecto:", root_dir, "\n")
cat("Outputs:", output_dir, "\n\n")


# Cargar datos
df <- read_csv(training_path, show_col_types = FALSE)
df$Cancer <- factor(df$Cancer, levels = c(0, 1), labels = c("Control", "Cáncer"))

cat("Dimensiones:", dim(df)[1], "filas x", dim(df)[2], "columnas\n")


# Paleta y tema
colores_cohorte <- c("Control" = "#43A047", "Cáncer" = "#E53935")
tema_medico <- theme_minimal(base_size = 12) +
    theme(
        plot.title = element_text(face = "bold", size = 14, color = "#0D47A1"),
        plot.subtitle = element_text(color = "#546E7A", size = 10),
        axis.title = element_text(face = "bold"),
        legend.position = "top",
        panel.grid.minor = element_blank(),
        plot.caption = element_text(size = 8, color = "#78909C", face = "italic"),
        text = element_text(family = "sans")
    )


# 1. Distribución de edad por cohorte
cat("[1/5] Distribución de edad por cohorte...\n")

p_edad <- ggplot(df, aes(x = Age, fill = Cancer, color = Cancer)) +
    geom_density(alpha = 0.35, linewidth = 0.8) +
    scale_fill_manual(values = colores_cohorte) +
    scale_color_manual(values = colores_cohorte) +
    scale_x_continuous(breaks = seq(20, 100, 10)) +
    labs(
        title = "Distribución de edad por cohorte",
        subtitle = "Comparación entre casos (cáncer) y controles sintéticos",
        x = "Edad (años)", y = "Densidad",
        fill = "Cohorte", color = "Cohorte",
        caption = "Fuente: Colorectal Cancer Risk Factors (Kaggle) + controles sintéticos"
    ) +
    tema_medico

ggsave(file.path(output_dir, "distribucion_edad.png"),
       p_edad, width = 9, height = 5.5, dpi = 300, bg = "white")


# 2. Boxplot edad por cohorte
cat("[2/5] Boxplot edad por cohorte...\n")

p_boxplot <- ggplot(df, aes(x = Cancer, y = Age, fill = Cancer)) +
    geom_violin(alpha = 0.5, width = 0.8) +
    geom_boxplot(width = 0.15, fill = "white", outlier.color = "#E53935",
                 outlier.alpha = 0.3) +
    scale_fill_manual(values = colores_cohorte) +
    labs(
        title = "Distribución de edad: casos vs controles",
        subtitle = "Violin plot con boxplot superpuesto",
        x = "", y = "Edad (años)", fill = "Cohorte"
    ) +
    tema_medico + theme(legend.position = "none")

ggsave(file.path(output_dir, "boxplot_edad_cancer.png"),
       p_boxplot, width = 7, height = 5.5, dpi = 300, bg = "white")


# 3. Frecuencias de factores de riesgo
cat("[3/5] Factores de riesgo por cohorte...\n")

variables_binarias <- c(
    "Family_History", "Smoking_History", "Alcohol_Consumption",
    "Diabetes", "Inflammatory_Bowel_Disease", "Genetic_Mutation"
)

df_long <- df %>%
    select(Cancer, all_of(variables_binarias)) %>%
    pivot_longer(cols = -Cancer, names_to = "Factor", values_to = "Valor") %>%
    filter(Valor == "Yes") %>%
    count(Factor, Cancer) %>%
    group_by(Cancer) %>%
    mutate(pct = n / sum(n) * 100) %>%
    ungroup()

nombres_legibles <- c(
    "Family_History" = "Antecedentes familiares",
    "Smoking_History" = "Tabaquismo",
    "Alcohol_Consumption" = "Alcohol",
    "Diabetes" = "Diabetes",
    "Inflammatory_Bowel_Disease" = "EII",
    "Genetic_Mutation" = "Mutación genética"
)
df_long$Factor <- nombres_legibles[df_long$Factor]

p_factores <- ggplot(df_long,
                     aes(x = reorder(Factor, pct), y = pct, fill = Cancer)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_text(aes(label = paste0(round(pct, 1), "%")),
              position = position_dodge(width = 0.8),
              hjust = -0.15, size = 3.2, color = "#455A64") +
    scale_fill_manual(values = colores_cohorte) +
    scale_y_continuous(labels = function(x) paste0(x, "%"),
                       expand = expansion(mult = c(0, 0.15))) +
    coord_flip() +
    labs(
        title = "Prevalencia de factores de riesgo",
        subtitle = "Porcentaje de 'Sí' dentro de cada cohorte",
        x = "", y = "Prevalencia (%)", fill = "Cohorte"
    ) +
    tema_medico

ggsave(file.path(output_dir, "factores_riesgo_frecuencias.png"),
       p_factores, width = 9, height = 6, dpi = 300, bg = "white")


# 4. Correlograma
cat("[4/5] Correlograma...\n")

df_num <- df %>%
    mutate(
        Cancer_num = as.numeric(Cancer) - 1,
        Obesity_num = case_when(Obesity_BMI == "Normal" ~ 1,
                                 Obesity_BMI == "Overweight" ~ 2,
                                 Obesity_BMI == "Obese" ~ 3),
        Diet_num = case_when(Diet_Risk == "Low" ~ 1,
                              Diet_Risk == "Moderate" ~ 2,
                              Diet_Risk == "High" ~ 3),
        Activity_num = case_when(Physical_Activity == "Low" ~ 1,
                                  Physical_Activity == "Moderate" ~ 2,
                                  Physical_Activity == "High" ~ 3),
        FamHist_num = ifelse(Family_History == "Yes", 1, 0),
        Smoke_num = ifelse(Smoking_History == "Yes", 1, 0),
        Alcohol_num = ifelse(Alcohol_Consumption == "Yes", 1, 0),
        Diabetes_num = ifelse(Diabetes == "Yes", 1, 0),
        IBD_num = ifelse(Inflammatory_Bowel_Disease == "Yes", 1, 0),
        Genetic_num = ifelse(Genetic_Mutation == "Yes", 1, 0)
    ) %>%
    select(
        Cáncer = Cancer_num, Edad = Age,
        `Fam. hist.` = FamHist_num, Tabaco = Smoke_num,
        Alcohol = Alcohol_num, Obesidad = Obesity_num,
        Dieta = Diet_num, Actividad = Activity_num,
        Diabetes = Diabetes_num, EII = IBD_num, Mutación = Genetic_num
    )

matriz_cor <- round(cor(df_num, use = "complete.obs"), 2)

p_cor <- ggcorrplot(
    matriz_cor, hc.order = FALSE, type = "lower", lab = TRUE, lab_size = 3,
    method = "circle",
    colors = c("#E53935", "#FFFFFF", "#1976D2"),
    outline.color = "white",
    title = "Correlograma de variables del Modelo 1",
    ggtheme = theme_minimal(base_size = 11)
) +
    theme(
        plot.title = element_text(face = "bold", color = "#0D47A1", size = 13),
        axis.text.x = element_text(angle = 45, hjust = 1)
    )

ggsave(file.path(output_dir, "correlograma.png"),
       p_cor, width = 9, height = 8, dpi = 300, bg = "white")


# 5. Tabla descriptiva
cat("[5/5] Tabla descriptiva...\n")

tabla_num <- df %>%
    group_by(Cancer) %>%
    summarise(
        n = n(),
        edad_media = round(mean(Age), 1),
        edad_sd = round(sd(Age), 1),
        edad_min = min(Age),
        edad_max = max(Age),
        pct_hombre = round(mean(Gender == "M") * 100, 1),
        pct_family_hist = round(mean(Family_History == "Yes") * 100, 1),
        pct_fumador = round(mean(Smoking_History == "Yes") * 100, 1),
        pct_alcohol = round(mean(Alcohol_Consumption == "Yes") * 100, 1),
        pct_obeso = round(mean(Obesity_BMI == "Obese") * 100, 1),
        pct_diabetes = round(mean(Diabetes == "Yes") * 100, 1),
        pct_eii = round(mean(Inflammatory_Bowel_Disease == "Yes") * 100, 1),
        pct_mutacion = round(mean(Genetic_Mutation == "Yes") * 100, 1)
    )

# Guardar con BOM UTF-8 para que Excel lo abra bien en Windows
readr::write_excel_csv(tabla_num, file.path(output_dir, "tabla_descriptiva.csv"))
cat("\nTabla descriptiva:\n")
print(tabla_num, width = Inf)

cat("\n=================================================\n")
cat("  EDA COMPLETADO\n")
cat("=================================================\n")
cat("Salidas en:", output_dir, "\n")
for (f in list.files(output_dir)) cat("  -", f, "\n")