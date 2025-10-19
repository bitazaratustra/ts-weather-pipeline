# Intervention Analysis of Air Pollution Data

# 1. SETUP
# -----------------------------------------------------------------------------

# Install and load necessary packages
if (!require("data.table")) install.packages("data.table")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("forecast")) install.packages("forecast")
if (!require("tseries")) install.packages("tseries")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("zoo")) install.packages("zoo")
if (!require("digest")) install.packages("digest") # For model caching
if (!require("grid")) install.packages("grid")     # For grid.draw

library(data.table)
library(ggplot2)
library(forecast)
library(tseries)
library(gridExtra)
library(zoo)
library(digest)
library(grid) # <-- ¡Esta es la corrección!

# Define file paths
data_file <- "weather-air-quality-clean.csv.bz2"
pdf_file <- "air_pollution_analysis_plots.pdf"
summary_file <- "air_pollution_analysis_summary.txt"
model_dir <- "fitted_models" # Directory to store cached model objects


# 2. DATA LOADING AND PREPROCESSING
# -----------------------------------------------------------------------------

# Create model directory if it doesn't exist
if (!dir.exists(model_dir)) {
    cat(paste("Creating model directory:", model_dir, "\n"))
    dir.create(model_dir)
}

# Load the dataset
cat("Loading data...\n")
dt <- fread(data_file)

# Convert date column to POSIXct
dt[, date := as.POSIXct(date, format = "%Y-%m-%d %H:%M:%S")]

# Create a complete hourly time sequence to ensure no gaps
full_time_seq <- seq(from = as.POSIXct("2009-10-01 00:00:00", tz = "UTC"),
                     to = as.POSIXct("2025-10-14 23:00:00", tz = "UTC"), by = "hour")
dt_full <- data.table(date = full_time_seq)

# Merge with the original data to fill in missing timestamps
dt <- merge(dt_full, dt, by = "date", all.x = TRUE)


# Consolidate data from multiple stations into single variables.
# We average station data and use openmeteo data as a fallback.
cat("Preprocessing data...\n")

dt[, temperature := rowMeans(.SD, na.rm = TRUE), .SDcols = c("temperature_aeroparque", "temperature_observatorio")]
dt[is.nan(temperature), temperature := temperature_openmeteo]

dt[, relativehumidity := rowMeans(.SD, na.rm = TRUE), .SDcols = c("relativehumidity_aeroparque", "relativehumidity_observatorio")]
dt[is.nan(relativehumidity), relativehumidity := relativehumidity_openmeteo]

dt[, pressure := rowMeans(.SD, na.rm = TRUE), .SDcols = c("pressure_aeroparque", "pressure_observatorio")]
dt[is.nan(pressure), pressure := pressure_openmeteo]

dt[, windspeed := rowMeans(.SD, na.rm = TRUE), .SDcols = c("windspeed_aeroparque", "windspeed_observatorio")]
dt[is.nan(windspeed), windspeed := windspeed_openmeteo]

dt[, windangle := rowMeans(.SD, na.rm = TRUE), .SDcols = c("windangle_aeroparque", "windangle_observatorio")]
dt[is.nan(windangle), windangle := windangle_openmeteo]

# Precipitation is only from openmeteo
dt[, precipitation := precipitation_openmeteo]

dt[, pm10 := rowMeans(.SD, na.rm = TRUE), .SDcols = c("pm10_centenario", "pm10_cordoba", "pm10_la_boca")]
dt[is.nan(pm10), pm10 := pm10_openmeteo]

dt[, no2 := rowMeans(.SD, na.rm = TRUE), .SDcols = c("no2_centenario", "no2_cordoba", "no2_la_boca")]
dt[is.nan(no2), no2 := no2_openmeteo]

dt[, co := rowMeans(.SD, na.rm = TRUE), .SDcols = c("co_centenario", "co_cordoba", "co_la_boca", "co_palermo")]
dt[is.nan(co), co := co_openmeteo]

# Select final columns for analysis
final_cols <- c("date", "temperature", "relativehumidity", "pressure",
                "windspeed", "windangle", "precipitation", "co", "no2", "pm10")
dt_final <- dt[, ..final_cols]

# Interpolate remaining missing values (using linear interpolation)
for (col in names(dt_final)) {
    if (any(is.na(dt_final[[col]]))) {
        dt_final[, (col) := na.approx(get(col), na.rm = FALSE)]
    }
}

# Remove rows with NAs (likely at the very start of the series)
dt_final <- na.omit(dt_final)


# 3. EXPLORATORY DATA ANALYSIS (EDA)
# -----------------------------------------------------------------------------
cat("Performing EDA and generating plots...\n")

# Define the intervention period (e.g., COVID-19 lockdown)
intervention_start <- as.POSIXct("2020-03-20 00:00:00")
intervention_end <- as.POSIXct("2020-05-10 23:00:00")

# Create a list to store plots
plot_list <- list()

# Plot time series for each pollutant
pollutants <- c("co", "no2", "pm10")

for (pollutant in pollutants) {
    p <- ggplot(dt_final, aes(x = date, y = get(pollutant))) +
        geom_line(alpha = 0.8) +
        geom_rect(aes(xmin = intervention_start, xmax = intervention_end,
                      ymin = -Inf, ymax = Inf),
                  fill = "red", alpha = 0.3) +
        labs(title = paste("Time Series of", toupper(pollutant)),
             x = "Date", y = toupper(pollutant)) +
        theme_minimal()
    plot_list[[pollutant]] <- p
}

# Open PDF device to save plots
pdf(pdf_file, width = 12, height = 6)
for (p in plot_list) {
    print(p)
}


# 4. INTERVENTION ANALYSIS
# -----------------------------------------------------------------------------

# Subset data to a smaller time range for faster analysis
cat("Subsetting data for analysis period...\n")
analysis_start <- as.POSIXct("2018-01-01 00:00:00")
analysis_end <- as.POSIXct("2022-12-31 23:00:00")
dt_analysis <- dt_final[date >= analysis_start & date <= analysis_end]

cat("Starting intervention analysis...\n")

# Create binary intervention variable (1 during the period, 0 otherwise)
dt_analysis[, intervention := 0]
dt_analysis[date >= intervention_start & date <= intervention_end, intervention := 1]

# Open connection to summary file to log results
sink(summary_file)

# --- Define auto.arima execution parameters (NOT HASHED) ---
# These control *how* the model is fit, not *what* model is fit
arima_exec_params <- list(
    trace = TRUE,
    parallel = TRUE,
    num.cores = 8
)


for (pollutant in pollutants) {
    cat(paste("\n\n\n========================================================\n"))
    cat(paste("         ANALYSIS FOR", toupper(pollutant), "\n"))
    cat(paste("========================================================\n"))

    # Create time series object (frequency=24 for daily seasonality)
    y <- ts(dt_analysis[[pollutant]], frequency = 24)

	# Define exogenous regressors
	xreg <- as.matrix(dt_analysis[, .(intervention, temperature, relativehumidity,
		                       pressure, windspeed, windangle, precipitation)])

    # --- START: Model Caching Logic ---
    
    # --- Define parameters that identify the model (HASHED) ---
    model_params <- list(
        pollutant = pollutant,
        analysis_start = format(analysis_start, "%Y%m%d"),
        analysis_end = format(analysis_end, "%Y%m%d"),
        intervention_start = format(intervention_start, "%Y%m%d"),
        intervention_end = format(intervention_end, "%Y%m%d"),
        ts_frequency = frequency(y),
        xreg_vars = colnames(xreg),
        # These are the auto.arima settings that define the final model
        auto_arima_settings = list(
            seasonal = TRUE,
            allowdrift = TRUE,
            allowmean = TRUE,
            stepwise = FALSE,
            approximation = TRUE
        )
    )

    # Create a unique hash and filename
    model_hash <- digest(model_params, algo = "md5")
    model_filename <- paste0("model_", pollutant, "_", model_hash, ".rds")
    model_filepath <- file.path(model_dir, model_filename)

    # Check if a cached model file exists
    if (file.exists(model_filepath)) {
        
        cat(paste("\nLoading cached model for", toupper(pollutant), "from:", model_filepath, "\n"))
        fit <- readRDS(model_filepath)
        
    } else {
        
        # Fit ARIMAX model using auto.arima
        cat(paste("\nFitting ARIMAX model for", toupper(pollutant), "...\n"))
        cat(paste("No cache found. Computing and saving to:", model_filepath, "\n"))
        
        # Prepare arguments for auto.arima
        # Combine the core data, the model settings (hashed), and execution settings (not hashed)
        all_args <- c(
            list(y = y, xreg = xreg), 
            model_params$auto_arima_settings, # Hashed model params
            arima_exec_params               # Non-hashed execution params
        )
        
        # Call auto.arima using do.call to pass the list of arguments
        fit <- do.call(auto.arima, all_args)

        # Save the computed model to the cache
        cat(paste("\nSaving model to", model_filepath, "...\n"))
        saveRDS(fit, file = model_filepath)
    }
    
    # --- END: Model Caching Logic ---


    # Print model summary
    cat(paste("\n--- Model Summary for", toupper(pollutant), "---\n"))
    print(summary(fit))

    # Check significance of the intervention variable
    cat(paste("\n--- Intervention Effect for", toupper(pollutant), "---\n"))
    
    # Robustly check for the intervention coefficient and its standard error
    if ("intervention" %in% names(coef(fit)) & !is.na(coef(fit)["intervention"])) {
        intervention_coef <- coef(fit)["intervention"]
        intervention_se <- sqrt(diag(vcov(fit)))["intervention"]
        
        if (!is.na(intervention_se) && intervention_se > 0) {
            intervention_t <- intervention_coef / intervention_se
            intervention_p <- 2 * pt(abs(intervention_t), df = length(y) - length(coef(fit)), lower.tail = FALSE)
            
            cat(paste("Coefficient for intervention:", intervention_coef, "\n"))
            cat(paste("Standard Error:", intervention_se, "\n"))
            cat(paste("T-value:", intervention_t, "\n"))
            cat(paste("P-value:", intervention_p, "\n"))
            
            if (intervention_p < 0.05 & intervention_coef < 0) {
                cat("Result: Significant decrease detected.\n")
            } else if (intervention_p < 0.05 & intervention_coef > 0) {
                cat("Result: Significant increase detected.\n")
            } else {
                cat("Result: No significant effect detected (p >= 0.05).\n")
            }
        } else {
            cat("Could not calculate p-value (Standard Error is NA or zero).\n")
        }
    } else {
        cat("Intervention coefficient is not available in the model (e.g., removed due to collinearity).\n")
    }


    # 5. ASSUMPTION TESTING
    # -------------------------------------------------------------------------
    cat(paste("\n--- Assumption Testing for", toupper(pollutant), "Model ---\n"))

    # Ljung-Box test for autocorrelation in residuals
    lb_test <- Box.test(residuals(fit), lag = 24, type = "Ljung-Box")
    cat("\nLjung-Box Test for Residuals:\n")
    print(lb_test)

    # Plot ACF/PACF for residuals
    p_acf <- ggAcf(residuals(fit)) + labs(title = paste("ACF of Residuals for", toupper(pollutant)))
    p_pacf <- ggPacf(residuals(fit)) + labs(title = paste("PACF of Residuals for", toupper(pollutant)))
    
    # Q-Q plot for normality of residuals
    qq_data <- data.table(residuals = as.vector(residuals(fit)))
    p_qq <- ggplot(qq_data, aes(sample = residuals)) +
            stat_qq() + stat_qq_line() +
            labs(title = paste("Q-Q Plot of Residuals for", toupper(pollutant))) +
            theme_minimal()


    # Arrange diagnostic plots in a grid
    grid_plot <- grid.arrange(p_acf, p_pacf, p_qq, ncol = 2,
                              top = paste("Residual Diagnostics for", toupper(pollutant)))
    plot_list[[paste0(pollutant, "_residuals")]] <- grid_plot
}

# Save residual plots to the same PDF
for (i in (length(pollutants) + 1):length(plot_list)) {
    grid.draw(plot_list[[i]])
}

# Close PDF and text file connections
dev.off()
sink()

cat("\nAnalysis complete.\n")
cat(paste("Plots saved to:", pdf_file, "\n"))
cat(paste("Summary saved to:", summary_file, "\n"))
cat(paste("Models saved in:", model_dir, "\n"))
