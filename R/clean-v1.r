# Install data.table if you don't have it already
# install.packages("data.table")

# Load the data.table library
library(data.table)

# Define file paths
input_file <- "weather-air-quality-merged.csv.bz2"
output_file <- "weather-air-quality-clean.csv"

# Read the compressed CSV file into a data.table
# fread is highly efficient and can read compressed files directly
dt <- fread(input_file)

# --- Data Reporting and Cleaning ---

cat("--- Reporting and Cleaning Data ---\n\n")

## 1. Clean Relative Humidity (>= 100)
humidity_cols <- c("relativehumidity_aeroparque", "relativehumidity_observatorio", "relativehumidity_openmeteo")
for (col in humidity_cols) {
  # Report values to be changed
  invalid_humidity <- dt[get(col) >= 100]
  if (nrow(invalid_humidity) > 0) {
    cat(sprintf("Found %d values in '%s' >= 100. Changing to NA.\n", nrow(invalid_humidity), col))
    print(invalid_humidity[, .SD, .SDcols = c("date", col)])
    cat("\n")
    
    # Change values to NA
    dt[get(col) >= 100, (col) := NA_real_]
  }
}

## 2. Clean Temperature (> 60)
temp_cols <- c("temperature_aeroparque", "temperature_observatorio", "temperature_openmeteo")
for (col in temp_cols) {
  # Report values to be changed
  invalid_temp <- dt[get(col) > 60]
  if (nrow(invalid_temp) > 0) {
    cat(sprintf("Found %d values in '%s' > 60. Changing to NA.\n", nrow(invalid_temp), col))
    print(invalid_temp[, .SD, .SDcols = c("date", col)])
    cat("\n")
    
    # Change values to NA
    dt[get(col) > 60, (col) := NA_real_]
  }
}

## 3. Clean Pressure (< 900)
pressure_cols <- c("pressure_aeroparque", "pressure_observatorio", "pressure_openmeteo")
for (col in pressure_cols) {
  # Report values to be changed
  invalid_pressure <- dt[get(col) < 900 & !is.na(get(col))]
  if (nrow(invalid_pressure) > 0) {
    cat(sprintf("Found %d values in '%s' < 900. Changing to NA.\n", nrow(invalid_pressure), col))
    print(invalid_pressure[, .SD, .SDcols = c("date", col)])
    cat("\n")
    
    # Change values to NA
    dt[get(col) < 900, (col) := NA_real_]
  }
}


# --- Data Transformation ---

cat("--- Transforming Wind Angle Data ---\n\n")

## 4. Transform windangle_aeroparque
# Formula: (value / 1000) * 360
dt[, windangle_aeroparque := (windangle_aeroparque / 1000) * 360]
cat("Transformed 'windangle_aeroparque' column.\n")

## 5. Conditionally transform windangle_observatorio
# Formula: (value / 1000) * 360 for values > 360
dt[windangle_observatorio > 360, windangle_observatorio := (windangle_observatorio / 1000) * 360]
cat("Conditionally transformed 'windangle_observatorio' for values > 360.\n\n")


# --- Save Cleaned Data ---

# Save the final data.table to a new compressed file
fwrite(dt, output_file)

cat(sprintf("Data cleaning complete. The cleaned data has been saved to '%s'.\n", output_file))
