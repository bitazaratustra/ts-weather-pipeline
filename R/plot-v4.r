# Install necessary packages if you don't have them
if (!require("data.table")) install.packages("data.table")
if (!require("ggplot2")) install.packages("ggplot2")

# Load the libraries
library(data.table)
library(ggplot2)

# --- 1. Data Loading and Manipulation with data.table ---

# Load the dataset efficiently
file_path <- "weather-air-quality-clean.csv.bz2"
if (!file.exists(file_path)) {
  stop("Error: The file 'weather-air-quality-merged.csv' was not found in the working directory.")
}
df <- fread(file_path)

# Convert the 'date' column to a proper date-time format
df[, date := as.POSIXct(date, format = "%Y-%m-%d %H:%M:%S")]

# Reshape the data from wide to long format
df_long <- melt(df, id.vars = "date", variable.name = "measurement", value.name = "value")

# Correctly separate 'measurement' into 'variable' and 'station'
df_long[, variable := sub("_.*", "", measurement)]
df_long[, station := sub("^[a-z0-9]*_", "", measurement)]

# --- CHANGE: The na.omit() line has been REMOVED ---
# By keeping rows where 'value' is NA, ggplot2 will create gaps in the lines.

# Define proper names and units for the y-axis labels
variable_labels <- c(
  "precipitation" = "Precipitation (mm)",
  "pressure" = "Pressure (hPa)",
  "relativehumidity" = "Relative Humidity (%)",
  "temperature" = "Temperature (°C)",
  "windangle" = "Wind Angle (°)",
  "windspeed" = "Wind Speed (km/h)",
  "co" = "CO (µg/m³)",
  "no2" = "NO2 (µg/m³)",
  "pm10" = "PM10 (µg/m³)"
)

# --- 2. Plotting with ggplot2 ---

# Get the unique variables to plot
unique_variables <- unique(df_long$variable)

# Open a PDF file to save the plots
pdf("timeseries_plots_final.pdf", width = 11, height = 8.5)

# Loop through each variable and create a ggplot
for (var in unique_variables) {
  
  # Filter data for the current variable
  plot_data <- df_long[variable == var]
  
  # Create the plot using ggplot2
  p <- ggplot(plot_data, aes(x = date, y = value, color = station)) +
    # geom_line() will automatically handle NAs by creating breaks
    geom_line(alpha = 0.8) +
    # --- CHANGE: The 'title' argument has been REMOVED ---
    labs(
      x = "Date",
      y = variable_labels[var],
      color = "Station"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      legend.position = "top"
    )
  
  # Print the plot to the PDF file
  print(p)
}

# Close the PDF file device
dev.off()

cat("Successfully generated 'timeseries_plots_final.pdf' with", length(unique_variables), "plots.\n")
