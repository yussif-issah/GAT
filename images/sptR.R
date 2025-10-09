library(ggplot2)
library(sf)
library(spdep)
library(gganimate)
library(gstat)
library(tidyr)
library(dplyr)
library(purrr)


data <- read.csv("results_masked3.csv")

data$geom <- gsub("'", '"', data$geom)
data$geom_sf <- st_as_sfc(data$geom, GeoJSON = TRUE)
data$wkt <- st_as_text(data$geom_sf)
data_sf <- st_as_sf(data, wkt = "wkt", crs = 32654)
data_sf <- st_transform(data_sf, crs = 32654) 

row.names(data_sf) <- paste(data_sf$grid_id, data_sf$Year, sep = "_")

neighbors <- poly2nb(data_sf)
weights <- nb2listw(neighbors, style = "W")
morans_test <- moran.test(data_sf$NDVI, listw = weights)
print(morans_test)

data_sf <- st_sf(data_sf, geom = data_sf$geom_sf)
st_crs(data_sf)


ggplot(data, aes(x = Year, y = NDVI, group = grid_id, color = as.factor(grid_id))) +
  geom_line() +
  theme_minimal() +
  labs(title = "NDVI Trends Over Time", x = "Year", y = "NDVI", color = "Grid ID")

data_sf <- data_sf[, !names(data_sf) %in% c("geometry", ".geo", "geom", "wkt",'geom_sf','system.index')]

ggplot(data_sf) +
  geom_sf(aes(fill = NDVI)) +
  theme_minimal()


ggplot(data_sf) +
  geom_sf(aes(fill = NDVI)) +
  transition_time(Year) +
  labs(title = "NDVI Over Time: {frame_time}", fill = "NDVI")





# plot code
animation <- ggplot(data_sf) +
  geom_sf(aes(fill = Mean_NDVI)) +
  transition_time(Year) +
  labs(title = "NDVI Over Time: {frame_time}", fill = "NDVI")

# Save or render the animation
animate(animation, renderer = gifski_renderer(), width = 800, height = 600)
anim_save("ndvi_animation.gif")


ggplot(data_sf %>% filter(Year == 2026)) +
  geom_sf(aes(fill = NDVI)) +
  scale_fill_viridis_c() +
  labs(title = "Time Series Transformer 2024 NDVI Prediction For Grids", fill = "NDVI")

model <- lm(Mean_NDVI ~ Year + factor(index), data = data)
summary(model)http://127.0.0.1:14107/graphics/plot_zoom_png?width=928&height=864



centroids <- st_centroid(data_sf)
data_sf$longitude <- st_coordinates(centroids)[, 1]
data_sf$latitude <- st_coordinates(centroids)[, 2]

write.csv(data_sf,"data_with_centroid.csv") 
grid <- st_make_grid(data_sf, n = c(8, 8)) # Adjust `n` for grid density
grid_sf <- st_sf(geometry = grid)

# Add unmeasured years to the grid
row.names(data_sf) <- paste(data_sf$index, data_sf$Year, sep = "_")
prediction_years <- seq(2025, 2026, by = 1)
prediction_grid <- expand.grid(Year = prediction_years, index = seq_along(grid_sf$geometry))
prediction_grid_sf <- st_sf(prediction_grid, geometry = grid_sf$geometry)







data_sf <- st_make_valid(data_sf)  # Fix invalid geometries if any

# Create neighbors
neighbors <- poly2nb(data_sf)

# Plot the polygons
plot(st_geometry(data_sf), col = "lightblue", border = "grey", main = "Neighbors Visualization")

# Add neighbors
plot.nb(neighbors, coords = st_centroid(st_geometry(data_sf)) |> st_coordinates(), 
        col = "red", lwd = 2, add = TRUE)

# Add legend
legend("topright", legend = c("Polygons", "Neighbors"), fill = c("lightblue", "red"), border = NA)

library(spdep)
library(sf)
library(CARBayesST)
library(spatialreg)


data_sf <- data_sf[order(data_sf$index, data_sf$Year), ]

# Create a space-time index
data_sf$SpaceTimeID <- interaction(data_sf$index, data_sf$Year)

formula <- Mean_NDVI ~ latitude + longitude + I(latitude^2) + I(longitude^2) + I(latitude * longitude)


neighbors <- make.sym.nb(neighbors)
W = nb2mat(neighbors, style = "W")
W <- (W + t(W)) / 2

data_sf <- st_as_sf(data_sf, coords = c("longitude", "latitude"), crs = 4326)
neighbors <- poly2nb(data_sf[data_sf$Year == unique(data_sf$Year)[1], ], queen = TRUE)
W <- nb2mat(neighbors, style = "W", zero.policy = TRUE)

# Ensure W is symmetric
W <- (W + t(W)) / 2
data_sf$time <- as.numeric(factor(data_sf$Year))

model <- tryCatch({
  ST.CARar(
    formula = formula,
    data = data_sf,
    family = "gaussian",
    W = W,
    burnin = 500,
    n.sample = 1000,
    thin = 5,
    AR=1
  )
}, error = function(e) {
  print("Error encountered")
  print(e)
})

response_residuals <- model$residuals$response
pearson_residuals <- model$residuals$pearson

# Add residuals to the dataset
data_sf$response_residuals <- response_residuals
data_sf$pearson_residuals <- pearson_residuals

plot(data_sf$response_residuals, main = "Response Residuals")
plot(data_sf$pearson_residuals, main = "Pearson Residuals")

#test for auto-correlation in residuals
morans_test <- moran.test(data_sf$response_residuals, listw = weights)
print(morans_test)


models <- data_sf %>%
  group_by(index) %>%
  summarize(
    model = list(lm(Mean_NDVI ~ Year, data = cur_data()))
  )

data_2025 <- data.frame(
  Year = 2025,
  index = unique(data_sf$index)
)



data_2025 <- data_2025 %>%
  left_join(models, by = "index") %>%
  mutate(
    Mean_NDVI_pred = map2_dbl(model, Year, ~ predict(.x, newdata = data.frame(Year = .y)))
  )

print(data_2025)


data_2025_sf <- merge(data_sf, data_2025, by = "index")

# Plot using ggplot2
ggplot(data_2025_sf, aes(x = longitude, y = latitude, fill = Mean_NDVI_pred)) +
  geom_tile() +
  scale_fill_viridis_c() +
  labs(title = "Predicted Mean_NDVI for 2025", fill = "Mean_NDVI") +
  theme_minimal()



# Load necessary libraries
library(dplyr)

# Extract fitted values for each year and grid ID
data_sf <- data_sf %>%
  mutate(Mean_NDVI_fitted = model$fitted.values)

# Calculate the average yearly change in Mean_NDVI for each grid ID
trend <- data_sf %>%
  group_by(index) %>%
  arrange(Year) %>%
  summarize(
    avg_change = mean(diff(Mean_NDVI_fitted))  # Average yearly difference
  )

# Filter data for the last available year (assume 2024 is the last year)
data_2024 <- data_sf %>% filter(Year == 2024)

# Predict Mean_NDVI for 2025 by adding the average yearly change
data_2025 <- data_2024 %>%
  st_join(trend, by = "index") %>%
  mutate(
    Year = 2025,
    Mean_NDVI_pred = Mean_NDVI_fitted + avg_change
  )

# View the predicted data for 2025
print(data_2025)


library(ggplot2)
library(sf)

# Plot Mean_NDVI predictions for 2025
ggplot(data_2025) +
  geom_sf(aes(fill = Mean_NDVI_pred), color = NA) +
  scale_fill_viridis_c() +  # Color scale for better visualization
  theme_minimal() +
  labs(
    title = "Predicted Mean NDVI for 2025",
    fill = "Mean NDVI"
  )


