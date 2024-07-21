# Load data, set working directory to input folder (ADJUST AS NECESSARY)
setwd("~/Desktop/src/Input")
ftse <- read.csv("HistoricalPrices(3).csv")
dax <- read.csv("HistoricalPrices(4).csv")
euro <- read.csv("HistoricalPrices(5).csv")
sp <- read.csv("HistoricalPrices(6).csv")
cac <- read.csv("HistoricalPrices(7).csv")
nikkei <- read.csv("HistoricalPrices(8).csv")
bse <- read.csv("HistoricalPrices(9).csv")
bovesda <- read.csv("HistoricalPrices(10).csv")
kospi <- read.csv("HistoricalPrices(11).csv")
dowjones <- read.csv("HistoricalPrices(14).csv")
nasdaq <- read.csv("HistoricalPrices(15).csv")
worldbank <- read.csv("World_Bank.csv")
worldbank <- worldbank[-c(10:12, 19:21, 31:35), ]

# Country indicators
ftse$Country <- "GBR"
dax$Country <- "DEU"
sp$Country <- "USA"
dowjones$Country <- "USA"
nasdaq$Country <- "USA"
cac$Country <- "FRA"
nikkei$Country <- "JPN"
bse$Country <- "IND"
bovesda$Country <- "BRA"
kospi$Country <- "KOR"

#Country Indices
ftse$Index <- "FTSE"
dax$Index <- "DAX"
sp$Index <- "SP"
dowjones$Index <- "DJ"
nasdaq$Index <- "NASDAQ"
cac$Index <- "CAC"
nikkei$Index <- "NIKKEI"
bse$Index <- "BSE"
bovesda$Index <- "BOVESDA"
kospi$Index <- "KOSPI"

#Event window
n <- 120 #days (120 days / 4 months on both sides of DST implementation)

# Function to clean and convert date
convert_date <- function(date_str) {
  date_parts <- strsplit(date_str, "/")[[1]]
  
  year <- as.integer(date_parts[3])
  if (year > 50) {
    year <- year + 1900
  } else {
    year <- year + 2000
  }
  
  new_date_str <- paste(date_parts[1], date_parts[2], year, sep = "/")
  
  return(as.Date(new_date_str, format = "%m/%d/%Y"))
}

# Function to determine the last sunday of a given month in a given year
last_sunday <- function(year, month) {
  dates <- seq(as.Date(paste(year, month, "01", sep = "-"), "%Y-%m-%d"),
               as.Date(paste(year, month + 1, "01", sep = "-"), "%Y-%m-%d") - 1,
               "days")
  
  sundays <- dates[weekdays(dates) == "Sunday"]
  return(max(sundays))
}

# Function to determine the nth sunday of a given month in a given year
nth_sunday <- function(year, month, nth = 1) {
  dates <- seq(as.Date(paste(year, month, "01", sep = "-"), "%Y-%m-%d"),
               as.Date(paste(year, ifelse(month == 12, 1, month + 1), "01", sep = "-"), "%Y-%m-%d") - 1,
               "days")
  
  sundays <- dates[weekdays(dates) == "Sunday"]
  
  return(sundays[nth])
}

# Function to determine if a given date faills within DST bounds according to
# EU rules
is_dst_europe <- function(date) {
  year <- as.integer(format(date, "%Y"))
  dst_start <- last_sunday(year, 3)  # Last Sunday in March
  dst_end <- last_sunday(year, 10)   # Last Sunday in October
  
  if (date >= dst_start & date <= dst_end) {
    return(1)
  } else {
    return(0)
  }
}

# Function to determine if a given date faills within DST bounds according to
# USA rules
is_dst_usa <- function(date) {
  year <- as.integer(format(date, "%Y"))
  dst_start <- nth_sunday(year, 3, 2)  # Second Sunday in March
  dst_end <- nth_sunday(year, 11, 1)   # First Sunday in November
  
  if (date >= dst_start & date <= dst_end) {
    return(1)
  } else {
    return(0)
  }
}

# Converting Date to 'Date' format
ftse$Date <- do.call("c", lapply(ftse$Date, convert_date))
dax$Date <- do.call("c", lapply(dax$Date, convert_date))
sp$Date <- do.call("c", lapply(sp$Date, convert_date))
dowjones$Date <- do.call("c", lapply(dowjones$Date, convert_date))
nasdaq$Date <- do.call("c", lapply(nasdaq$Date, convert_date))
cac$Date <- do.call("c", lapply(cac$Date, convert_date))
nikkei$Date <- do.call("c", lapply(nikkei$Date, convert_date))
bse$Date <- do.call("c", lapply(bse$Date, convert_date))
bovesda$Date <- do.call("c", lapply(bovesda$Date, convert_date))
kospi$Date <- do.call("c", lapply(kospi$Date, convert_date))

# Adding year column to each dataset
ftse$Year <- as.numeric(format(ftse$Date, "%Y"))
dax$Year <- as.numeric(format(dax$Date, "%Y"))
sp$Year <- as.numeric(format(sp$Date, "%Y"))
dowjones$Year <- as.numeric(format(dowjones$Date, "%Y"))
nasdaq$Year <- as.numeric(format(nasdaq$Date, "%Y"))
cac$Year <- as.numeric(format(cac$Date, "%Y"))
nikkei$Year <- as.numeric(format(nikkei$Date, "%Y"))
bse$Year <- as.numeric(format(bse$Date, "%Y"))
bovesda$Year <- as.numeric(format(bovesda$Date, "%Y"))
kospi$Year <- as.numeric(format(kospi$Date, "%Y"))

# Adding macroeconomic data to indiidual datasets
names_vector <- as.character(1990:2022) 
current_names <- colnames(worldbank)
current_names[5:length(current_names)] <- names_vector
colnames(worldbank) <- current_names

# Impute mean where data unavailable
worldbank <- data.frame(t(apply(worldbank, 1, function(x) {
  x[x == ".."] <- mean(as.numeric(x[5:27]), na.rm = TRUE)
  return(x)
})))
worldbank <- as.data.frame(worldbank)

# Merging macroeconomic data at the country level with country datasets
list_of_dfs <- list(ftse, dax, sp, dowjones, nasdaq, cac, nikkei, bse, bovesda, kospi)
Countries <- c("GBR", "DEU", "USA", "FRA", "JPN", "IND", "BRA", "KOR") 
num_indices <- c(1,1,3,1,1,1,1,1)
j <- 1
k <- 0
l <- 1
for (ind in Countries) {
  for (i in 1:num_indices[l]) {
    subset_data <- subset(worldbank, Country.Code == ind)
    subset_data <- subset_data[,5:ncol(subset_data)]
    df_controls <- as.data.frame(t(subset_data))
    df_controls <- cbind(Year = c(1990:2022), df_controls)
    rownames(df_controls) <- 1:nrow(df_controls)
    colnames(df_controls)[2:4] <- c("Inflation", "Unemployment", "GDP")
    list_of_dfs[[j+i-1]] <- merge(list_of_dfs[[j+i-1]], df_controls, by = 'Year', all.x = TRUE)
    k <- k + 1
  }
  j <- j + k
  l <- l + 1
  k <- 0
}

ftse <- list_of_dfs[[1]]
dax <- list_of_dfs[[2]]
sp <- list_of_dfs[[3]]
dowjones <- list_of_dfs[[4]]
nasdaq <- list_of_dfs[[5]]
cac <- list_of_dfs[[6]]
nikkei <- list_of_dfs[[7]]
bse <- list_of_dfs[[8]]
bovesda <- list_of_dfs[[9]]
kospi <- list_of_dfs[[10]]

# Create DST indicator
ftse$DST <- sapply(ftse$Date, is_dst_europe)
cac$DST <- sapply(cac$Date, is_dst_europe)
dax$DST <- sapply(dax$Date, is_dst_europe)
sp$DST <- sapply(sp$Date, is_dst_usa)
dowjones$DST <- sapply(dowjones$Date, is_dst_usa)
nasdaq$DST <- sapply(nasdaq$Date, is_dst_usa)
nikkei$DST <- 0
bse$DST <- 0
kospi$DST <- 0
bovesda$DST <- 0

# Capping data at 2022
ftse <- ftse[as.numeric(format(ftse$Date, "%Y")) <= 2022, ]
cac <- cac[as.numeric(format(cac$Date, "%Y")) <= 2022, ]
dax <- dax[as.numeric(format(dax$Date, "%Y")) <= 2022, ]
sp <- sp[as.numeric(format(sp$Date, "%Y")) <= 2022, ]
dowjones <- dowjones[as.numeric(format(dowjones$Date, "%Y")) <= 2022, ]
nasdaq <- nasdaq[as.numeric(format(nasdaq$Date, "%Y")) <= 2022, ]
nikkei <- nikkei[as.numeric(format(nikkei$Date, "%Y")) <= 2022, ]
bse <- bse[as.numeric(format(bse$Date, "%Y")) <= 2022, ]
kospi <- kospi[as.numeric(format(kospi$Date, "%Y")) <= 2022, ]
bovesda <- bovesda[as.numeric(format(bovesda$Date, "%Y")) <= 2022, ]

# Apply event window to each dataset of DST-observing countries individually:
# remove observations outside of event window, centered around initial DST flip, for each year
ftse <- ftse[order(ftse$Date),]
ftse <- mutate(ftse, DST_change = DST - lag(DST, default = 0))
result <- data.frame()
for (i in which(ftse$DST_change == -1)) {
  subset <- ftse[max(1, i - n):min(nrow(ftse), i + n),]
  result <- rbind(result, subset)
}
ftse <- result

sp <- sp[order(sp$Date),]
sp <- sp %>%
  mutate(DST_change = DST - lag(DST))
result <- data.frame()
for (i in which(sp$DST_change ==-1)) {
  subset <- sp[max(1, i - n):min(nrow(sp), i + n),]
  result <- rbind(result, subset)
}
sp <- result

dax <- dax[order(dax$Date),]
dax <- dax %>%
  mutate(DST_change = DST - lag(DST))
result <- data.frame()
for (i in which(dax$DST_change ==-1)) {
  subset <- dax[max(1, i - n):min(nrow(dax), i + n),]
  result <- rbind(result, subset)
}
dax <- result

cac <- cac[order(cac$Date),]
cac <- cac %>%
  mutate(DST_change = DST - lag(DST))
result <- data.frame()
for (i in which(cac$DST_change ==-1)) {
  subset <- cac[max(1, i - n):min(nrow(cac), i + n),]
  result <- rbind(result, subset)
}
cac <- result

dowjones <- dowjones[order(dowjones$Date),]
dowjones <- dowjones %>%
  mutate(DST_change = DST - lag(DST))
result <- data.frame()
for (i in which(cac$DST_change ==-1)) {
  subset <- dowjones[max(1, i - n):min(nrow(dowjones), i + n),]
  result <- rbind(result, subset)
}
dowjones <- result

nasdaq <- nasdaq[order(nasdaq$Date),]
nasdaq <- nasdaq %>%
  mutate(DST_change = DST - lag(DST))
result <- data.frame()
for (i in which(cac$DST_change ==-1)) {
  subset <- nasdaq[max(1, i - n):min(nrow(nasdaq), i + n),]
  result <- rbind(result, subset)
}
nasdaq <- result

nikkei$DST_change <- 0
bse$DST_change <- 0
kospi$DST_change <- 0
bovesda$DST_change <- 0

#Concatenate datasets
df <- do.call("rbind", list(ftse, dax, sp, dowjones, nasdaq, cac, nikkei, bse, kospi, bovesda))

#Compute measures of returns
df$Returns <- (df$Close - df$Open) / df$Open
df$ReturnsLag <- (df$Close - lag(df$Close)) / lag(df$Close)

#Create new variables, necessary for proper indexing at the year level for CSA
df$Month <- as.numeric(format(df$Date, "%m"))
df$Country_numeric <- as.numeric(factor(df$Country))

# Subset data by year to accomodate for CSA assumption of irreversibility of treatment
dfs_by_year <- split(df, df$Year)

#To accomodate CSA estimator requirements, reformulate DST as cumulative sum such that
#'1' represents the first period of treatment, incrementing for all subsequent periods
for (i in 1:length(unique(df$Year))) {
  df_temp <- dfs_by_year[[i]]
  df_temp$DST <- ave(df_temp$DST, df_temp$Index, FUN = cumsum)
  dfs_by_year[[i]] <- df_temp
}

# Generate group-time ATTs for each year
att_by_year <- list()
for (i in 1:length(unique(df$Year))) {
  att_by_year[[i]] <- att_gt(
    yname = "ReturnsLag",
    tname = "Month",
    idname = "Country_numeric",
    gname = "DST",
    data = dfs_by_year[[i]],
    panel = FALSE,
    control_group = "nevertreated",
  )
}

# Generate plots
unique_years <- unique(df$Year)
att_back_plots <- list()
years <- list()
att_time_0 <- list()
for (i in 1:length(att_by_year)) {
  ATT<-unlist(att_by_year[[i]][3])
  names(ATT) <- NULL
  ATT <- head(ATT[which(!is.na(ATT))], 7)
  
  CI<-unlist(att_by_year[[i]][5])
  names(CI) <- NULL
  CI <- head(CI[which(!is.na(CI))], 7)
  
  if (length(ATT) >= 4) {
    att_time_0[[i]] <- ATT[4]
    years[[i]] <- unique_years[i]
  }
  df_plot <- data.frame(
    time_period = seq(from=-3, by=1, length.out=length(ATT)),
    att_estimate = ATT,
    lower_ci = ATT - CI,
    upper_ci = ATT + CI
  )
  
  df_plot$color <- ifelse(df_plot$time_period < 0, "blue", "red")
  
  p <- ggplot(df_plot, aes(x = time_period, y = att_estimate, color = color)) +
    
    geom_point() +
    
    geom_errorbar(aes(ymin = lower_ci, ymax = upper_ci), width = 0.2) +
    
    geom_hline(yintercept = 0, linetype = "dashed") +
    
    labs(x = "Months since treatment", y = " % Change on Market Returns",
         title = paste("Treatment Effect on Market Returns, year=", unique_years[i])) +
    theme(legend.position = "none")
  
  # Storing all of the plots, one for each year, in list att_back_plots
  att_back_plots[[i]] <- p
  
}

# Generate aggregate ATT vs time plot, to be stored in 'plot'
ATT_vs_time <- data.frame(
  Years = unlist(years),
  AttTime = unlist(att_time_0)
)
