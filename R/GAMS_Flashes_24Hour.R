# ============================ SETUP ============================
library(mgcv)
library(nlme)
library(readr)
library(ggplot2)
library(dplyr)
library(lubridate)
library(tidyr)
library(stringr)

# ============================ CONFIG ============================
TZ_USE <- "America/New_York"
LIGHTS_ON_SHIFT_H <- 7.5
K_HOUR <- 16

# ============================ LOAD & PREP ============================
whole_df <- read_csv(
  "../data/processed_rat_data.csv.gz",
  show_col_types = FALSE, progress = FALSE,
  na = c("", "NA", "NaN", "null")
)

# keep 24h sessions only
whole_df <- whole_df %>%
  mutate(
    daily_chr = tolower(trimws(as.character(daily))),
    is_24h = daily_chr %in% c("24","24h","24hr","24-hr","24 hrs","24 hours","24 hour","24 hr") |
             suppressWarnings(!is.na(as.numeric(daily_chr)) & as.numeric(daily_chr) == 24)
  ) %>%
  filter(is_24h)

message("Kept animals (24h sessions only):")
print(whole_df %>% distinct(name) %>% arrange(name))

# robust-ish datetime parse into desired TZ
parse_dt <- function(x) {
  x <- str_trim(x)
  parse_date_time(x, orders = c("Y-m-d H:M:S", "Y-m-d H:M"), tz = TZ_USE, quiet = TRUE)
}
whole_df <- whole_df %>% mutate(trial_datetime = parse_dt(trial_datetime))

bad_idx <- which(is.na(whole_df$trial_datetime))
if (length(bad_idx) > 0) {
  message("WARNING: Some timestamps failed to parse. Examples (up to 10):")
  print(unique(whole_df$trial_datetime[bad_idx])[1:min(10, length(bad_idx))])
}

df <- whole_df %>%
  select(name, correct, trial_datetime, rt) %>%
  mutate(
    name = factor(name),
    correct = as.integer(as.logical(correct)),  # force to {0,1}
    hour_of_day = hour(trial_datetime) + minute(trial_datetime)/60 + second(trial_datetime)/3600,
    hour_cont   = (hour_of_day - LIGHTS_ON_SHIFT_H) %% 24,
    hour_0p5    = (floor(hour_of_day * 2) / 2 - LIGHTS_ON_SHIFT_H) %% 24,
    rt_bin      = round(rt / 0.1)  * 0.1,
    rt_bin_025  = round(rt / 0.25) * 0.25
  ) %>%
  filter(!is.na(name), !is.na(correct), !is.na(hour_cont), !is.na(rt))

# ============================ MODELS ============================
# ---- ACCURACY (binomial) ----
K_POP <- 20  # population k
K_FS  <- 8   # per-animal deviation k

m_acc_hour <- bam(
  correct ~
    s(hour_cont, bs = "cc", k = K_POP) +
    s(name, bs = "re") +
    s(hour_cont, name, bs = "fs", k = K_FS, m = 1, xt = list(bs = "cc")),
  family   = binomial("logit"),
  data     = df,
  method   = "fREML",
  discrete = TRUE,
  knots    = list(hour_cont = c(0, 24)),
  gamma    = 1.3, select = TRUE
)
print(logLik(m_acc_hour))

# ---- RT (Gamma with log link, response = seconds) ----
K_POP_RT <- 16
K_FS_RT  <- 8
m_rt_hour <- bam(
  rt ~ s(hour_cont, bs="cc", k=K_POP_RT) +
       s(name, bs="re") +
       s(hour_cont, name, bs="fs", k=K_FS_RT, m=1, xt=list(bs="cc")),
  family = Gamma(link="log"),
  data   = df, method="fREML", discrete=TRUE,
  knots  = list(hour_cont=c(0,24)), gamma=1.3, select=TRUE
)
print(logLik(m_rt_hour))

# ---- TRIAL PRODUCTION (NB with offset; 10-min bins default) ----
USE_AR <- FALSE
RHO    <- 0.3
K_POP_P <- 16
K_FS_P  <- 8
BW_HOURS <- 1/6
bw_hours <- BW_HOURS

# bin edges
bin_seq <- seq(0, 24 - bw_hours, by = bw_hours)

# counts per animal x bin
counts_raw <- df %>%
  mutate(
    session  = as.Date(trial_datetime, tz = TZ_USE),
    hour_bin = floor(hour_cont / bw_hours) * bw_hours
  ) %>%
  group_by(name, hour_bin) %>%
  summarise(
    n_trials   = n(),
    n_sessions = n_distinct(session),
    .groups = "drop"
  )

# complete grid + offset(exposure)
count_df <- tidyr::complete(
  counts_raw,
  name     = levels(df$name),
  hour_bin = seq(0, 24 - bw_hours, by = bw_hours),
  fill = list(n_trials = 0, n_sessions = 0)
) %>%
  mutate(
    name = factor(name, levels = levels(df$name)),
    hour_bin = as.numeric(hour_bin)
  ) %>%
  # drop rows with truly zero exposure (no sessions ever contributed to that bin)
  filter(n_sessions > 0) %>%
  mutate(
    offset_log_exposure = log(n_sessions * bw_hours)
  ) %>%
  arrange(name, hour_bin)

stopifnot(nrow(count_df) > 0, is.factor(count_df$name), is.numeric(count_df$hour_bin),
          all(count_df$hour_bin >= 0 & count_df$hour_bin <= 24))

count_df <- count_df %>% arrange(name, hour_bin) %>%
  group_by(name) %>% mutate(AR_start = row_number()==1) %>% ungroup()

m_trials <- bam(
  n_trials ~ offset(offset_log_exposure) +
             s(hour_bin, bs="cc", k=K_POP_P) +
             s(name, bs="re") +
             s(hour_bin, name, bs="fs", k=K_FS_P, m=1, xt=list(bs="cc")),
  family   = nb(), data = count_df,
  method   = "fREML", discrete = TRUE,
  knots    = list(hour_bin = c(0, 24)),
  rho      = if (USE_AR) RHO else 0,
  AR.start = if (USE_AR) count_df$AR_start else NULL,
  gamma    = 1.3, select = TRUE
)
print(logLik(m_trials))

# Okabe–Ito / "Wong" palette
julia_colors <- c(
  "#0072B2", # blue
  "#D55E00", # vermillion
  "#009E73", # green
  "#CC79A7", # purple
  "#F0E442", # yellow
  "#56B4E9", # sky
  "#E69F00", # orange
  "#000000"  # black
)

scale_color_julia <- function(...) scale_color_manual(values = julia_colors, ...)
scale_fill_julia  <- function(...) scale_fill_manual(values  = julia_colors, ...)

theme_julia <- function(base_size = 12, base_family = "sans") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # box frame like Makie/Plots.jl
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.6),
      # very light grid; minor off
      panel.grid.major = element_line(color = "#D9D9D9", linewidth = 0.35),
      panel.grid.minor = element_blank(),
      # axis text/labels
      axis.title = element_text(face = "bold", color = "black"),
      axis.text  = element_text(color = "black"),
      # ticks: a bit longer, square caps
      axis.ticks       = element_line(color = "black", linewidth = 0.6, lineend = "butt"),
      axis.ticks.length = unit(6, "pt"),
      # legend with thin border, white fill
      legend.background = element_rect(fill = "white", color = "black", linewidth = 0.4),
      legend.key = element_rect(fill = "white", color = NA),
      legend.position = "top",
      # facet headers
      strip.background = element_rect(fill = "#F3F3F3", color = "black", linewidth = 0.4),
      strip.text = element_text(face = "bold"),
      # titles
      plot.title = element_text(face = "bold", size = base_size * 1.2, margin = margin(b = 4))
    )
}

# ============================ RAW DOTS (same as before) ============================
# Accuracy raw summary (pooled trials per integer hour)
raw_summary <- df %>%
  mutate(hour_bin = floor(hour_cont * 2) / 2) %>%
  group_by(name, hour_bin) %>%
  summarise(p_correct = mean(correct), .groups = "drop") %>%
  group_by(hour_bin) %>%
  summarise(
    n_animals = n(),
    p_hat = mean(p_correct),
    se = sd(p_correct) / sqrt(n_animals),
    .groups = "drop"
  ) %>%
  mutate(hour_mid = hour_bin + 0.25)

# RT raw summary (integer hours)
raw_rt <- df %>%
  mutate(hour_bin = floor(hour_cont * 2) / 2) %>%
  group_by(name, hour_bin) %>%
  summarise(mean_rt = mean(rt, na.rm = TRUE), .groups = "drop") %>%
  group_by(hour_bin) %>%
  summarise(
    n_animals = n(),
    mean_rt = mean(mean_rt),  # average of animal means
    se = sd(mean_rt) / sqrt(n_animals),  # SE across animals
    .groups = "drop"
  ) %>%
  mutate(hour_mid = hour_bin + 0.25)

# Trial production: per-hour per-animal dots
raw_trials_per_animal <- count_df %>%
  mutate(hour_int = floor(hour_bin)) %>%
  group_by(hour_int) %>%
  summarise(
    total_trials = sum(n_trials),
    n_animals    = n_distinct(name),
    rate_per_animal = (total_trials / n_animals) / 1.0,
    se_rate = (sqrt(total_trials) / n_animals) / 1.0,
    .groups = "drop"
  ) %>%
  mutate(hour_mid = hour_int + 0.5)

# ============================ MARGINAL CURVES (avg over animals) ============================
set.seed(123)
B <- 300  # bootstrap replicates for ribbons
animal_ids <- levels(df$name)

# ---- Accuracy ----
hour_seq <- seq(0, 24, by = 0.05)
grid_all_acc <- tidyr::expand_grid(name = levels(df$name), hour_cont = hour_seq)
grid_all_acc$pred <- predict(m_acc_hour, newdata = grid_all_acc, type = "response")

avg_acc <- grid_all_acc %>%
  group_by(hour_cont) %>%
  summarise(fit = mean(pred), .groups = "drop")

boot_mat_acc <- replicate(B, {
  samp <- sample(animal_ids, length(animal_ids), replace = TRUE)
  grid_all_acc %>% filter(name %in% samp) %>%
    group_by(hour_cont) %>% summarise(fit = mean(pred), .groups = "drop") %>%
    pull(fit)
})
avg_acc <- avg_acc %>%
  mutate(lower = apply(boot_mat_acc, 1, quantile, 0.025),
         upper = apply(boot_mat_acc, 1, quantile, 0.975))

# ---- RT (Gamma/log; predictions on response) ----
grid_all_rt <- tidyr::expand_grid(name = levels(df$name), hour_cont = hour_seq)
grid_all_rt$pred <- predict(m_rt_hour, newdata = grid_all_rt, type = "response")

avg_rt <- grid_all_rt %>%
  group_by(hour_cont) %>%
  summarise(fit = mean(pred), .groups = "drop")

boot_mat_rt <- replicate(B, {
  samp <- sample(animal_ids, length(animal_ids), replace = TRUE)
  grid_all_rt %>% filter(name %in% samp) %>%
    group_by(hour_cont) %>% summarise(fit = mean(pred), .groups = "drop") %>%
    pull(fit)
})
avg_rt <- avg_rt %>%
  mutate(lower = apply(boot_mat_rt, 1, quantile, 0.025),
         upper = apply(boot_mat_rt, 1, quantile, 0.975))

# ---- Trial production (NB + offset) -> per-hour rate ----
hour_seq_rate <- seq(0, 24, by = 0.05)
grid_all_trials <- tidyr::expand_grid(
  name = levels(count_df$name),
  hour_bin = hour_seq_rate
) %>%
  mutate(offset_log_exposure = log(1))  # = 0, means "1 hour window"

# expected count per 1-hour window (includes RE + fs)
grid_all_trials$mu_bin <- predict(m_trials, newdata = grid_all_trials, type = "response")
grid_all_trials$rate_per_hour <- grid_all_trials$mu_bin  # Already per hour

avg_trials <- grid_all_trials %>%
  group_by(hour_bin) %>%
  summarise(fit = mean(rate_per_hour), .groups = "drop")

boot_mat_tr <- replicate(B, {
  samp <- sample(levels(count_df$name), length(levels(count_df$name)), replace = TRUE)
  grid_all_trials %>% filter(name %in% samp) %>%
    group_by(hour_bin) %>% summarise(fit = mean(rate_per_hour), .groups = "drop") %>%
    pull(fit)
})
avg_trials <- avg_trials %>%
  mutate(lower = apply(boot_mat_tr, 1, quantile, 0.025),
         upper = apply(boot_mat_tr, 1, quantile, 0.975))

# ============================ PLOTS ============================
# ---- Accuracy ----
p_acc_vs_tod <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "grey90", alpha = 0.5) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "lightyellow", alpha = 0.8) +
  geom_ribbon(data = avg_acc, aes(x = hour_cont, ymin = lower, ymax = upper),
              fill = "royalblue3", alpha = 0.2) +
  geom_line(data = avg_acc, aes(x = hour_cont, y = fit),
            color = "royalblue3", linewidth = 1.2) +
  geom_point(data = raw_summary,
             aes(x = hour_mid, y = p_hat),
             color = "black", alpha = 0.8, size = 2) +
  scale_x_continuous(breaks = seq(0, 24, by = 4), limits = c(0, 24)) +
  labs(x = "Time from lights on (h)", 
       y = "Accuracy (proportion correct)") +
  theme_julia(base_size = 14) +
  theme(
    axis.title = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 13),
    panel.grid.minor = element_blank()
  )

print(p_acc_vs_tod)

# ---- RT ----
p_rt_vs_tod <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "grey90", alpha = 0.5) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "lightyellow", alpha = 0.8) +
  geom_ribbon(data = avg_rt, aes(x = hour_cont, ymin = lower, ymax = upper),
              fill = "royalblue3", alpha = 0.2) +
  geom_line(data = avg_rt, aes(x = hour_cont, y = fit),
            color = "royalblue3", linewidth = 1.2) +
  geom_linerange(data = raw_rt,
                 aes(x = hour_mid, ymin = mean_rt - se, ymax = mean_rt + se),
                 color = "black", linewidth = 0.5) +
  geom_point(data = raw_rt,
             aes(x = hour_mid, y = mean_rt),
             color = "black", alpha = 0.8, size = 2) +
  scale_x_continuous(breaks = seq(0, 24, by = 4), limits = c(0, 24)) +
  labs(x = "Time from lights on (h)", 
       y = "Reaction time (s)") +
  theme_julia(base_size = 14) +
  theme(
    axis.title = element_text(size = 16, face = "bold"),
    axis.text = element_text(size = 13),
    panel.grid.minor = element_blank()
  )

print(p_rt_vs_tod)

# ---- Trial production ----
# 1) per animal × integer hour: sum trials & exposure, then rate
per_animal_hour <- count_df %>%
  mutate(hour_int = floor(hour_bin),
         exposure = n_sessions * bw_hours) %>%
  group_by(name, hour_int) %>%
  summarise(
    trials   = sum(n_trials),
    exposure = sum(exposure),
    rate     = trials / exposure,     # trials per hour for that animal & hour
    .groups = "drop"
  )

# 2) aggregate across animals for the dots & error bars
raw_trials_per_animal <- per_animal_hour %>%
  group_by(hour_int) %>%
  summarise(
    n_animals = n(),
    rate_per_animal = mean(rate),
    se_rate = sd(rate) / sqrt(n_animals),
    .groups = "drop"
  ) %>%
  mutate(hour_mid = hour_int + 0.5)

p_trials_rate <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "grey90", alpha = 0.5) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "lightyellow", alpha = 0.8) +
  geom_ribbon(data = avg_trials,
              aes(x = hour_bin, ymin = lower, ymax = upper),
              alpha = 0.2, fill = "royalblue3") +
  geom_line(data = avg_trials, aes(x = hour_bin, y = fit),
            color = "royalblue3", linewidth = 1.2) +
  geom_point(data = raw_trials_per_animal,
             aes(x = hour_mid, y = rate_per_animal),
             color = "black", alpha = 0.85, size = 2) +
  scale_x_continuous(breaks = seq(0, 24, by = 4), limits = c(0, 24)) +
  labs(x = "Time from lights on (h)", y = "Trial rate (trials/h)") +
  theme_julia(base_size = 14) +
  theme(
    axis.title = element_text(size = 16, face = "bold"),
    axis.text  = element_text(size = 13),
    panel.grid.minor = element_blank()
  )

print(p_trials_rate)

ggsave("flashes_acc_vs_tod_marginal.pdf", p_acc_vs_tod, width = 8, height = 6)
ggsave("flashes_rt_vs_tod_marginal.pdf",  p_rt_vs_tod,  width = 8, height = 6)
ggsave("flashes_trials_rate_marginal.pdf", p_trials_rate, width = 8, height = 6)

# ============================ PER-ANIMAL RAW DOTS FOR FACETS ============================
# Accuracy per animal
raw_summary_indiv <- df %>%
  mutate(hour_bin = floor(hour_cont * 2) / 2) %>%
  group_by(name, hour_bin) %>%
  summarise(
    n_trials = n(),
    p_hat = mean(correct),
    se = sqrt(p_hat * (1 - p_hat) / n_trials),
    .groups = "drop"
  ) %>%
  mutate(hour_mid = hour_bin + 0.25)

# RT per animal
raw_rt_indiv <- df %>%
  mutate(hour_bin = floor(hour_cont * 2) / 2) %>%
  group_by(name, hour_bin) %>%
  summarise(
    n_trials = n(),
    mean_rt = mean(rt, na.rm = TRUE),
    se = sd(rt, na.rm = TRUE) / sqrt(n_trials),
    .groups = "drop"
  ) %>%
  mutate(hour_mid = hour_bin + 0.25)

# Trial production per animal (using existing count_df)
raw_trials_indiv <- count_df %>%
  mutate(
    hour_int = floor(hour_bin),
    exposure = n_sessions * bw_hours          # total hours observed for this row
  ) %>%
  group_by(name, hour_int) %>%
  summarise(
    total_trials    = sum(n_trials),
    total_exposure  = sum(exposure),
    rate_per_animal = ifelse(total_exposure > 0, total_trials / total_exposure, NA_real_),
    .groups = "drop"
  ) %>%
  filter(!is.na(rate_per_animal)) %>%
  mutate(hour_mid = hour_int + 0.5)
# ============================ INDIVIDUAL ANIMAL FACETS WITH DOTS ============================

# Accuracy by animal
pred_acc_indiv <- grid_all_acc %>%
  mutate(name = factor(name, levels = levels(df$name)))

p_acc_facet <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "lightgrey", alpha = 0.3) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "lightyellow", alpha = 0.8) +
  geom_point(data = raw_summary_indiv,
             aes(x = hour_mid, y = p_hat, size = n_trials),
             color = "black", alpha = 0.5) +
  geom_line(data = pred_acc_indiv, aes(x = hour_cont, y = pred),
            color = "royalblue3", linewidth = 0.8) +
  facet_wrap(~ name, ncol = 4, scales = "free_y") +
  scale_size_continuous(name = "trials", range = c(0.5, 3)) +
  scale_x_continuous(breaks = c(0, 12, 24)) +
  labs(x = "hours from light onset", y = "accuracy") +
  theme_julia() +
  theme(strip.text = element_text(size = 8))

# RT by animal
pred_rt_indiv <- grid_all_rt %>%
  mutate(name = factor(name, levels = levels(df$name)))

p_rt_facet <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "lightgrey", alpha = 0.3) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "lightyellow", alpha = 0.8) +
  geom_point(data = raw_rt_indiv,
             aes(x = hour_mid, y = mean_rt, size = n_trials),
             color = "black", alpha = 0.5) +
  geom_line(data = pred_rt_indiv, aes(x = hour_cont, y = pred),
            color = "royalblue3", linewidth = 0.8) +
  facet_wrap(~ name, ncol = 4, scales = "free_y") +
  scale_size_continuous(name = "trials", range = c(0.5, 3)) +
  scale_x_continuous(breaks = c(0, 12, 24)) +
  labs(x = "hours from light onset", y = "RT (s)") +
  theme_minimal() +
  theme(strip.text = element_text(size = 8))

# Trial production by animal
pred_trials_indiv <- grid_all_trials %>%
  mutate(name = factor(name, levels = levels(count_df$name)))

p_trials_facet <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "lightgrey", alpha = 0.3) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "lightyellow", alpha = 0.8) +
  geom_point(data = raw_trials_indiv,
             aes(x = hour_mid, y = rate_per_animal, size = total_exposure),
             color = "black", alpha = 0.5) +
  geom_line(data = pred_trials_indiv, aes(x = hour_bin, y = rate_per_hour),
            color = "royalblue3", linewidth = 0.8) +
  facet_wrap(~ name, ncol = 4, scales = "free_y") +
  scale_size_continuous(name = "exposure (h)", range = c(0.5, 3)) +
  scale_x_continuous(breaks = c(0, 12, 24)) +
  labs(x = "hours from light onset", y = "trials/hour") +
  theme_julia() +
  theme(strip.text = element_text(size = 8))

print(p_acc_facet)
print(p_rt_facet)
print(p_trials_facet)

# ggsave("flashes_acc_facet.pdf", p_acc_facet, width = 12, height = 10)
# ggsave("flashes_rt_facet.pdf", p_rt_facet, width = 12, height = 10)
# ggsave("flashes_trials_facet.pdf", p_trials_facet, width = 12, height = 10)

# =========================== VARIABILITY PLOTS =============================================
# --- Per-day stats per animal ---
day_stats <- df %>%
  mutate(session = as.Date(trial_datetime, tz = TZ_USE)) %>%
  group_by(name, session) %>%
  summarise(
    rt_mean   = mean(rt, na.rm = TRUE),
    acc_mean  = mean(correct, na.rm = TRUE),
    trials    = n(),
    .groups   = "drop"
  )

# --- Summaries across days (center = median; error bars = min..max) ---
rt_sum <- day_stats %>%
  group_by(name) %>%
  summarise(
    center = median(rt_mean, na.rm = TRUE),
    min    = min(rt_mean, na.rm = TRUE),
    max    = max(rt_mean, na.rm = TRUE),
    n_days = n(),
    .groups = "drop"
  )

acc_sum <- day_stats %>%
  group_by(name) %>%
  summarise(
    center = median(acc_mean, na.rm = TRUE),
    min    = min(acc_mean, na.rm = TRUE),
    max    = max(acc_mean, na.rm = TRUE),
    n_days = n(),
    .groups = "drop"
  )

trials_sum <- day_stats %>%
  group_by(name) %>%
  summarise(
    center = median(trials, na.rm = TRUE),
    min    = min(trials, na.rm = TRUE),
    max    = max(trials, na.rm = TRUE),
    n_days = n(),
    .groups = "drop"
  )

# --- Helper: ranked range plot (Makie/Julia-esque) ---
make_ranked_range_plot <- function(sum_df, ylab, title, fill_col, decreasing = FALSE) {
  ord_df <- sum_df %>%
    arrange(if (decreasing) dplyr::desc(center) else center) %>%
    mutate(name_ord = factor(name, levels = name))

  ggplot(ord_df, aes(x = name_ord)) +
    # range across days
    geom_linerange(aes(ymin = min, ymax = max),
                   linewidth = 1.0, color = "black", lineend = "butt") +
    # central tendency point
    geom_point(aes(y = center),
               shape = 21, size = 3.2, stroke = 0.55, fill = fill_col, color = "black") +
    coord_flip() +
    labs(x = NULL, y = ylab, title = title) +
    theme_julia(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      axis.title.y = element_blank()
    )
}

# --- Colors from your Okabe–Ito palette ---
col_rt     <- julia_colors[1]  # blue
col_acc    <- julia_colors[3]  # green
col_trials <- julia_colors[7]  # orange

# --- Build the three ranked plots ---
# RT: ascending (fastest first)
p_rt_rank <- make_ranked_range_plot(
  rt_sum,
  ylab  = "Reaction time (s)",
  title = "RT — median daily value (point) with range across days (bar)",
  fill_col = col_rt,
  decreasing = TRUE
)

# Accuracy: descending (best first)
p_acc_rank <- make_ranked_range_plot(
  acc_sum,
  ylab  = "Accuracy (proportion correct)",
  title = "Accuracy — median daily value with range across days",
  fill_col = col_acc,
  decreasing = TRUE
)

# Trials/day: descending (most productive first)
p_trials_rank <- make_ranked_range_plot(
  trials_sum,
  ylab  = "Trials per day",
  title = "Trials/day — median with range across days",
  fill_col = col_trials,
  decreasing = TRUE
)

# --- Show them ---
print(p_rt_rank)
print(p_acc_rank)
print(p_trials_rank)

# ggsave("ranked_rt.svg",     p_rt_rank,     width = 7.5, height = 6.0)
# ggsave("ranked_acc.svg",    p_acc_rank,    width = 7.5, height = 6.0)
# ggsave("ranked_trials.svg", p_trials_rank, width = 7.5, height = 6.0)

MIN_TRIALS_PER_HOUR <- 200   # set to 0 to keep all hours

# ---- 1) Build per-session, per-hour stats (one row per name × session × hour)
hourly_base <- df %>%
  mutate(
    session  = as.Date(trial_datetime, tz = TZ_USE),
    hour_int = floor(hour_cont)                    # 0..23
  ) %>%
  group_by(name, session, hour_int) %>%
  summarise(
    n_trials = n(),
    acc_hour = mean(correct),                      # accuracy within that session-hour
    rt_hour  = mean(rt, na.rm = TRUE),             # mean RT within that session-hour
    .groups  = "drop"
  )

# ---- 2) Collapse across sessions to get per-animal, per-hour metrics
# Weighted by trials per session-hour (so fuller hours count more).
hourly_by_hour <- hourly_base %>%
  group_by(name, hour_int) %>%
  summarise(
    n_sessions = n(),                              # sessions that contributed to this hour
    total_trials = sum(n_trials),
    acc_hour  = stats::weighted.mean(acc_hour, w = n_trials, na.rm = TRUE),
    rt_hour   = stats::weighted.mean(rt_hour,  w = n_trials, na.rm = TRUE),
    rate_hour = total_trials / n_sessions,         # avg trials per session for this hour
    .groups   = "drop"
  ) %>%
  filter(total_trials >= MIN_TRIALS_PER_HOUR)      # optional sparsity filter

# ---- 3) Summaries across HOURS per animal: median (point) + min..max (bar)
summarize_hourly <- function(df, var) {
  df %>%
    group_by(name) %>%
    summarise(
      center  = median({{var}}, na.rm = TRUE),
      min     = min({{var}},    na.rm = TRUE),
      max     = max({{var}},    na.rm = TRUE),
      n_hours = sum(!is.na({{var}})),
      .groups = "drop"
    )
}

rt_hour_sum     <- summarize_hourly(hourly_by_hour, rt_hour)
acc_hour_sum    <- summarize_hourly(hourly_by_hour, acc_hour)
trials_hour_sum <- summarize_hourly(hourly_by_hour, rate_hour)

# ---- 4) Generic ranked plot helper (Julia/Makie-ish)
make_ranked_range_plot <- function(sum_df, ylab, title, fill_col, decreasing = TRUE) {
  ord_df <- sum_df %>%
    arrange(if (decreasing) dplyr::desc(center) else center) %>%
    mutate(name_ord = factor(name, levels = name))

  ggplot(ord_df, aes(x = name_ord)) +
    geom_linerange(aes(ymin = min, ymax = max),
                   linewidth = 1.0, color = "black", lineend = "butt") +
    geom_point(aes(y = center),
               shape = 21, size = 3.2, stroke = 0.55, fill = fill_col, color = "black") +
    coord_flip() +
    labs(x = NULL, y = ylab, title = title) +
    theme_julia(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          axis.title.y = element_blank())
}

# ---- 5) Colors from your Okabe–Ito palette
col_rt     <- julia_colors[1]  # blue
col_acc    <- julia_colors[3]  # green
col_trials <- julia_colors[7]  # orange

# ---- 6) Build the three ranked plots
# Order all three with larger center at the top (your current preference)
p_rt_hour_rank <- make_ranked_range_plot(
  rt_hour_sum,
  ylab  = "Reaction time (s) — hour-by-hour",
  title = paste0("RT (hourly) — median across hours (point) with range (bar)",
                 if (MIN_TRIALS_PER_HOUR > 0) paste0("  [≥", MIN_TRIALS_PER_HOUR, " trials/hr]") else ""),
  fill_col  = col_rt,
  decreasing = TRUE
)

p_acc_hour_rank <- make_ranked_range_plot(
  acc_hour_sum,
  ylab  = "Accuracy — hour-by-hour",
  title = paste0("Accuracy (hourly) — median across hours with range",
                 if (MIN_TRIALS_PER_HOUR > 0) paste0("  [≥", MIN_TRIALS_PER_HOUR, " trials/hr]") else ""),
  fill_col  = col_acc,
  decreasing = TRUE
)

p_trials_hour_rank <- make_ranked_range_plot(
  trials_hour_sum,
  ylab  = "Trials per hour (avg per session)",
  title = paste0("Trials/hour (hourly) — median across hours with range",
                 if (MIN_TRIALS_PER_HOUR > 0) paste0("  [≥", MIN_TRIALS_PER_HOUR, " trials/hr]") else ""),
  fill_col  = col_trials,
  decreasing = TRUE
)

# ---- 7) Show them
print(p_rt_hour_rank)
print(p_acc_hour_rank)
print(p_trials_hour_rank)

ggsave("ranked_rt.svg",     p_rt_rank,     width = 4, height = 6.0)
ggsave("ranked_acc.svg",    p_acc_rank,    width = 4, height = 6.0)
ggsave("ranked_trials.svg", p_trials_rank, width = 4, height = 6.0)