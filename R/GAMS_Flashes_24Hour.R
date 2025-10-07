# ------------------------ SETUP ------------------------
library(mgcv)
library(nlme)
library(readr)      # fast I/O; reads .gz directly
library(ggplot2)
library(dplyr)
library(lubridate)
library(tidyr)
library(stringr)

# ------------------------ CONFIG ------------------------
# Rig timezone (change if needed)
TZ_USE <- "America/New_York"

# Lights-on offset in hours (shift so lights-on = 0)
LIGHTS_ON_SHIFT_H <- 7.5

# Population smooth basis size (modest; increase a bit if clearly underfitting)
K_HOUR <- 16

# ------------------------ LOAD & PREP ------------------------
# read_csv auto-detects gzip; set col types if you want strictness
whole_df <- read_csv(
  "../data/processed_rat_data.csv.gz",
  show_col_types = FALSE,
  progress = FALSE,
  na = c("", "NA", "NaN", "null")
)

# --- Keep only 24-hour sessions, drop 2-hour sessions ---
# Normalize 'daily' and detect 24h values robustly
whole_df <- whole_df %>%
  mutate(
    daily_chr = tolower(trimws(as.character(daily))),
    # flag as 24h if textual match OR numeric 24
    is_24h = daily_chr %in% c("24", "24h", "24hr", "24-hr", "24 hrs", "24 hours", "24 hour", "24 hr") |
             suppressWarnings(!is.na(as.numeric(daily_chr)) & as.numeric(daily_chr) == 24)
  ) %>%
  filter(is_24h)

# (Optional) sanity check: which animals remain
kept_animals <- whole_df %>% distinct(name) %>% arrange(name)
message("Kept animals (24h sessions only):")
print(kept_animals)

# Robust datetime parse:
# - Accepts "YYYY-mm-dd HH:MM:SS" or "YYYY-mm-dd HH:MM"
# - If your file includes timezone offsets, readr will parse them; otherwise we set TZ.
parse_dt <- function(x) {
  # Trim stray whitespace
  x <- str_trim(x)
  # Try two common orders; add more if you see parsing failures
  dt <- parse_date_time(x, orders = c("Y-m-d H:M:S", "Y-m-d H:M"), tz = TZ_USE, quiet = TRUE)
  dt
}

whole_df <- whole_df %>%
  mutate(trial_datetime = parse_dt(trial_datetime))

# Show a quick sample of any unparsed timestamps
bad_idx <- which(is.na(whole_df$trial_datetime))
if (length(bad_idx) > 0) {
  message("WARNING: Some timestamps failed to parse. Examples (up to 10):")
  print(unique(whole_df$trial_datetime[bad_idx])[1:min(10, length(bad_idx))])
}

df <- whole_df %>%
  select(name, correct, trial_datetime, rt) %>%
  mutate(
    name = factor(name),
    # Ensure binomial 0/1 numeric. If 'correct' is logical or "TRUE"/"FALSE", adjust here:
    correct = as.integer(correct),
    # Decimal hour-of-day (0..24)
    hour_of_day = hour(trial_datetime) + minute(trial_datetime) / 60 + second(trial_datetime) / 3600,
    # Hours from lights-on, continuous on [0,24)
    hour_cont   = (hour_of_day - LIGHTS_ON_SHIFT_H) %% 24,
    # Optional half-hour bin for summaries (kept for convenience)
    hour_0p5    = (floor(hour_of_day * 2) / 2 - LIGHTS_ON_SHIFT_H) %% 24,
    # RT bins if/when you need them later
    rt_bin      = round(rt / 0.1)  * 0.1,
    rt_bin_025  = round(rt / 0.25) * 0.25
  ) %>%
  # Drop unusable rows
  filter(!is.na(name), !is.na(correct), !is.na(hour_cont))

# ------------------------ MODEL: ACCURACY ~ TIME OF DAY (trial-level) ------------------------
K_POP <- 16   # population smooth df (try 16–20)
K_FS  <- 8    # per-animal deviation df (try 6–10)

m_acc_hour <- bam(
  correct ~
    s(hour_cont, bs = "cc", k = K_POP) +             # population cyclic smooth
    s(name, bs = "re") +                             # random intercepts
    s(hour_cont, name, bs = "fs", k = K_FS,          # per-animal cyclic deviations
      m = 1, xt = list(bs = "cc")),
  family   = binomial("logit"),
  data     = df,
  method   = "fREML",
  discrete = TRUE,
  knots    = list(hour_cont = c(0, 24)),             # REQUIRED for cyclic
  gamma    = 1.3,
  select   = TRUE
)

print(logLik(m_acc_hour))

# ------------------------ POPULATION PREDICTIONS (exclude RE + per-animal deviations) ------------------------
grid_hour <- tibble(
  hour_cont = seq(0, 24, by = 0.05),
  name      = levels(df$name)[1]   # dummy level
)

pred <- predict(
  m_acc_hour,
  newdata = grid_hour,
  type    = "link",
  se.fit  = TRUE,
  exclude = c("s(name)", "s(hour_cont,name)")
)

pop_curve <- grid_hour %>%
  mutate(
    fit   = plogis(pred$fit),
    upper = plogis(pred$fit + 1.96 * pred$se.fit),
    lower = plogis(pred$fit - 1.96 * pred$se.fit)
)

# ------------------------ RAW WEIGHTED SUMMARY (DOTS & BARS) ------------------------
# Pool across all animals within each *integer* hour bin.
# This weights by the number of trials (busy hours get larger dots and tighter SE).
raw_summary <- df %>%
  mutate(hour_bin = floor(hour_cont)) %>%
  group_by(hour_bin) %>%
  summarise(
    n_trials    = n(),
    p_hat       = mean(correct),
    # Binomial SE for pooled trials
    se          = sqrt(p_hat * (1 - p_hat) / n_trials),
    .groups = "drop"
  ) %>%
  mutate(
    # Midpoint for plotting (center of the integer bin)
    hour_mid = hour_bin + 0.5
  )

# ------------------------ PLOT ------------------------
p_acc_vs_tod <- ggplot() +
  # Background shading (edit ranges as you like)
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "lightgrey", alpha = 0.3) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "paleturquoise3", alpha = 0.2) +

  # Population curve (smooth + 95% band)
  geom_ribbon(
    data = pop_curve,
    aes(x = hour_cont, ymin = lower, ymax = upper),
    fill = "royalblue3", alpha = 0.2
  ) +
  geom_line(
    data = pop_curve,
    aes(x = hour_cont, y = fit),
    color = "royalblue3", linewidth = 1
  ) +

  # Weighted raw dots and binomial SE bars (size ~ # trials)
  geom_linerange(
    data = raw_summary,
    aes(x = hour_mid, ymin = p_hat - se, ymax = p_hat + se),
    color = "black", linewidth = 0.4
  ) +
  geom_point(
    data = raw_summary,
    aes(x = hour_mid, y = p_hat, size = n_trials),
    color = "black", alpha = 0.8
  ) +

  scale_size_continuous(name = "trials in bin", range = c(1, 5)) +
  scale_x_continuous(breaks = seq(0, 24, by = 2), limits = c(0, 24)) +
  labs(x = "hours from light onset", y = "accuracy") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 22),
    axis.text.x  = element_text(size = 20),
    axis.title.y = element_text(size = 22),
    axis.text.y  = element_text(size = 20),
    legend.position = "right"
  )

print(p_acc_vs_tod)
# ggsave("flashes_acc_vs_tod_weighted.pdf", p_acc_vs_tod, width = 8, height = 6)

# ------------------------ OPTIONAL: FACETS BY ANIMAL (population overlay + per-animal dots) ------------------------
# Per-animal raw summaries (integer hour bins), to visualize who contributed where

grid_by_animal <- tidyr::expand_grid(
  name      = levels(df$name),
  hour_cont = seq(0, 24, by = 0.05)
)

pr_an <- predict(m_acc_hour, newdata = grid_by_animal, type = "response", se.fit = FALSE)
per_animal_curve <- dplyr::mutate(grid_by_animal, fit = pr_an)

raw_by_animal <- df %>%
  mutate(hour_bin = floor(hour_cont)) %>%
  group_by(name, hour_bin) %>%
  summarise(
    n_trials = n(),
    p_hat    = mean(correct),
    se       = sqrt(p_hat * (1 - p_hat) / n_trials),
    .groups  = "drop"
  ) %>%
  mutate(hour_mid = hour_bin + 0.5)

p_facets <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "lightgrey", alpha = 0.3) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "paleturquoise3", alpha = 0.2) +

  geom_line(
    data = per_animal_curve,
    aes(x = hour_cont, y = fit),
    linewidth = 0.8,
    color = "black",
    alpha = 0.7
  ) +

  # Per-animal raw dots sized by bin trials
  geom_linerange(
    data = raw_by_animal,
    aes(x = hour_mid, ymin = p_hat - se, ymax = p_hat + se),
    color = "black", linewidth = 0.3
  ) +
  geom_point(
    data = raw_by_animal,
    aes(x = hour_mid, y = p_hat, size = n_trials),
    alpha = 0.8
  ) +

  facet_wrap(~ name, ncol = 4, scales = "fixed") +
  scale_size_continuous(name = "trials in bin", range = c(0.8, 3.5)) +
  scale_x_continuous(breaks = seq(0, 24, by = 4), limits = c(0, 24)) +
  coord_cartesian(ylim = c(0.6, 0.9)) +      # <-- restrict y-axis
  labs(
    title = "Flashes: accuracy by time of day (per animal)",
    x = "hours from light onset",
    y = "accuracy"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text      = element_text(face = "bold", size = 10),
    plot.title      = element_text(size = 16, face = "bold"),
    axis.text       = element_text(size = 8),
    axis.title      = element_text(size = 10),
    legend.position = "right"
  )


# print(p_facets)
# ggsave("flashes_acc_vs_tod_by_animal_weighted.pdf", p_facets, width = 11, height = 8.5)

# ======================== RT ~ TIME OF DAY (trial-level) ========================
# ====================== TOGGLES & CONSTANTS ======================
USE_AR   <- TRUE          # turn AR(1) on/off
RHO      <- 0.3           # 0.2–0.6 typical
K_POP    <- 16            # population smooth k
K_FS     <- 8             # per-animal deviation k

BW_HOURS <- 1/6           # 10 min bins for counts

# ====================== TRIAL-LEVEL DF WITH AR STARTS ======================
df_trial <- df %>%
  mutate(session = as.factor(as.Date(trial_datetime, tz = TZ_USE))) %>%
  arrange(name, session, trial_datetime) %>%
  group_by(name, session) %>%
  mutate(AR_start = row_number() == 1) %>%
  ungroup()

# ====================== ACCURACY ~ TIME OF DAY (Binomial) ======================
m_acc_hour <- mgcv::bam(
  correct ~
    s(hour_cont, bs = "cc", k = K_POP) +
    s(name, bs = "re") +
    s(hour_cont, name, bs = "fs", k = K_FS, m = 1, xt = list(bs = "cc")) +
    s(session, bs = "re"),
  family   = binomial("logit"),
  data     = df_trial,
  method   = "fREML",
  discrete = TRUE,
  knots    = list(hour_cont = c(0, 24)),
  gamma    = 1.3,
  select   = TRUE,
  rho      = if (USE_AR) RHO else 0,
  AR.start = if (USE_AR) df_trial$AR_start else NULL
)

# population curve (exclude RE + fs)
grid_hour <- tibble::tibble(
  hour_cont = seq(0, 24, by = 0.05),
  name      = levels(df_trial$name)[1],
  session   = df_trial$session[1]
)
pr_acc <- predict(
  m_acc_hour, grid_hour, type = "link", se.fit = TRUE,
  exclude = c("s(name)", "s(hour_cont,name)", "s(session)")
)
pop_curve <- grid_hour %>%
  mutate(
    fit   = plogis(pr_acc$fit),
    lower = plogis(pr_acc$fit - 1.96 * pr_acc$se.fit),
    upper = plogis(pr_acc$fit + 1.96 * pr_acc$se.fit)
  )

# ====================== TRIAL PRODUCTION COUNTS (NB) ======================
# Per animal × day × bin counts; include zero bins so AR is estimable
bin_seq <- seq(0, 24 - BW_HOURS, by = BW_HOURS)   # left edges: 0 .. 24 - BW

counts_raw <- df %>%
  mutate(
    session  = as.factor(as.Date(trial_datetime, tz = TZ_USE)),
    hour_bin = floor(hour_cont / BW_HOURS) * BW_HOURS
  ) %>%
  group_by(name, session, hour_bin) %>%
  summarise(n_trials = dplyr::n(), .groups = "drop")

count_df <- tidyr::complete(
  counts_raw,
  name     = levels(df$name),
  session  = unique(as.factor(as.Date(df$trial_datetime, tz = TZ_USE))),
  hour_bin = bin_seq,
  fill     = list(n_trials = 0)
) %>%
  mutate(
    name     = factor(name, levels = levels(df$name)),
    session  = factor(session),
    hour_bin = as.numeric(hour_bin)
  ) %>%
  arrange(name, session, hour_bin) %>%
  group_by(name, session) %>%
  mutate(AR_start = row_number() == 1) %>%
  ungroup()

stopifnot(nrow(count_df) > 0, is.factor(count_df$name), is.numeric(count_df$hour_bin))

# NB GAMM with cyclic smooth + per-animal deviations + day RE + optional AR(1)
m_trials <- mgcv::bam(
  n_trials ~
    s(hour_bin, bs = "cc", k = K_POP) +
    s(name, bs = "re") +
    s(hour_bin, name, bs = "fs", k = max(4, K_FS - 2), m = 1, xt = list(bs = "cc")) +
    s(session, bs = "re"),
  family   = nb(link = "log"),
  data     = count_df,
  method   = "fREML",
  discrete = TRUE,
  knots    = list(hour_bin = c(0, 24)),
  gamma    = 1.3,
  select   = TRUE,
  rho      = if (USE_AR) RHO else 0,
  AR.start = if (USE_AR) count_df$AR_start else NULL
)

# ====================== POPULATION PREDICTIONS (per-animal rate) ======================
grid_trials <- tibble::tibble(
  hour_bin = seq(0, 24, by = 0.05),
  name     = levels(count_df$name)[1],
  session  = count_df$session[1]
)
pr_trials <- predict(
  m_trials, newdata = grid_trials, type = "link", se.fit = TRUE,
  exclude = c("s(name)", "s(hour_bin,name)", "s(session)")
)
pop_trials <- grid_trials %>%
  mutate(
    mu_bin        = exp(pr_trials$fit),                      # expected count per bin per animal
    rate_per_hour = mu_bin / BW_HOURS,                       # make rate scale
    rate_lo       = exp(pr_trials$fit - 1.96 * pr_trials$se.fit) / BW_HOURS,
    rate_hi       = exp(pr_trials$fit + 1.96 * pr_trials$se.fit) / BW_HOURS
  )

# ====================== RAW DOTS (per-animal rate; same scale as curve) ======================
# Average across ALL animal×day series present in that hour
raw_trials_per_animal <- count_df %>%
  mutate(
    hour_int = floor(hour_bin),
    rate     = n_trials / BW_HOURS
  ) %>%
  group_by(hour_int) %>%
  summarise(
    n_series       = dplyr::n(),
    n_animals      = n_distinct(name),
    rate_per_animal= mean(rate),                  # average per-animal rate among those present
    se_rate        = sd(rate)/sqrt(n_series),
    .groups = "drop"
  ) %>%
  mutate(hour_mid = hour_int + 0.5)

# ====================== PLOT: population vs per-animal dots ======================
p_trials_rate <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "lightgrey", alpha = 0.3) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "paleturquoise3", alpha = 0.2) +
  geom_ribbon(
    data = pop_trials,
    aes(x = hour_bin, ymin = rate_lo, ymax = rate_hi),
    alpha = 0.2, fill = "royalblue3"
  ) +
  geom_line(
    data = pop_trials,
    aes(x = hour_bin, y = rate_per_hour),
    color = "royalblue3", linewidth = 1
  ) +
  geom_linerange(
    data = raw_trials_per_animal,
    aes(x = hour_mid, ymin = rate_per_animal - se_rate, ymax = rate_per_animal + se_rate),
    color = "black", linewidth = 0.3, alpha = 0.8
  ) +
  geom_point(
    data = raw_trials_per_animal,
    aes(x = hour_mid, y = rate_per_animal, size = n_animals),
    color = "black", alpha = 0.85
  ) +
  scale_size_continuous(name = "# animals contributing", range = c(1, 5)) +
  scale_x_continuous(breaks = seq(0, 24, by = 2), limits = c(0, 24)) +
  labs(
    x = "hours from light onset",
    y = sprintf("trials per hour per animal (bin = %.3f h)", BW_HOURS)
  ) +
  theme_minimal()

print(p_trials_rate)

# ====================== FACETS: per-animal production curves ======================
grid_by_animal_trials <- tidyr::expand_grid(
  name     = levels(count_df$name),
  hour_bin = seq(0, 24, by = 0.05),
  session  = count_df$session[1]        # dummy; excluded when predicting below
)
per_animal_trials <- grid_by_animal_trials %>%
  mutate(
    rate_per_hour = exp(predict(m_trials, newdata = cur_data_all(), type = "link")) / BW_HOURS
  )

raw_trials_by_animal <- count_df %>%
  mutate(hour_mid = hour_bin + BW_HOURS/2,
         rate_per_hour = n_trials / BW_HOURS)

p_trials_facets <- ggplot() +
  annotate("rect", xmin = 12, xmax = 24, ymin = -Inf, ymax = Inf,
           fill = "lightgrey", alpha = 0.3) +
  annotate("rect", xmin = 6.5, xmax = 8.5, ymin = -Inf, ymax = Inf,
           fill = "paleturquoise3", alpha = 0.2) +
  geom_line(
    data = per_animal_trials,
    aes(x = hour_bin, y = rate_per_hour),
    linewidth = 0.8, color = "black", alpha = 0.75
  ) +
  geom_point(
    data = raw_trials_by_animal,
    aes(x = hour_mid, y = rate_per_hour),
    size = 0.7, alpha = 0.7
  ) +
  facet_wrap(~ name, ncol = 4, scales = "fixed") +
  scale_x_continuous(breaks = seq(0, 24, by = 4), limits = c(0, 24)) +
  labs(
    title = "Flashes: trial production rate by time of day (per animal)",
    x = "hours from light onset",
    y = "trials per hour"
  ) +
  theme_minimal(base_size = 12)

print(p_trials_facets)



