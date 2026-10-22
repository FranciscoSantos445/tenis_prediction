# tenis_prediction

Predicts the winner of ATP men's singles matches with an XGBoost model trained on about 200,000 tour-level matches from
1968 to 2026, and compares its chances with the bookmakers' odds. It comes with a desktop app and a command-line tool.

## Results

Tested on matches the model never saw during training: July 2012 to October 2026, about 40,000 matches
(chronological 80/20 split).

| | ROC AUC | Log loss | Accuracy |
|---|---|---|---|
| Model | 0.741 | 0.597 | 67.2% |
| Bookmakers (Bet365 odds) | 0.758 | 0.582 | 68.7% |

The bookmakers are still more accurate. The model's chances are honest, though: after a small calibration step, when it
says 70% the player wins about 70% of the time.

### Betting backtests (2012–2026, matches the model never saw)

- **Betting whenever the model saw value lost money:** backing a player whenever the model's chance was higher than the
  odds implied lost 8–15% of the stakes. Most of the model's apparent edges were cases where the market knew better.
- **The safest rule still didn't make money:** when the model and the bookmaker both gave the same player 80% or more,
  that player won 89.7% of 4,602 bets. The return was −2.8% at Bet365, −1.4% at Pinnacle and −0.1% at the best price
  available across bookmakers.

Use the model as a second opinion, not as a betting system.

## Setup

Python 3.11 (pinned in `.python-version` for pyenv):

```bash
pip install pandas numpy requests tqdm matplotlib seaborn scikit-learn xgboost optuna joblib openpyxl xlrd
```

- **App:** needs Tkinter; on Ubuntu, `sudo apt install python3-tk`.
- **Training:** set up for an NVIDIA GPU (`device='cuda'` in XGBoost).

The data isn't stored in the repo. Build it once before predicting (about 15 minutes) by running the first two
notebooks below. The notebooks use paths relative to `src/`, so run them from that folder; Jupyter and VS Code do this by
default.

## Weekly update

| Step | Notebook | Time | What it does |
|---|---|---|---|
| 1 | `src/atp-tennis-daily-pull.ipynb` | ~2 min | Downloads the latest results and data (see below) |
| 2 | `src/data_extraction.ipynb` | ~15 min | Merges the sources into one match list, builds the features, writes `training_data_set/` |
| 3 | `src/data_training.ipynb` | seconds | Run the cells up to and including **Final model for real predictions**: refits the final model on every match with the tuned settings |

Step 1 downloads:
- tennis-data.co.uk results and odds, 2000 to today;
- Jeff Sackmann's matches (1968 to May 2026) and player list;
- the current ATP ranking.

The cells after the final-model cell in step 3 are feature experiments; you don't need to run them.

**Re-tuning** (every few months, or after changing the features): delete `src/logloss_model.json`, `src/scaler.joblib`,
`src/accuracy_model.json` and `src/accuracy_scaler.joblib`, then run step 3. Optuna then searches the settings again
(50 trials per model) before the final model is refitted.

## Predicting

Run from `src/`.

### App

```bash
python3 predict_app.py
```

- **Player search:** suggests names as you type, from every currently ranked ATP player and everyone active on tour
  since 2024.
- **Odds:** enter the decimal odds you're offered to see each bet's chance, the bookmaker's chance and the expected
  return.
- **Confidence badge:**
  - *green:* the model and the bookmaker both give the same player 80% or more;
  - *amber:* they disagree on the favourite;
  - *neutral:* otherwise.
- **Notes:** flags for newcomers, long breaks and recent retirements.
- **Bet slip (combined bets):**
  - the chance that all picks win;
  - the model's fair combined odds;
  - the expected return at the combined odds you're offered.

### Command line

```bash
python3 predict_matches.py "Alexander Bublik" "Fabian Marozsan" --surface Hard --level A --round R16
python3 predict_matches.py "Bublik" "Marozsan" --odds 1.45 2.75   # decimal odds for player a and player b
python3 predict_matches.py --file ../predictions/matches.csv     # columns: player_a, player_b, surface, tourney_level, round
                                                                 #          (+ optional odds_a, odds_b)
python3 predict_matches.py --recent 20260921                      # honest predictions of the matches played since then
python3 predict_matches.py --check                                # rebuilt features == data_extraction's features
```

- **Levels:** `G` Grand Slam, `M` Masters 1000, `A` ATP 250/500, `F` Tour Finals, `D` Davis Cup.
- **Rounds:** `R128` … `R16`, `QF`, `SF`, `F`, `RR`.

### Players with no tour-level matches

Players without tour-level matches can still be searched, as long as they are in the current ranking. They are
predicted as debutants: starting Elo, no form, 0 career matches, and their age from the ranking's birth date. With so
little to go on, treat those predictions loosely.

## How it works

### Data

The two match sources overlap from 2000 to May 2026 and name players differently ("Sinner J." vs "Jannik Sinner").
`data_extraction.ipynb` handles this as follows:
- **Matching:** it finds the same match in both sources by player names and dates.
- **Names:** it learns which tennis-data name belongs to which player, year by year, because tennis-data reuses names
  ("Ruud C." was Christian Ruud, later Casper).
- **Merging:** it keeps one row per match, under Sackmann's full player names, with tennis-data's odds and Sackmann's
  serve statistics.

### Features

All features are computed from matches **before** the one being predicted.

| Feature | Description |
|---|---|
| `elo_diff` | Overall Elo difference |
| `surface_elo_diff` | Elo difference on the match's surface |
| `h2h_ratio` | Head-to-head wins, smoothed: (wins + 1) / (matches + 2) |
| `surface_win_rate` | Each player's career win rate on the surface |
| `days_since_last` | Days since each player's previous match |
| `recent_perf` | Each player's performance over their last 5 matches |
| `age` | Age on match day |
| `career_matches` | Tour-level matches played before this one |
| `tourney_level`, `surface`, `round` | One-hot encoded |

How Elo is computed:
- **K-factor:** shrinks with experience, `K = 200 / (matches + 5)^0.4`.
- **Margin of victory:** each result is weighted by the share of games won.
- **Starting rating:** depends on the level of the player's first match: Grand Slam 1600, Masters 1500, ATP 1400,
  other 1300.

### Model

- **Algorithm:** XGBoost, tuned with Optuna on log loss, with early stopping choosing the number of trees.
- **Tested models:** `logloss_model.json` and `accuracy_model.json` are trained on matches up to July 2012 and scored on
  the rest.
- **Final model:** `final_model.json` uses the same settings, retrained on every match; the app and command line use it.
- **Symmetric predictions:** each prediction averages both player orders, so the two chances add up to 100%.
- **Calibration:** chances are slightly shrunk (`p → sigmoid(0.965 · logit(p))`) because the raw model is a little
  overconfident.

## Project structure

```
src/
  atp-tennis-daily-pull.ipynb   step 1: download data
  data_extraction.ipynb         step 2: merge sources, build features
  data_training.ipynb           step 3: tune, test and refit the model
  predict_matches.py            prediction library + command line
  predict_app.py                desktop app
  final_model.json, final_scaler.joblib        model used for predictions
  logloss_model.json, scaler.joblib            tested model (its tree count is reused by the final model)
  accuracy_model.json, accuracy_scaler.joblib  tested model, tuned for accuracy
  best_params.txt                              tuned settings
data_set_2/          Sackmann matches, players, current ranking   (downloaded, not in git)
data_set_3/          tennis-data.co.uk matches                    (downloaded, not in git)
training_data_set/   merged matches and features                  (built, not in git)
predictions/         prediction output files
```

## Data sources

- **[tennis-data.co.uk](http://www.tennis-data.co.uk):** ATP results and betting odds, 2000 to today.
- **Jeff Sackmann's [tennis_atp](https://www.tennisabstract.com) data:** matches from 1968, player list, serve
  statistics. Licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The original GitHub
  repository was taken down in September 2026, so the data comes from the archived mirror
  [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive), pinned to one commit.
- **[Tennis Abstract ATP ranking](https://www.tennisabstract.com/reports/atpRankings.html):** current ranking with birth
  dates, used only for the player search.

Because Sackmann's data is licensed for non-commercial use, keep this project non-commercial.
