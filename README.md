# tenis_prediction

Predicts ATP men's singles matches with XGBoost, trained on ~200,000 tour-level matches (1968–2026), and compares its
chances with the bookmakers' odds. Includes a desktop app and a command-line tool.

## Results

Test: July 2012 – October 2026, ~40,000 matches the model never saw (chronological 80/20 split).

| | ROC AUC | Log loss | Accuracy |
|---|---|---|---|
| **Model** | **0.755** | **0.585** | **68.3%** |
| Model before lower-level ratings | 0.741 | 0.597 | 67.2% |
| Bookmakers (Bet365) | 0.758 | 0.582 | 68.7% |

- **Probabilities are honest:** when the model says 70%, the player wins about 70% of the time.
- **No betting edge:** the bookmakers are still more accurate.
  - Betting whenever the model saw value lost 8–15% of the stakes.
  - Even backing players that both the model and the bookmaker rated at 80% or more won 89.7% of bets but returned −2.8% at Bet365 (−0.1% at the best price available).
- **Use it as a second opinion,** not a betting system.

## Setup

Python 3.11 (pinned in `.python-version`):

```bash
pip install pandas numpy requests tqdm matplotlib seaborn scikit-learn xgboost optuna joblib openpyxl xlrd
```

- **App:** needs Tkinter (`sudo apt install python3-tk`).
- **Training:** uses an NVIDIA GPU.
- **Data:** not in the repo. Run steps 1–2 below once before predicting.
- **Notebooks:** run them from `src/`, which Jupyter and VS Code do by default.

## Weekly update

| Step | Notebook | Time | What it does |
|---|---|---|---|
| 1 | `atp-tennis-daily-pull.ipynb` | ~1 min (first run ~35 min) | Downloads results, odds, lower-level matches, new Challengers and the ATP ranking |
| 2 | `data_extraction.ipynb` | ~15 min | Merges the sources, computes the ratings and features |
| 3 | `data_training.ipynb` | seconds | Run the cells up to **Final model for real predictions** to refit the final model |

**Re-tuning** (every few months, or after changing features): delete `logloss_model.json`, `scaler.joblib`,
`accuracy_model.json` and `accuracy_scaler.joblib` in `src/`, then run step 3 (about 10 minutes).

## Predicting (from `src/`)

```bash
python3 predict_app.py                                            # desktop app
python3 predict_matches.py "Bublik" "Marozsan" --surface Hard --level A --round R16 --odds 1.45 2.75
python3 predict_matches.py --file ../predictions/matches.csv     # player_a, player_b, surface, tourney_level, round (+ odds_a, odds_b)
python3 predict_matches.py --recent 20260921                      # predictions of matches since a date, model trained before it
python3 predict_matches.py --check                                # rebuilt features match data_extraction
```

Levels: `G` Grand Slam, `M` Masters, `A` ATP 250/500, `F` Finals, `D` Davis Cup. Rounds: `R128`…`R16`, `QF`, `SF`, `F`, `RR`.

**App:**
- **Search:** suggests names as you type, from every ranked player.
- **Odds:** for each player, the model's and the bookmaker's chance and the expected return.
- **Confidence badge:** green when both rate the same player 80%+, amber when they disagree on the favourite.
- **Bet slip:** the chance that a combined bet wins.

Players without tour-level matches are rated from their Challenger, qualifying and Futures results.

## How it works

- **Data:** tennis-data.co.uk and Sackmann's data overlap from 2000 to May 2026 and name players differently ("Sinner J."
  vs "Jannik Sinner"). The extraction matches the same match in both sources and keeps one row, with the odds from
  tennis-data.
- **Ratings** (`ratings.py`, shared by the extraction notebook and the predictor):
  - **Learn from lower levels too:** Elo and Glicko-2 also count Challenger, qualifying and Futures matches, so players
    arrive on tour already rated.
  - **Elo:** K-factor shrinks with experience; results are weighted by the margin of victory.
  - **Glicko-2:** also tracks how uncertain each rating is, which grows during breaks.
  - **No look-ahead:** lower-level matches are rated after that day's tour matches.
- **Features**, all computed from matches *before* the one being predicted:
  - Elo, surface Elo and Glicko-2 differences;
  - rating uncertainty;
  - head-to-head;
  - surface win rate;
  - days since last match;
  - recent form;
  - age;
  - career matches;
  - level, surface, round.
- **Model:**
  - **Tuning:** XGBoost tuned with Optuna on log loss.
  - **Final model:** `final_model.json` is retrained on every match; the app and command line use it.
  - **Symmetric predictions:** each prediction averages both player orders.
  - **Alternatives tested:** LightGBM, CatBoost, HistGradientBoosting, logistic regression and model averages were no
    better.

## Project structure

```
src/
  atp-tennis-daily-pull.ipynb, data_extraction.ipynb, data_training.ipynb   steps 1-3
  predict_matches.py, predict_app.py      command line and app
  ratings.py                              Elo and Glicko-2
  tennis_explorer.py                      Challenger results since June 2026
  final_model.json, final_scaler.joblib   model used for predictions
  logloss_model.json, accuracy_model.json (+ scalers), best_params*.txt     tested models and tuned settings
data_set_2/, data_set_3/, training_data_set/   downloaded and built data (not in git)
```

## Data sources

- **[tennis-data.co.uk](http://www.tennis-data.co.uk):** results and odds, 2000 to today.
- **Jeff Sackmann's tennis_atp data:** tour matches from 1968, plus Challenger, qualifying and Futures matches.
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), via the archived mirror
  [Aneeshers/tennis-sackmann-archive](https://github.com/Aneeshers/tennis-sackmann-archive). It ends in May 2026.
- **[Tennis Explorer](https://www.tennisexplorer.com):** Challenger results since June 2026, at one request per second.
  It has no men's Futures results, so Futures stop at May 2026.
- **[Tennis Abstract](https://www.tennisabstract.com/reports/atpRankings.html):** current ranking, used for the player search.

Sackmann's data is licensed for non-commercial use, so keep this project non-commercial.
