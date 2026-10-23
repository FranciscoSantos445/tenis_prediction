# tenis_prediction

Predicts ATP men's singles matches with XGBoost, trained on ~200,000 tour-level matches (1968–2026), and compares its
chances with the bookmakers' odds. Includes a desktop app and a command-line tool.

## Results

Test: July 2012 – October 2026, ~40,000 matches the model never saw (chronological 80/20 split).

| Matches | | ROC AUC | Log loss | Accuracy |
|---|---|---|---|---|
| All ~40,000 | **Model** | **0.755** | **0.585** | **68.3%** |
| All ~40,000 | Model before lower-level ratings | 0.741 | 0.597 | 67.2% |
| 35,830 with Bet365 odds | Model | 0.745 | 0.593 | 67.6% |
| 35,830 with Bet365 odds | Bookmaker (Bet365) | 0.758 | 0.582 | 67.5% |

- **Probabilities are honest:** when the model says 70%, the player wins about 70% of the time.
- **No betting edge:** on the same matches the bookmaker is more accurate, at every tournament level and also for newcomers.
  - Betting whenever the model saw value lost 10–17% of the stakes.
  - The app's betting rule (below), starting with 1,000 (`--backtest 1000`): Bet365 235 bets, 991.93 at the end (−1.5% of
    the stakes); Pinnacle 2,861 bets, 1,154.91 (+2.0%). Both are within chance of zero.
- **Use it as a second opinion,** not a betting system.

## Setup

Python 3.11 (pinned in `.python-version`):

```bash
pip install pandas numpy requests tqdm matplotlib seaborn scikit-learn xgboost optuna joblib openpyxl xlrd
```

- **App:** needs Tkinter (`sudo apt install python3-tk`).
- **Training:** uses an NVIDIA GPU.
- **Data:** not in the repo. Build it once with `python3 update.py` from `src/`, about 45 minutes. The first run
  downloads everything, including Challenger results since June 2026.
- **Notebooks:** run them from `src/`, which Jupyter and VS Code do by default.

## Updates

**Automatic.** The first time the app opens each day, it runs `src/update.py` in the background, about 3 minutes:

| Step | What it does |
|---|---|
| `atp-tennis-daily-pull.ipynb` | Downloads the current season's results and odds, new Challengers and the ranking. Past seasons and the archive are cached |
| `data_extraction.ipynb` | Rebuilds the player data and ratings, skipping the cells that only print or plot |
| `data_training.ipynb`, final-model cell | Refits the model on every match, once every 7 days |

- **While it runs:** the app works with the previous data and switches to the new data when the update finishes.
- **Log:** `training_data_set/update.log`.
- **By hand:** `python3 update.py`, or `python3 update.py --force` to run again on the same day.
- **Tour results lag:** tennis-data publishes a tournament's results once it ends, so the current week's tour matches
  arrive the following week.
- **Re-tuning** (every few months, or after changing features): delete `logloss_model.json`, `scaler.joblib`,
  `accuracy_model.json` and `accuracy_scaler.joblib` in `src/`, then run `data_training.ipynb` up to **Final model
  for real predictions** (about 10 minutes).

## Predicting (from `src/`)

```bash
python3 predict_app.py                                            # desktop app
python3 predict_matches.py "Bublik" "Marozsan" --surface Hard --level A --round R16 --odds 1.45 2.75
python3 predict_matches.py --file ../predictions/matches.csv     # player_a, player_b, surface, tourney_level, round (+ odds_a, odds_b)
python3 predict_matches.py --recent 20260921                      # predictions of matches since a date, model trained before it
python3 predict_matches.py --check                                # rebuilt features match data_extraction
python3 predict_matches.py --backtest 1000                        # the app's betting rule on 2012-2026, from 1000
```

Levels: `G` Grand Slam, `M` Masters, `A` ATP 250/500, `F` Finals, `D` Davis Cup. Rounds: `R128`…`R16`, `QF`, `SF`, `F`, `RR`.

**App:**
- **Search:** suggests names as you type, from every ranked player.
- **Odds:** for each player, the model's and the bookmaker's chance and the expected return.
- **Confidence badge:** green when both rate the same player 80%+, amber when they disagree on the favourite.
- **Bet slip:** the chance that a combined bet wins.
- **Upcoming matches:** tour matches of the next few days, from Tennis Explorer. Click one to fill the form; only the
  odds are left to enter.
- **Risk manager:** with both odds and your bankroll, BET or PASS for matches picked from Upcoming matches (green rows:
  bet). Rules (`risk()` in `predict_matches.py`):
  - **Chance:** 11% the model's, 89% the bookmaker's without its margin (the mix that fitted 2007–2012 best).
  - **Bet only if** the expected return is positive and the odds are between 1.20 and 3.00.
  - **Stake:** quarter-Kelly, at most 1% of the bankroll per bet and 3% per day. It also gives the lowest odds still
    worth taking.
  - **Expect PASS on almost every match:** in the backtest it bet on 235 of 35,830 matches.

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
  - **Betting rules tested:** a log-odds mix with the bookmaker and XGBoost with the odds as an input bet more often and
    lost more than the 11% mix.

## Project structure

```
src/
  atp-tennis-daily-pull.ipynb, data_extraction.ipynb, data_training.ipynb   steps 1-3
  predict_matches.py, predict_app.py      command line and app
  update.py                               daily update (run by the app)
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
