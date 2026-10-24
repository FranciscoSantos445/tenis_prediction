"""Predict ATP matches with the final model (data_training.ipynb, trained on every match in the data).

    python predict_matches.py "Alexander Bublik" "Fabian Marozsan" --surface Hard --level A --round R16
    python predict_matches.py "Bublik" "Marozsan" --odds 1.45 2.75   # + your bets' decimal odds for player a and player b
    python predict_matches.py --file ../predictions/matches.csv     # columns: player_a, player_b, surface, tourney_level, round
                                                                    #          (+ optional odds_a, odds_b)
    python predict_matches.py --recent 20260921                      # honest predictions of the matches played since then
    python predict_matches.py --check                                # rebuilt features == data_extraction's features
    python predict_matches.py --backtest 1000                        # the app's betting rule on 2012-2026 matches, from 1000

Player features are rebuilt from training_data_set/dataset_more_metrics.csv the same way data_extraction.ipynb computes
them, using every match up to the latest one in the data (run the daily pull + data_extraction first to be up to date).
Elo and Glicko-2 ratings come from training_data_set/player_ratings.csv, written by data_extraction with ratings.py.
"""
import argparse, math, os, re, unicodedata
from collections import Counter, defaultdict
import joblib, numpy as np, pandas as pd, xgboost as xgb
from sklearn.preprocessing import StandardScaler
import ratings

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SRC = os.path.join(ROOT, 'src')
SURFACES, LEVEL_START = ratings.SURFACES, ratings.LEVEL_START
CATEGORICAL = ['tourney_level', 'surface', 'round']
# p -> sigmoid(slope * logit(p)). The model before the lower-level ratings was slightly overconfident (slope 0.965); the
# current one isn't: a slope fitted on its 2012-2018 test predictions did no better on 2019-2026 than leaving it at 1.
CALIBRATION_SLOPE = 1.0
# Best-of-5 (Grand Slams): the model is not confident enough in the favourite; slope fitted on 2007-2012, which on 2012-2026
# Slam matches improved log loss 0.5085 -> 0.5079 (the same for best-of-3 made it worse)
SLAM_SLOPE = 1.27
GAVE_UP = re.compile(ratings.GAVE_UP)  # retired injured or gave a walkover
# Risk manager (predict_app.py and --backtest). The model alone is less accurate than the bookmaker (2012-2026, same matches:
# log loss 0.592 vs 0.582), so a bet is judged on a mix: MODEL_WEIGHT of the model's chance, the rest the bookmaker's (margin
# removed). The weight was fitted on 2007-2012 matches; on 2012-2026 the mix was no better than the bookmaker alone.
MODEL_WEIGHT = 0.14
MIN_ODDS, MAX_ODDS = 1.20, 3.00             # no bets below 1.20, nor on long shots above 3.00 (high variance)
KELLY, MAX_BET, MAX_DAY = 0.25, 0.01, 0.03  # quarter-Kelly, at most 1% of the bankroll on one bet and 3% on one day


def calibrate(p, best_of=3):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    slope = np.where(np.asarray(best_of) == 5, SLAM_SLOPE, CALIBRATION_SLOPE)
    return 1 / (1 + np.exp(-slope * np.log(p / (1 - p))))


def _norm(s):
    return ' '.join(re.sub(r'[^a-z]+', ' ', unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()).split())


class Predictor:
    def __init__(self, before=None):
        """Player states from all matches in the data, or only those before `before` (yyyymmdd)."""
        m = pd.read_csv(f'{ROOT}/training_data_set/dataset_more_metrics.csv',
                        usecols=['tourney_date', 'surface', 'tourney_level', 'Player_1', 'Player_2', 'score',
                                 'Player_1_performance', 'Player_2_performance'])
        lower = ratings.load_lower(f'{ROOT}/data_set_2')
        if before:
            m, lower = m[m['tourney_date'] < before], lower[lower['tourney_date'] < before]
            state = ratings.rate(m, lower)[1]  # same code data_extraction uses
        else:
            state = pd.read_csv(f'{ROOT}/training_data_set/player_ratings.csv', index_col='player')
        self.ratings = state.to_dict('index')  # Elo (overall + per surface) and Glicko-2 of every player, incl. lower levels
        self.latest = int(m['tourney_date'].max())
        self.matches, self.last, self.perf = defaultdict(int), {}, defaultdict(list)
        self.surface_record, self.h2h, self.gave_up = defaultdict(lambda: [0, 0]), defaultdict(int), {}
        # for ratings.rest(): each player's tour and lower-level match days, whether he gave up his last tour match, the days
        # he gave up a lower-level one
        days = {d: ratings.day_number(d) for d in set(m['tourney_date']) | set(lower['tourney_date'])}
        self.tour_days, self.tour_gave_up, self.lower_days, self.lower_gave_up = defaultdict(list), {}, defaultdict(list), defaultdict(set)
        for w, l, d, g in zip(lower['winner_name'], lower['loser_name'], lower['tourney_date'], lower['score'].astype(str).str.contains(ratings.GAVE_UP)):
            self.lower_days[w].append(days[d])
            self.lower_days[l].append(days[d])
            if g:
                self.lower_gave_up[l].add(days[d])
        for r in m.itertuples(index=False):  # Player_1 is the winner
            gave_up = bool(GAVE_UP.search(str(r.score)))
            for side, player, won in ((1, r.Player_1, 1), (2, r.Player_2, 0)):
                self.matches[player] += 1
                self.last[player] = r.tourney_date
                self.tour_days[player].append(days[r.tourney_date])
                self.tour_gave_up[player] = gave_up and not won
                record = self.surface_record[(player, str(r.surface).strip())]
                record[0] += won
                record[1] += 1
                self.perf[player].append(getattr(r, f'Player_{side}_performance'))
            self.h2h[(r.Player_1, r.Player_2)] += 1
            if gave_up:
                self.gave_up[r.Player_2] = r.tourney_date

        players = pd.read_csv(f'{ROOT}/data_set_2/atp_players_till_2026.csv', usecols=['name_first', 'name_last', 'dob'])
        players['full_name'] = (players['name_first'].astype(str).str.strip() + ' ' + players['name_last'].astype(str).str.strip()).str.strip()
        dob = players.drop_duplicates('full_name').set_index('full_name')['dob'].dropna().astype(int).astype(str)
        self.dob = pd.to_datetime(dob, format='%Y%m%d', errors='coerce').to_dict()

        # Every currently ranked player can be searched (data_set_2/atp_current_ranking.csv, from the daily pull), even without
        # tour-level matches: they get the debut values (starting Elo, no form, 0 career matches) and the ranking's birth date
        self.alias = {}  # name in the current ranking -> same player's name in the match data (itself if he has no matches yet)
        if os.path.exists(f'{ROOT}/data_set_2/atp_current_ranking.csv'):
            known = set(self.ratings) | set(self.last)  # players with matches at any level
            by_norm = {_norm(p): p for p in known}
            by_dob = {(d, _norm(p).split()[-1], _norm(p)[0]): p for p, d in self.dob.items() if p in known and not pd.isna(d) and _norm(p)}
            # tour players data_extraction couldn't give a full name keep tennis-data's short one: 'Simakin I.' -> ('simakin', 'i')
            short = {p: (_norm(m[1]), _norm(m[2]).replace(' ', '')) for p in self.last if (m := re.match(r'^(.+)\s+([\w.-]+)\.$', p))}
            ranking = pd.read_csv(f'{ROOT}/data_set_2/atp_current_ranking.csv')
            keys = ranking['name'].map(_norm)
            fits = [[p for p, (sur, ini) in short.items() if k.endswith(' ' + sur) and k[:-len(sur)].replace(' ', '').startswith(ini)] for k in keys]
            claimed = Counter(p for f in fits for p in f)
            for r, key, f in zip(ranking.itertuples(), keys, fits):
                born = pd.to_datetime(r.dob, errors='coerce')
                # same player when spelled alike, with the same birth date, surname and initial (Khumoyun / Khumoun Sultanov, but not
                # twins), or the only ranked player who fits a short name
                name = by_norm.get(key) or by_dob.get((born, key.split()[-1], key[0])) or (f[0] if len(f) == 1 and claimed[f[0]] == 1 else None)
                if name is None:
                    name = r.name
                    self.dob.setdefault(name, born)
                self.alias[r.name] = name

        self.model = xgb.XGBClassifier()
        self.model.load_model(f'{SRC}/final_model.json')
        self.scaler = joblib.load(f'{SRC}/final_scaler.joblib')
        self.columns = list(self.scaler.feature_names_in_)

    def resolve(self, name):
        """Exact name, else a unique player whose name contains every word given ('Sinner', 'J Sinner'): first among the
        tour players active since 2025, then among every currently ranked player."""
        if name in self.last or name in self.ratings:
            return name
        if name in self.alias:
            return self.alias[name]
        words = _norm(name).split()
        fits = lambda p: all(w in _norm(p).split() or any(t.startswith(w) for t in _norm(p).split()) for w in words)
        for pool in ([p for p, last in self.last.items() if last >= 20250101], self.alias):
            hits = sorted({self.alias.get(p, p) for p in pool if fits(p)})
            if len(hits) == 1:
                return hits[0]
            if hits:
                raise ValueError(f"Several players match '{name}': {', '.join(hits[:6])}. Please be more specific.")
        raise ValueError(f"No player found for '{name}' (searching players active since 2025 and the current ranking).")

    def profile(self, p, surface, day, tourney_level='A'):
        """A player's values before a match on `day` (yyyymmdd), as the model sees them."""
        start = LEVEL_START.get(str(tourney_level).upper(), 1300)
        r = self.ratings.get(p, {})
        if pd.notna(r.get('glicko', np.nan)):  # Glicko-2 deviation grows with the weeks since the player's last match
            glicko, rd = r['glicko'], ratings.inflate(r['glicko_phi'], r['glicko_sigma'], ratings.day_number(day) - r['glicko_day']) * ratings.Q
        else:  # never rated by Glicko-2 (new, or Futures only): starting values, as in data_extraction
            glicko, rd = start, ratings.PHI0 * ratings.Q
        since, recent, injury = ratings.rest(self.tour_days.get(p, []), self.tour_gave_up.get(p, False), self.lower_days.get(p, []),
                                             self.lower_gave_up.get(p, set()), ratings.day_number(day))
        day = pd.Timestamp(str(day))
        wins, n = self.surface_record.get((p, surface), (0, 0))
        perf = self.perf.get(p, [])[-5:]
        born = self.dob.get(p)
        age = (day - born).days / 365.25 if born is not None and not pd.isna(born) else np.nan
        return {'elo': r.get('elo_overall', start),
                'surface_elo': r.get(f'elo_{surface.lower()}', start) if surface in SURFACES else 0.0,
                'glicko': glicko, 'rd': rd,
                'surface_wins': wins, 'surface_matches': n,
                'surface_win_rate': wins / n if n else np.nan,
                'days_since_last': since,  # every level
                'matches_last_X_days': recent, 'injury_break': injury,
                'recent_perf': np.mean(perf) if perf else np.nan,
                'age': age if 14 <= age <= 50 else np.nan,
                'career_matches': self.matches.get(p, 0),
                'days_since_retirement': (day - pd.Timestamp(str(self.gave_up[p]))).days if p in self.gave_up else np.nan}

    def features(self, a, b, surface, tourney_level, round_, day):
        pa, pb = self.profile(a, surface, day, tourney_level), self.profile(b, surface, day, tourney_level)
        wa, wb = self.h2h.get((a, b), 0), self.h2h.get((b, a), 0)
        row = {'elo_diff': pa['elo'] - pb['elo'], 'surface_elo_diff': pa['surface_elo'] - pb['surface_elo'],
               'h2h_ratio': (wa + 1) / (wa + wb + 2), 'glicko_diff': pa['glicko'] - pb['glicko'], 'player1_rd': pa['rd'], 'player2_rd': pb['rd'],
               'best_of': 5 if str(tourney_level).upper() == 'G' else 3}
        for k in ('surface_win_rate', 'days_since_last', 'recent_perf', 'age', 'career_matches', 'matches_last_X_days', 'injury_break'):
            row[f'player1_{k}'], row[f'player2_{k}'] = pa[k], pb[k]
        for c, v in (('tourney_level', tourney_level), ('surface', surface), ('round', round_)):
            row[f'{c}_{v}'] = 1.0
        X = pd.DataFrame([row]).reindex(columns=self.columns)
        dummies = [c for c in self.columns if c.split('_')[0] in ('tourney', 'round') or (c.startswith('surface_') and c != 'surface_elo_diff')]
        X[dummies] = X[dummies].fillna(0.0)
        return X

    def predict(self, a, b, surface='Hard', tourney_level='A', round_='R32', day=None):
        """Probability that player a beats player b (average of both orders, so P(a) + P(b) = 1)."""
        a, b = self.resolve(a), self.resolve(b)
        day = day or pd.Timestamp.today().strftime('%Y%m%d')
        p = lambda x, y: self.model.predict_proba(self.scaler.transform(self.features(x, y, surface, tourney_level, round_, day)))[0, 1]
        return a, b, float(calibrate((p(a, b) + 1 - p(b, a)) / 2, 5 if str(tourney_level).upper() == 'G' else 3))

    def assess(self, a, b, p, fair_a=None, surface='Hard', day=None):
        """How much to trust a prediction: ('green' | 'neutral' | 'amber', headline, notes).
        The notes are context only: on 2012-2026 test matches they did not lower the favourite's win rate (the model already uses them)."""
        day = day or pd.Timestamp.today().strftime('%Y%m%d')
        if fair_a is None:
            level, headline = 'neutral', 'Add the odds to compare with the bookmaker'
        elif (p > .5) != (fair_a > .5):
            level, headline = 'amber', 'Model and bookmaker disagree on the favourite'
        elif min(max(p, 1 - p), max(fair_a, 1 - fair_a)) >= .8:
            level, headline = 'green', 'Strong favourite: model and bookmaker agree (both 80%+)'
        else:
            level, headline = 'neutral', 'Model and bookmaker agree on the favourite'
        warnings = []
        for player in (a, b):
            pr = self.profile(player, surface, day)
            if pr['career_matches'] < 30:
                warnings.append(f"{player}: only {pr['career_matches']} tour-level matches, rating still settling")
            if pr['days_since_last'] > 60:
                warnings.append(f"{player}: {pr['days_since_last']:.0f} days since the last match")
            if pr['days_since_retirement'] <= 30:
                warnings.append(f"{player}: retired or gave a walkover {pr['days_since_retirement']:.0f} days ago")
        return level, headline, warnings


def risk(p, odds_a, odds_b, bankroll, open_today=0.0):
    """Bet or pass, given the model's chance p for player a, both decimal odds, the bankroll and the stakes already bet that
    day: dict(side=0 (a) | 1 (b) | None, stake, floor (lowest odds still worth taking), reason (why not), sides=[for a and b:
    model, fair (bookmaker's chance without its margin), chance (the mix), odds, ev (expected return per 1 staked)])."""
    fair_a = (1 / odds_a) / (1 / odds_a + 1 / odds_b)
    sides = [dict(model=m, fair=f, chance=MODEL_WEIGHT * m + (1 - MODEL_WEIGHT) * f, odds=o) for m, f, o in ((p, fair_a, odds_a), (1 - p, 1 - fair_a, odds_b))]
    for x in sides:
        x['ev'] = x['chance'] * x['odds'] - 1
    side = 0 if sides[0]['ev'] >= sides[1]['ev'] else 1
    x = sides[side]
    stake = round(min(KELLY * x['ev'] / (x['odds'] - 1) * bankroll, MAX_BET * bankroll, MAX_DAY * bankroll - open_today), 2)
    if x['ev'] <= 0:
        reason = f"no value on either player (best expected return {x['ev']:+.1%})"
    elif not MIN_ODDS <= x['odds'] <= MAX_ODDS:
        reason = f"the value is at odds {x['odds']:.2f}, outside {MIN_ODDS:.2f}-{MAX_ODDS:.2f}"
    elif MAX_DAY * bankroll - open_today < .01:
        reason = f'daily limit reached ({MAX_DAY:.0%} of the bankroll)'
    elif stake < .01:
        reason = 'edge too small for a stake'
    else:
        return dict(side=side, stake=stake, floor=max(MIN_ODDS, math.ceil(100 / x['chance']) / 100), reason='', sides=sides)
    return dict(side=None, stake=0.0, floor=None, reason=reason, sides=sides)


def recent(cutoff):
    """Train on matches before `cutoff` with the tuned settings, predict every match since (results unseen by the model)."""
    b = pd.read_csv(f'{ROOT}/training_data_set/tennis_matches_balanced.csv')
    info = pd.read_csv(f'{ROOT}/training_data_set/dataset_more_metrics.csv',
                       usecols=['tourney_name', 'tourney_date', 'round', 'Player_1', 'Player_2', 'score', 'Odd_1', 'Odd_2', 'PS_1', 'PS_2', 'best_of'])
    columns = list(joblib.load(f'{SRC}/final_scaler.joblib').feature_names_in_)
    numeric = [c for c in columns if c in b.columns]
    X = pd.get_dummies(b[numeric + CATEGORICAL], columns=CATEGORICAL).reindex(columns=columns, fill_value=False)
    train = (b['tourney_date'] < cutoff).values
    params = dict(line.split(': ') for line in open(f'{SRC}/best_params.txt').read().splitlines())
    params = {k: int(v) if k in ('max_depth', 'min_child_weight') else float(v) for k, v in params.items()}
    tested = xgb.XGBClassifier(); tested.load_model(f'{SRC}/logloss_model.json')
    scaler = StandardScaler().fit(X[train])
    model = xgb.XGBClassifier(objective='binary:logistic', tree_method='hist', device='cuda', eval_metric='logloss',
                              n_estimators=tested.best_iteration + 1, **params).fit(scaler.transform(X[train]), b['outcome'][train])
    p = model.predict_proba(scaler.transform(X))[:, 1]
    p_winner = calibrate((p[0::2] + 1 - p[1::2]) / 2, info['best_of'])  # balanced rows come in (winner first, loser first) pairs
    out = info.assign(p_winner=p_winner)[info['tourney_date'] >= cutoff]
    odds = out[['Odd_1', 'Odd_2']].apply(pd.to_numeric, errors='coerce').where(lambda o: o > 1)
    out = out.assign(bookmaker_p_winner=(1 / odds['Odd_1']) / (1 / odds['Odd_1'] + 1 / odds['Odd_2']))
    return out.rename(columns={'Player_1': 'winner', 'Player_2': 'loser'})  # Odd_* (Bet365), PS_* (Pinnacle): winner's, loser's


def backtest(bankroll, since=20120712):
    """The risk() rule on every match since `since` (default: those the tested model never saw) with a model trained only
    on the matches before it, at Bet365's and Pinnacle's prices. Each day's stakes come from the bankroll at the start of
    the day, highest expected return first."""
    out = recent(since)
    for book, name in (('Odd', 'Bet365'), ('PS', 'Pinnacle')):
        odds = out[[f'{book}_1', f'{book}_2']].apply(pd.to_numeric, errors='coerce')
        x = out.assign(o1=odds.iloc[:, 0], o2=odds.iloc[:, 1]).query('o1 > 1 and o2 > 1')
        money, peak, drop, bets, won, staked = bankroll, bankroll, 0.0, 0, 0, 0.0
        for _, day in x.groupby('tourney_date'):
            first = [(risk(r.p_winner, r.o1, r.o2, money), r) for r in day.itertuples()]
            open_today, profit = 0.0, 0.0
            for _, r in sorted(((d, r) for d, r in first if d['side'] is not None), key=lambda t: -t[0]['sides'][t[0]['side']]['ev']):
                d = risk(r.p_winner, r.o1, r.o2, money, open_today)  # player a is the winner
                if d['side'] is None:
                    break  # daily limit
                open_today += d['stake']
                profit += d['stake'] * (r.o1 - 1) if d['side'] == 0 else -d['stake']
                bets, won, staked = bets + 1, won + (d['side'] == 0), staked + d['stake']
            money += profit
            peak, drop = max(peak, money), max(drop, 1 - money / peak)
        print(f"{name}: {bets} bets on {len(x)} matches ({won / max(bets, 1):.0%} won), staked {staked:,.2f}, profit {money - bankroll:+,.2f} "
              f"({(money - bankroll) / max(staked, 1e-9):+.1%} of the stakes), bankroll {bankroll:,.2f} -> {money:,.2f}, worst drop {drop:.0%}")


def check():
    """The features rebuilt for the latest match in the data must equal the ones data_extraction computed for it."""
    info = pd.read_csv(f'{ROOT}/training_data_set/dataset_more_metrics.csv', usecols=['tourney_date', 'surface', 'tourney_level', 'round', 'Player_1', 'Player_2'])
    b = pd.read_csv(f'{ROOT}/training_data_set/tennis_matches_balanced.csv')
    i = len(info) - 1
    r = info.iloc[i]
    pr = Predictor(before=int(r.tourney_date))
    X = pr.features(r.Player_1, r.Player_2, r.surface, r.tourney_level, r['round'], r.tourney_date)
    expected = b.iloc[2 * i]  # the winner-first row of that match
    for c in X.columns:
        if c in expected.index:
            got, exp = X[c].iloc[0], expected[c]
            assert (pd.isna(got) and pd.isna(exp)) or np.isclose(got, exp), (c, got, exp)
    print(f"check OK: features of {r.Player_1} vs {r.Player_2} ({r.tourney_date}) match data_extraction")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('players', nargs='*')
    ap.add_argument('--surface', default='Hard'); ap.add_argument('--level', default='A'); ap.add_argument('--round', default='R32')
    ap.add_argument('--date', help='yyyymmdd, default today'); ap.add_argument('--file'); ap.add_argument('--recent', type=int); ap.add_argument('--check', action='store_true')
    ap.add_argument('--backtest', type=float, metavar='BANKROLL', help='bet the risk rule on 2012-2026 matches starting with this bankroll')
    ap.add_argument('--odds', nargs=2, type=float, metavar=('ODDS_A', 'ODDS_B'), help='decimal odds offered for player a and player b')
    args = ap.parse_args()
    if args.check:
        check()
    elif args.backtest:
        backtest(args.backtest)
    elif args.recent:
        out = recent(args.recent)
        path = f'{ROOT}/predictions/recent_since_{args.recent}.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_csv(path, index=False)
        k = out['bookmaker_p_winner'].notna()
        print(f"{len(out)} matches since {args.recent}: model right {(out.p_winner > 0.5).mean():.1%}, "
              f"bookmaker right {(out.bookmaker_p_winner[k] > 0.5).mean():.1%} (on {k.sum()} with odds) -> {path}")
    else:
        pr = Predictor()
        single = dict(player_a=args.players[0], player_b=args.players[1], surface=args.surface, tourney_level=args.level, round=args.round) if args.players else None
        if single and args.odds:
            single.update(odds_a=args.odds[0], odds_b=args.odds[1])
        matches = pd.read_csv(args.file) if args.file else pd.DataFrame([single])
        print(f"(player data up to {pr.latest})")
        for m in matches.to_dict('records'):
            a, b, p = pr.predict(m['player_a'], m['player_b'], m['surface'], m['tourney_level'], m['round'], args.date)
            print(f"{a:28s} {p:6.1%}  vs  {1 - p:6.1%} {b}")
            oa, ob = m.get('odds_a', np.nan), m.get('odds_b', np.nan)
            fair_a = (1 / oa) / (1 / oa + 1 / ob) if oa > 1 and ob > 1 else np.nan  # bookmaker's chance without its margin
            for player, chance, odds, fair in ((a, p, oa, fair_a), (b, 1 - p, ob, 1 - fair_a)):
                if odds > 1:
                    print(f"    bet on {player} @ {odds:.2f}: wins {chance:.1%} of the time per the model"
                          f"{f' ({fair:.1%} per the bookmaker)' if fair == fair else ''}, expected return {10 * (chance * odds - 1):+.2f} per 10 staked")
            level, headline, warnings = pr.assess(a, b, p, fair_a if fair_a == fair_a else None, m['surface'], args.date)
            print(f"    [{level}] {headline}" + ''.join(f"\n    note: {w}" for w in warnings))
        if 'odds_a' in matches:
            print("Note: on 2012-2026 matches the bookmaker's chance was the more accurate one, and betting whenever the model saw"
                  " value (expected return > 0) lost 10-17% of the stakes. Treat the model's number as a second opinion.")
