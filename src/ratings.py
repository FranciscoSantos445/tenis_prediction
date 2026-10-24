"""Elo and Glicko-2 ratings, shared by data_extraction.ipynb (features for training) and predict_matches.py (today's ratings),
and the rest features both compute the same way (rest()).

Besides tour-level matches, the ratings also learn from lower levels, so a player arriving on tour already has a rating:
Elo and Glicko-2 (rating + how uncertain it is) both learn from Challengers, tour qualifying and Futures.
On 2012-2026 test matches this improved log loss 0.5968 -> 0.5849 (matches with a newcomer: 0.5879 -> 0.5611).
"""
import math, re
from bisect import bisect_left
import numpy as np, pandas as pd

SURFACES = ['Hard', 'Clay', 'Grass', 'Carpet']
POOLS = ['Overall'] + SURFACES
K_SCALE = 200         # Elo K = K_SCALE / (matches played in the pool + 5) ** 0.4: fast for newcomers, stable for veterans
MARGIN_WEIGHT = 1.5   # K multiplier = 1 + MARGIN_WEIGHT * (2 * share of games won - 1); 6-0 6-0 = x2.5, walkovers = x1
LEVEL_START = {'G': 1600, 'M': 1500, 'A': 1400}      # first rating by the level of the first match (other tour levels: 1300)
LOWER_START = {'qual_chall': 1400, 'futures': 1300}  # first rating when the first match is a Challenger/qualifying or Futures
GLICKO_LEVELS = {'qual_chall', 'futures'}             # lower levels Glicko-2 learns from
Q, PHI0, SIGMA0, TAU = 173.7178, 350 / 173.7178, 0.06, 0.5  # Glicko-2 scale, starting deviation (RD 350), volatility, tau
ROUND_ORDER = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'RR': 4, 'R128': 5, 'R64': 6, 'R32': 7, 'R16': 8, 'QF': 9, 'SF': 10, 'F': 11}
ARCHIVE_END = 20260601  # Sackmann's Challenger files stop here; Tennis Explorer covers the Challengers from then on
GAVE_UP = r'RET|W/O|Walkover'  # score of a match the loser retired from or gave a walkover in
REST_DAYS = 30                 # window of rest()'s "matches in the last days"


def load_lower(data_set_2):
    """Lower-level matches in rating order: Sackmann archive (Challengers/qualifying before ARCHIVE_END, all Futures) and the
    Tennis Explorer Challengers since then (atp-tennis-daily-pull.ipynb writes both files)."""
    archive = pd.read_csv(f'{data_set_2}/atp_matches_lower_till_2026.csv', low_memory=False)
    archive = archive[(archive['kind'] == 'futures') | (archive['tourney_date'] < ARCHIVE_END)]
    try:
        lower = pd.concat([archive, pd.read_csv(f'{data_set_2}/atp_matches_lower_recent.csv')], ignore_index=True)
    except FileNotFoundError:
        lower = archive
    lower = lower.assign(surface=lower['surface'].str.capitalize(), r=lower['round'].map(ROUND_ORDER).fillna(5))
    return lower.sort_values(['tourney_date', 'r'], kind='stable').reset_index(drop=True)


def day_number(yyyymmdd):
    return (pd.Timestamp(str(int(yyyymmdd))) - pd.Timestamp('1900-01-01')).days


def margin(score):
    games = re.findall(r'(\d+)-(\d+)', str(score))
    won, lost = sum(int(a) for a, b in games), sum(int(b) for a, b in games)
    return 1 + MARGIN_WEIGHT * (2 * won / (won + lost) - 1) if won + lost else 1.0


def inflate(phi, sigma, days):
    """Glicko-2 deviation after `days` without matches: the rating gets less certain week by week, up to the starting RD."""
    return min(math.sqrt(phi * phi + sigma * sigma * max(days, 0) / 7), PHI0)


def _glicko_update(mu, phi, sigma, mu_o, phi_o, score):
    """Glicko-2 update after one match (Glickman's algorithm, one match per rating period)."""
    g = 1 / math.sqrt(1 + 3 * phi_o * phi_o / math.pi ** 2)
    e = 1 / (1 + math.exp(-g * (mu - mu_o)))
    v = 1 / (g * g * e * (1 - e))
    delta = v * g * (score - e)
    a = math.log(sigma * sigma)
    f = lambda x: math.exp(x) * (delta ** 2 - phi ** 2 - v - math.exp(x)) / (2 * (phi ** 2 + v + math.exp(x)) ** 2) - (x - a) / TAU ** 2
    A = a
    if delta ** 2 > phi ** 2 + v:
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU
    fa, fb = f(A), f(B)
    while abs(B - A) > 1e-6:
        C = A + (A - B) * fa / (fb - fa)
        fc = f(C)
        if fc * fb <= 0:
            A, fa = B, fb
        else:
            fa /= 2
        B, fb = C, fc
    sigma = math.exp(A / 2)
    phi = 1 / math.sqrt(1 / (phi * phi + sigma * sigma) + 1 / v)
    return mu + phi * phi * g * (score - e), phi, sigma


def rest(tour_days, tour_gave_up, lower_days, lower_gave_up, day):
    """A player's (days since his last match, matches in the last REST_DAYS days, injury break) before a match on `day`,
    counting every level, so a player coming back through Challengers isn't on a break. Days are day_number()s.
    - tour_days: his tour matches so far, sorted (earlier rounds of the same event included); tour_gave_up: he retired or
      gave a walkover in the last of them;
    - lower_days: his lower-level matches, sorted; only those before `day` count (their order within a day is unknown, as
      in rate()); lower_gave_up: the days he gave one up;
    - injury break: days since his last match if he gave that match up, else 0 (an injury layoff, not a rest)."""
    j = bisect_left(lower_days, day)
    last_tour, last_lower = (tour_days[-1] if tour_days else None), (lower_days[j - 1] if j else None)
    if last_tour is None and last_lower is None:
        return np.nan, 0, 0
    tour_latest = last_lower is None or (last_tour is not None and last_tour >= last_lower)
    since = day - (last_tour if tour_latest else last_lower)
    gave_up = tour_gave_up if tour_latest else last_lower in lower_gave_up
    recent = len(tour_days) - bisect_left(tour_days, day - REST_DAYS) + j - bisect_left(lower_days, day - REST_DAYS)
    return since, recent, since if gave_up else 0


def rate(tour, lower):
    """Ratings before and after every tour match (rows of `tour` in the order given, Player_1 = winner) and every player's
    latest state. A date's lower-level matches are rated after that date's tour matches, so a tour match never sees a
    result from its own day or later."""
    days = {d: day_number(d) for d in set(tour['tourney_date']) | set(lower['tourney_date'])}
    date = np.r_[tour['tourney_date'].to_numpy(), lower['tourney_date'].to_numpy()]
    group = np.r_[np.zeros(len(tour), int), np.ones(len(lower), int)]
    pos = np.r_[np.arange(len(tour)), np.arange(len(lower))]
    order = np.lexsort((pos, group, date))
    T = list(zip(tour['Player_1'], tour['Player_2'], tour['surface'], tour['tourney_level'].astype(str), tour['score'], tour['tourney_date']))
    L = list(zip(lower['winner_name'], lower['loser_name'], lower['surface'], lower['kind'], lower['score'], lower['tourney_date']))

    elo, played = {p: {} for p in POOLS}, {p: {} for p in POOLS}
    glicko = {}  # player -> [mu, phi, sigma, day number of the last match]
    out = {f'{who}_{p.lower()}_elo_{when}': np.zeros(len(tour)) for who in ('winner', 'loser') for p in POOLS for when in ('pre', 'post')}
    out.update({f'{who}_{k}_pre': np.zeros(len(tour)) for who in ('winner', 'loser') for k in ('glicko', 'rd')})
    for g, i in zip(group[order].tolist(), pos[order].tolist()):
        if g == 0:
            w, l, surface, level, score, d = T[i]
            start, with_glicko = LEVEL_START.get(level.upper(), 1300), True
        else:
            w, l, surface, kind, score, d = L[i]
            start, with_glicko = LOWER_START[kind], kind in GLICKO_LEVELS
        for p in (w, l):
            if p not in elo['Overall']:
                for pool in POOLS:
                    elo[pool][p], played[pool][p] = start, 0
        multiplier = margin(score)
        for pool in ['Overall'] + ([surface] if surface in SURFACES else []):
            rw, rl = elo[pool][w], elo[pool][l]
            expected = 1 / (1 + 10 ** ((rl - rw) / 400))
            new_w = rw + K_SCALE / (played[pool][w] + 5) ** 0.4 * multiplier * (1 - expected)
            new_l = rl - K_SCALE / (played[pool][l] + 5) ** 0.4 * multiplier * (1 - expected)
            if g == 0:
                p_ = pool.lower()
                out[f'winner_{p_}_elo_pre'][i], out[f'loser_{p_}_elo_pre'][i] = rw, rl
                out[f'winner_{p_}_elo_post'][i], out[f'loser_{p_}_elo_post'][i] = new_w, new_l
            elo[pool][w], elo[pool][l] = new_w, new_l
            played[pool][w] += 1
            played[pool][l] += 1
        if with_glicko:
            today = days[d]
            for p in (w, l):
                state = glicko.setdefault(p, [(start - 1500) / Q, PHI0, SIGMA0, today])
                state[1], state[3] = inflate(state[1], state[2], today - state[3]), today
            (mw, pw, sw, _), (ml, pl, sl, _) = glicko[w], glicko[l]
            if g == 0:
                out['winner_glicko_pre'][i], out['loser_glicko_pre'][i] = mw * Q + 1500, ml * Q + 1500
                out['winner_rd_pre'][i], out['loser_rd_pre'][i] = pw * Q, pl * Q
            glicko[w][:3] = _glicko_update(mw, pw, sw, ml, pl, 1.0)
            glicko[l][:3] = _glicko_update(ml, pl, sl, mw, pw, 0.0)

    state = pd.DataFrame({f'elo_{p.lower()}': pd.Series(elo[p]) for p in POOLS})
    gl = pd.DataFrame.from_dict(glicko, orient='index', columns=['mu', 'phi', 'sigma', 'day'])
    state = state.join(pd.DataFrame({'glicko': gl['mu'] * Q + 1500, 'glicko_phi': gl['phi'], 'glicko_sigma': gl['sigma'], 'glicko_day': gl['day']}))
    state.index.name = 'player'
    return pd.DataFrame(out, index=tour.index), state


if __name__ == '__main__':  # self-checks: Glicko-2 on Glickman's worked example (rating 1500, RD 200 vs 1400 / RD 30, a win), rest()
    mu, phi, sigma = _glicko_update(0, 200 / Q, 0.06, (1400 - 1500) / Q, 30 / Q, 1.0)
    assert abs(mu * Q + 1500 - 1563.6) < 0.1 and abs(phi * Q - 175.4) < 0.1, (mu * Q + 1500, phi * Q)
    # rest(): tour matches 40 and 10 days ago (gave the last one up), a Challenger 5 days ago and one today (not counted yet)
    assert rest([60, 90], True, [95, 100], set(), 100) == (5, 2, 0)
    assert rest([60, 90], True, [], set(), 100) == (10, 1, 10)
    assert rest([], False, [95], {95}, 100) == (5, 1, 5) and np.isnan(rest([], False, [], set(), 100)[0])
    print('ok')
