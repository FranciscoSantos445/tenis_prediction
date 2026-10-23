"""Challenger results (main draw + qualifying) from tennisexplorer.com, used by atp-tennis-daily-pull.ipynb.

Sackmann's Challenger files stop in June 2026 (the archive), so the ratings in data_extraction.ipynb get the Challengers
played since then from here. One request per second; finished events and known players are cached between runs, so a
weekly update only fetches that week's events and new players.
"""
import html, re, time, unicodedata
import pandas as pd, requests

BASE = 'https://www.tennisexplorer.com'
HEADERS = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'}
SURFACE = {'Clay': 'Clay', 'Hard': 'Hard', 'Indoors': 'Hard', 'Grass': 'Grass'}
ROUND = {'1R': 'R32', '2R': 'R16', 'R16': 'R16', 'QF': 'QF', 'SF': 'SF', 'F': 'F',
         'Q-1R': 'Q1', 'Q-2R': 'Q2', 'Q-3R': 'Q3', 'Q-R16': 'Q1', 'Q-QF': 'Q2', 'Q-F': 'Q3'}
_last_request = [0.0]


def _get(path):
    time.sleep(max(0.0, 1 - (time.time() - _last_request[0])))
    r = requests.get(BASE + path, headers=HEADERS, timeout=60)
    _last_request[0] = time.time()
    r.raise_for_status()
    return r.text


def _text(s):
    return html.unescape(re.sub(r'<[^>]+>', '', s)).replace('\xa0', ' ').strip()


def norm(s):
    return ' '.join(re.sub(r'[^a-z]+', ' ', unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()).split())


def challengers(year):
    """Challenger events in a year's calendar: url, name, start (yyyymmdd), surface."""
    events = []
    for row in re.findall(r'<tr[^>]*>((?:(?!</tr>).)*?<th class="t-name".*?)</tr>', _get(f'/calendar/atp-men/{year}/'), re.S):
        url = re.search(r'href="(/[^"]+/\d{4}/atp-men/)"', row).group(1)
        if 'challenger' not in url:
            continue
        d, m, y = re.search(r'(\d\d)\.(\d\d)\.<br\s*/?>(\d{4})', row).groups()
        strong = re.search(r'<strong>(.*?)</strong>', row, re.S).group(1)
        title = re.search(r'title="([^"]+)"', strong)
        surface = re.search(r'class="s-color"[^>]*><span title="([^"]+)"', row)
        events.append({'url': url, 'name': html.unescape(title.group(1)) if title else _text(strong), 'start': int(f'{y}{m}{d}'),
                       'surface': SURFACE.get(surface.group(1)) if surface else None})
    return events


def results(event, qualifying=False):
    """Finished singles matches of an event (winner listed first): date, round, winner/loser slug, score."""
    page = _get(event['url'] + ('?phase=qualification' if qualifying else ''))
    start = pd.Timestamp(str(event['start']))
    rows = {}
    for rid, second, body in re.findall(r'<tr id="r(\d+)(b?)"[^>]*>(.*?)</tr>', page, re.S):
        slug = re.search(r'href="/player/([^/"]+)/"', body)
        games = [re.match(r'\s*(\d*)', re.sub(r'<sup>.*?</sup>', '', c)).group(1) for c in re.findall(r'<td class="score">(.*?)</td>', body, re.S)]
        row = {'slug': slug.group(1) if slug else None, 'games': [int(g) for g in games if g]}
        if not second:
            date = re.search(r'class="first time"[^>]*>\s*(?:<span[^>]*>)?(\d\d)\.(\d\d)\.', body)
            rnd = re.search(r'<td title="[^"]*"[^>]*>([^<]+)</td>', body)
            if date:
                day = pd.Timestamp(year=start.year, month=int(date.group(2)), day=int(date.group(1)))
                row['date'] = day if day >= start - pd.Timedelta(days=10) else day.replace(year=start.year + 1)
            row['round'] = rnd.group(1).strip() if rnd else None
        rows.setdefault(rid, [None, None])[1 if second else 0] = row
    today = pd.Timestamp.today().normalize()
    out = []
    for w, l in rows.values():
        # finished = played before today, both players known and at least one set scored (walkovers have no score)
        if w and l and w.get('date') is not None and w['date'] < today and w['slug'] and l['slug'] and w['games'] and l['games']:
            out.append({'event': event['url'], 'tourney_name': event['name'], 'surface': event['surface'],
                        'tourney_date': int(w['date'].strftime('%Y%m%d')), 'round': ROUND.get(w['round'], w['round']),
                        'winner_slug': w['slug'], 'loser_slug': l['slug'],
                        'score': ' '.join(f'{a}-{b}' for a, b in zip(w['games'], l['games']))})
    return out


def player(slug):
    """Name as Tennis Explorer writes it (surname first), birth date (yyyymmdd or None) and country."""
    page = _get(f'/player/{slug}/')
    name = _text(re.search(r'<h3>(.*?)</h3>', page, re.S).group(1))
    born = re.search(r'Age: \d+ \((\d+)\. (\d+)\. (\d{4})\)', page)
    country = re.search(r'Country: ([^<]+)<', page)
    return {'slug': slug, 'te_name': name, 'dob': int(f'{born.group(3)}{int(born.group(2)):02d}{int(born.group(1)):02d}') if born else None,
            'country': country.group(1).strip() if country else None}


def update(matches_csv, players_csv, since, log=print):
    """Bring the cached Challenger matches (events starting on or after `since`, yyyymmdd) and their players up to date."""
    try:
        cached = pd.read_csv(matches_csv)
    except FileNotFoundError:
        cached = pd.DataFrame(columns=['event', 'fetched'])
    try:
        players = pd.read_csv(players_csv)
    except FileNotFoundError:
        players = pd.DataFrame(columns=['slug', 'te_name', 'dob', 'country'])
    today = pd.Timestamp.today().normalize()
    # an event is finished once a run happened more than 8 days after its start; those are not fetched again
    done = set(cached.loc[cached['fetched'] > cached['start'] + 8, 'event']) if len(cached) and 'start' in cached else set()
    events = [e for y in range(since // 10000, today.year + 1) for e in challengers(y)
              if since <= e['start'] <= int(today.strftime('%Y%m%d')) and e['url'] not in done]
    fresh = []
    for k, e in enumerate(events, 1):
        for qualifying in (False, True):
            fresh += [dict(m, start=e['start'], fetched=int(today.strftime('%Y%m%d'))) for m in results(e, qualifying)]
        if k % 10 == 0 or k == len(events):
            log(f'  events {k}/{len(events)}')
    keep = cached[~cached['event'].isin([e['url'] for e in events])] if len(cached) else cached
    matches = pd.concat([keep, pd.DataFrame(fresh)], ignore_index=True)
    new = sorted(set(matches['winner_slug']).union(matches['loser_slug']) - set(players['slug']))
    for k, slug in enumerate(new, 1):
        try:
            players = pd.concat([players, pd.DataFrame([player(slug)])], ignore_index=True)
        except Exception as err:  # a missing player page shouldn't stop the update; the slug is retried next run
            log(f'  player {slug}: {err}')
        if k % 100 == 0 or k == len(new):
            log(f'  players {k}/{len(new)}')
    matches.to_csv(matches_csv, index=False)
    players.to_csv(players_csv, index=False)
    return matches, players


def full_names(players, sackmann_players, ranking):
    """Tennis Explorer slug -> the name used in the match data: Sackmann's 'First Last' (same birth date and surname, or else
    the only Sackmann player with exactly the same words), else the current ranking's name (same birth date and a shared
    name), else Tennis Explorer's name with the last word (usually the first name) moved first."""
    full = (sackmann_players['name_first'].fillna('').astype(str) + ' ' + sackmann_players['name_last'].fillna('').astype(str)).str.strip()
    words = {}  # name words -> {(full name, birth date or None)}
    for n, born in zip(full, sackmann_players['dob']):
        words.setdefault(frozenset(norm(n).split()), set()).add((n, int(born) if pd.notna(born) else None))
    sk = {}  # (birth date, surname) -> [(first-name words, full name)]; twins share both, so the first name must fit too
    for r in sackmann_players.assign(full=full).dropna(subset=['dob']).itertuples():
        sk.setdefault((int(r.dob), norm(r.name_last)), []).append((set(norm(r.name_first).split()), r.full))
    rk = {}
    for name, born in zip(ranking['name'], pd.to_datetime(ranking['dob'], errors='coerce')):
        if pd.notna(born):
            rk.setdefault(int(born.strftime('%Y%m%d')), []).append(name)
    names, taken = {}, set(full)
    for p in players.itertuples():
        te = norm(p.te_name).split()
        dob = int(p.dob) if pd.notna(p.dob) else None
        # Tennis Explorer writes the surname first; try every split point for multi-word surnames
        hit = next((full for k in range(len(te) - 1, 0, -1) for first, full in sk.get((dob, ' '.join(te[:k])), []) if first & set(te[k:])), None)
        same_words = words.get(frozenset(te), ())
        if hit is None and len(same_words) == 1:
            n, born = next(iter(same_words))
            # Tennis Explorer often only knows the birth year (stored as 1 January), so compare years: same words but
            # born in another year is someone else (Joao Silva 1984 vs 2007)
            hit = n if dob is None or born is None or born // 10000 == dob // 10000 else None
        if hit is None and dob in rk:
            hit = next((n for n in rk[dob] if set(norm(n).split()) & set(te)), None)
        if hit is None:
            hit = ' '.join(p.te_name.split()[-1:] + p.te_name.split()[:-1])
            if hit in taken:  # a different player already has this name (two Joao Silvas): keep them apart
                hit = f'{hit} ({p.slug})'
        names[p.slug] = hit
        taken.add(hit)
    return names
