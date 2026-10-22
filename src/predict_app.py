"""Tennis match predictor - desktop app on top of predict_matches.py.

    python predict_app.py
"""
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from predict_matches import Predictor, SURFACES

BG, CARD, INK, MUTED, LINE = '#f2f4f7', '#ffffff', '#1d2433', '#6b7280', '#e3e6eb'
NAVY, ACCENT, ACCENT_DARK = '#13213c', '#2f9e74', '#257f5d'
COLOR_A, COLOR_B = '#2f9e74', '#e07a5f'
GOOD, BAD, WARN = '#1f8a4c', '#c0392b', '#a86400'
BADGE = {'green': ('#e3f4ea', GOOD), 'neutral': ('#eef1f5', MUTED), 'amber': ('#fff3dc', WARN)}  # background, text
STRIP = {'green': GOOD, 'neutral': '#c5cbd3', 'amber': '#e0a030'}
FONT = 'Noto Sans'
LEVELS = {'ATP 250 / 500': 'A', 'Masters 1000': 'M', 'Grand Slam': 'G', 'Tour Finals': 'F', 'Davis Cup': 'D'}
ROUNDS = {'Round of 128': 'R128', 'Round of 64': 'R64', 'Round of 32': 'R32', 'Round of 16': 'R16',
          'Quarterfinal': 'QF', 'Semifinal': 'SF', 'Final': 'F', 'Round robin': 'RR'}
NOTE = ("The bookmaker's odds were more accurate than the model on 2012-2026 matches; betting on the model's value lost 8-15%. "
        "The safest rule, both at 80%+, won 90% of bets but lost 2.8% at Bet365 (about break-even at the best price on offer).")


class PlayerBox(ttk.Entry):
    """Entry that lists matching players under it while typing (Up/Down, Enter or click to pick, Esc to close)."""
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self.names, self.recent, self.popup = [], {}, None
        self.bind('<KeyRelease>', self._typed)
        self.bind('<Down>', lambda e: self._move(1))
        self.bind('<Up>', lambda e: self._move(-1))
        self.bind('<Return>', self._choose)
        self.bind('<Escape>', lambda e: self._hide())
        self.bind('<FocusOut>', lambda e: self.after(150, self._hide))
        self.winfo_toplevel().bind('<Configure>', lambda e: self._hide(), add='+')

    def set(self, text):
        self.delete(0, 'end')
        self.insert(0, text)

    def _typed(self, event):
        if event.keysym in ('Up', 'Down', 'Return', 'Escape', 'Tab'):
            return
        words = self.get().lower().split()
        def rank(name):  # words that start a name part first ('sin' -> Sinner), then the most recently active
            parts = name.lower().split()
            return (not all(any(p.startswith(w) for p in parts) for w in words), -self.recent.get(name, 0))
        self._show(sorted((n for n in self.names if all(w in n.lower() for w in words)), key=rank)[:8] if words else [])

    def _show(self, hits):
        if not hits:
            return self._hide()
        if self.popup is None:
            self.popup = tk.Toplevel(self)
            self.popup.overrideredirect(True)
            self.list = tk.Listbox(self.popup, font=(FONT, 11), bd=0, highlightthickness=1, highlightbackground=LINE, fg=INK,
                                   bg=CARD, selectbackground=ACCENT, selectforeground='white', activestyle='none')
            self.list.pack(fill='both', expand=True)
            self.list.bind('<ButtonRelease-1>', self._choose)
        self.list.delete(0, 'end')
        for name in hits:
            self.list.insert('end', '  ' + name)
        self.list.configure(height=len(hits))
        self.popup.update_idletasks()
        self.popup.geometry(f'{self.winfo_width()}x{self.list.winfo_reqheight()}+{self.winfo_rootx()}+{self.winfo_rooty() + self.winfo_height() + 2}')
        self.popup.deiconify()
        self.popup.lift()

    def _visible(self):
        return self.popup is not None and self.popup.winfo_viewable()

    def _move(self, step):
        if not self._visible():
            return
        current = self.list.curselection()
        i = min(max((current[0] if current else -1) + step, 0), self.list.size() - 1)
        self.list.selection_clear(0, 'end')
        self.list.selection_set(i)
        self.list.see(i)
        return 'break'

    def _choose(self, event=None):
        if not self._visible():
            return  # Enter without suggestions falls through to Predict
        picked = self.list.curselection()
        self.set(self.list.get(picked[0] if picked else 0).strip())
        self.icursor('end')
        self._hide()
        self.focus_set()
        return 'break'

    def _hide(self):
        if self.popup is not None:
            self.popup.withdraw()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Tennis Match Predictor')
        self.configure(bg=BG)
        self.minsize(1280, 600)
        self.predictor, self.names, self.p, self.current, self.slip = None, [], None, None, []
        self._style()
        self._header()
        body = tk.Frame(self, bg=BG)
        body.pack(fill='both', expand=True, padx=24, pady=(14, 10))
        left, right = tk.Frame(body, bg=BG), tk.Frame(body, bg=BG)
        left.pack(side='left', fill='both', expand=True)
        right.pack(side='left', fill='y', padx=(16, 0))
        self._form(left)
        self._result(left)
        self._slip_card(right)
        threading.Thread(target=self._load, daemon=True).start()
        self.after(150, self._wait_until_loaded)

    # ---------- look ----------
    def _style(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure('TCombobox', fieldbackground=CARD, background=CARD, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE,
                    arrowcolor=INK, padding=6, font=(FONT, 11))
        s.map('TCombobox', fieldbackground=[('readonly', CARD)], bordercolor=[('focus', ACCENT)])
        s.configure('TEntry', fieldbackground=CARD, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, padding=6)
        s.map('TEntry', bordercolor=[('focus', ACCENT)])
        s.configure('Accent.TButton', background=ACCENT, foreground='white', font=(FONT, 11, 'bold'), padding=(22, 10), borderwidth=0)
        s.map('Accent.TButton', background=[('disabled', '#a7d8c4'), ('active', ACCENT_DARK)])
        s.configure('Slip.TButton', background='#eef6f2', foreground=ACCENT_DARK, font=(FONT, 10, 'bold'), padding=(12, 6), borderwidth=0)
        s.map('Slip.TButton', background=[('disabled', BG), ('active', '#dcefe6')], foreground=[('disabled', '#b3b9c2')])
        self.option_add('*TCombobox*Listbox.font', (FONT, 11))
        self.option_add('*TCombobox*Listbox.selectBackground', ACCENT)

    def _label(self, parent, text='', size=10, bold=False, color=INK, **kw):
        return tk.Label(parent, text=text, bg=parent['bg'], fg=color, font=(FONT, size, 'bold' if bold else 'normal'), **kw)

    def _card(self, parent, title):
        card = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1, padx=22, pady=14)
        card.pack(fill='x', pady=(0, 12))
        self._label(card, title.upper(), 9, True, MUTED).grid(row=0, column=0, columnspan=6, sticky='w', pady=(0, 10))
        return card

    def _header(self):
        bar = tk.Frame(self, bg=NAVY, padx=24, pady=12)
        bar.pack(fill='x')
        ball = tk.Canvas(bar, width=38, height=38, bg=NAVY, highlightthickness=0)
        ball.create_oval(3, 3, 35, 35, fill='#d4e84a', outline='')
        ball.create_arc(-14, 6, 14, 32, start=-60, extent=120, style='arc', outline='white', width=2)
        ball.create_arc(24, 6, 52, 32, start=120, extent=120, style='arc', outline='white', width=2)
        ball.pack(side='left', padx=(0, 14))
        text = tk.Frame(bar, bg=NAVY)
        text.pack(side='left')
        self._label(text, 'Tennis Match Predictor', 17, True, 'white').pack(anchor='w')
        self.status = self._label(text, 'Loading player data...', 10, color='#aab4c8')
        self.status.pack(anchor='w')

    # ---------- inputs ----------
    def _form(self, parent):
        card = self._card(parent, 'Match')
        for c in (0, 2):
            card.columnconfigure(c, weight=1)
        self.a, self.b = self._player(card, 'Player A', 1, 0, COLOR_A), self._player(card, 'Player B', 1, 2, COLOR_B)
        self._label(card, 'vs', 13, True, MUTED).grid(row=2, column=1, padx=14)

        row = tk.Frame(card, bg=CARD)
        row.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(16, 0))
        self.surface = tk.StringVar(value='Hard')
        box = tk.Frame(row, bg=CARD)
        box.pack(side='left', padx=(0, 18))
        self._label(box, 'Surface', 9, color=MUTED).pack(anchor='w')
        seg = tk.Frame(box, bg=LINE, padx=1, pady=1)
        seg.pack(anchor='w', pady=(4, 0))
        for s in SURFACES:
            tk.Radiobutton(seg, text=s, value=s, variable=self.surface, indicatoron=0, width=7, font=(FONT, 10),
                           bg=CARD, fg=INK, selectcolor=ACCENT, activebackground='#e8f5ef', bd=0, relief='flat',
                           padx=6, pady=6, command=self._repaint_surface).pack(side='left', padx=(0, 1))
        self._seg = seg
        self.level = self._choice(row, 'Tournament', list(LEVELS), 15)
        self.round = self._choice(row, 'Round', list(ROUNDS), 13, default='Round of 32')
        self.date = self._entry(row, 'Date', 11, pd.Timestamp.today().strftime('%Y-%m-%d'))

        odds = tk.Frame(card, bg=CARD)
        odds.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(16, 0))
        self.odds_a = self._entry(odds, 'Odds on player A (optional)', 12)
        self.odds_b = self._entry(odds, 'Odds on player B (optional)', 12)
        self.button = ttk.Button(odds, text='Predict', style='Accent.TButton', command=self._predict, state='disabled')
        self.button.pack(side='right', anchor='s')
        self.bind('<Return>', lambda e: self._predict())
        self._repaint_surface()

    def _player(self, card, title, row, col, color):
        self._label(card, title, 9, True, color).grid(row=row, column=col, sticky='w')
        box = PlayerBox(card, font=(FONT, 12))
        box.grid(row=row + 1, column=col, sticky='ew', pady=(4, 0), ipady=3)
        return box

    def _choice(self, parent, title, values, width, default=None):
        box = tk.Frame(parent, bg=CARD)
        box.pack(side='left', padx=(0, 18))
        self._label(box, title, 9, color=MUTED).pack(anchor='w')
        c = ttk.Combobox(box, values=values, state='readonly', width=width, font=(FONT, 11))
        c.set(default or values[0])
        c.pack(anchor='w', pady=(4, 0))
        return c

    def _entry(self, parent, title, width, default=''):
        box = tk.Frame(parent, bg=CARD)
        box.pack(side='left', padx=(0, 18))
        self._label(box, title, 9, color=MUTED).pack(anchor='w')
        e = ttk.Entry(box, width=width, font=(FONT, 11))
        e.insert(0, default)
        e.pack(anchor='w', pady=(4, 0))
        return e

    def _repaint_surface(self):
        for b in self._seg.winfo_children():
            b.configure(fg='white' if b['value'] == self.surface.get() else INK)

    # ---------- result ----------
    def _result(self, parent):
        card = self._card(parent, 'Prediction')
        card.columnconfigure(1, weight=1)
        self.name_a = self._label(card, '', 13, True, COLOR_A)
        self.name_b = self._label(card, '', 13, True, COLOR_B)
        self.pct_a = self._label(card, '-', 26, True, COLOR_A)
        self.pct_b = self._label(card, '-', 26, True, COLOR_B)
        self.name_a.grid(row=1, column=0, sticky='w'); self.name_b.grid(row=1, column=2, sticky='e')
        self.pct_a.grid(row=2, column=0, sticky='w'); self.pct_b.grid(row=2, column=2, sticky='e')
        self.bar = tk.Canvas(card, height=14, bg=CARD, highlightthickness=0)
        self.bar.grid(row=3, column=0, columnspan=3, sticky='ew', pady=(4, 12))
        self.bar.bind('<Configure>', lambda e: self._draw_bar())
        actions = tk.Frame(card, bg=CARD)
        actions.grid(row=4, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        self.add_a = ttk.Button(actions, text='+ Add to bet slip', style='Slip.TButton', state='disabled', command=lambda: self._add_pick(0))
        self.add_b = ttk.Button(actions, text='+ Add to bet slip', style='Slip.TButton', state='disabled', command=lambda: self._add_pick(1))
        self.add_a.pack(side='left')
        self.add_b.pack(side='right')
        self.badge = tk.Label(actions, text='', bg=CARD, font=(FONT, 10, 'bold'), padx=12, pady=5)
        self.badge.pack(side='left', expand=True)

        self.bets = tk.Frame(card, bg=CARD)  # right under the bar: what matters most when betting
        self.bets.grid(row=5, column=0, columnspan=3, sticky='ew')
        self.stats = tk.Frame(card, bg=CARD)
        self.stats.grid(row=6, column=0, columnspan=3, sticky='ew', pady=(6, 0))
        for c, w in ((0, 1), (1, 0), (2, 1)):
            self.stats.columnconfigure(c, weight=w)
        self.warnings = self._label(card, '', 10, False, MUTED, wraplength=760, justify='left')
        self.warnings.grid(row=7, column=0, columnspan=3, sticky='w', pady=(8, 0))
        self.warnings.grid_remove()
        self.message = self._label(card, 'Choose two players and press Predict.', 10, color=MUTED, wraplength=760, justify='left')
        self.message.grid(row=8, column=0, columnspan=3, sticky='w', pady=(10, 0))

    def _draw_bar(self):
        self.bar.delete('all')
        w = self.bar.winfo_width()
        if self.p is None:
            self.bar.create_rectangle(0, 0, w, 14, fill=LINE, outline='')
            return
        x = int(w * self.p)
        self.bar.create_rectangle(0, 0, x, 14, fill=COLOR_A, outline='')
        self.bar.create_rectangle(x, 0, w, 14, fill=COLOR_B, outline='')
        self.bar.create_line(w // 2, 0, w // 2, 14, fill='white', width=2)

    def _show_stats(self, a, b, surface, day):
        for child in self.stats.winfo_children():
            child.destroy()
        pr = self.predictor
        pa, pb = pr.profile(a, surface, day), pr.profile(b, surface, day)
        f = lambda v, fmt='{:.0f}': '-' if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)
        rows = [('Elo rating', f(pa['elo']), f(pb['elo'])),
                (f'{surface} Elo', f(pa['surface_elo']), f(pb['surface_elo'])),
                (f'{surface} record', f"{pa['surface_wins']}-{pa['surface_matches'] - pa['surface_wins']}",
                 f"{pb['surface_wins']}-{pb['surface_matches'] - pb['surface_wins']}"),
                ('Head-to-head', str(pr.h2h.get((a, b), 0)), str(pr.h2h.get((b, a), 0))),
                ('Age', f(pa['age'], '{:.1f}'), f(pb['age'], '{:.1f}')),
                ('Career matches', f(pa['career_matches']), f(pb['career_matches'])),
                ('Days since last match', f(pa['days_since_last']), f(pb['days_since_last'])),
                ('Form (last 5, performance)', f(pa['recent_perf'], '{:.1f}'), f(pb['recent_perf'], '{:.1f}'))]
        for i, (name, va, vb) in enumerate(rows):
            shade = '#f7f8fa' if i % 2 == 0 else CARD
            for col, (text, anchor, color, bold) in enumerate(((va, 'w', INK, True), (name, 'center', MUTED, False), (vb, 'e', INK, True))):
                tk.Label(self.stats, text=text, bg=shade, fg=color, anchor=anchor, font=(FONT, 10, 'bold' if bold else 'normal'),
                         padx=10, pady=2).grid(row=i, column=col, sticky='ew')

    def _show_bets(self, a, b, p, odds_a, odds_b):
        for child in self.bets.winfo_children():
            child.destroy()
        fair_a = (1 / odds_a) / (1 / odds_a + 1 / odds_b) if odds_a and odds_b else None
        row = 0
        for player, chance, odds, fair, color in ((a, p, odds_a, fair_a, COLOR_A), (b, 1 - p, odds_b, None if fair_a is None else 1 - fair_a, COLOR_B)):
            if not odds:
                continue
            ev = 10 * (chance * odds - 1)
            line = tk.Frame(self.bets, bg='#eef6f2' if chance * odds > 1 else '#fbf1ee', padx=12, pady=7)
            line.grid(row=row, column=0, sticky='ew', pady=(0, 6))
            self.bets.columnconfigure(0, weight=1)
            self._label(line, f'Bet on {player} @ {odds:.2f}', 10, True, color).pack(side='left')
            sure = f'   wins {chance:.1%} of the time' + (f'  (bookmaker: {fair:.1%})' if fair is not None else '')
            self._label(line, sure, 10).pack(side='left')
            self._label(line, f'expected return per 10 staked: {ev:+.2f}', 10, True, GOOD if ev > 0 else BAD).pack(side='right')
            row += 1

    # ---------- actions ----------
    def _load(self):
        try:
            self.predictor = Predictor()
        except Exception as e:  # shown in the header
            self.load_error = e

    def _wait_until_loaded(self):
        if self.predictor is None and not hasattr(self, 'load_error'):
            return self.after(150, self._wait_until_loaded)
        if hasattr(self, 'load_error'):
            self.status.configure(text=f'Could not load the model or data: {self.load_error}', fg='#ffb4a8')
            return
        pr = self.predictor
        # ranked players (as the ranking spells them, also those without matches) + players active since 2024 not in the ranking
        self.names = sorted(set(pr.alias) | ({p for p, last in pr.last.items() if last >= 20240101} - set(pr.alias.values())))
        latest = pd.Timestamp(str(pr.latest)).strftime('%d %b %Y')
        self.status.configure(text=f'Player data up to {latest}  -  {len(pr.last):,} players  -  final model')
        for box in (self.a, self.b):
            box.names, box.recent = self.names, {n: pr.last.get(pr.alias.get(n, n), 0) for n in self.names}
        self.button.configure(state='normal')

    @staticmethod
    def _odds(entry):
        text = entry.get().strip().replace(',', '.')
        if not text:
            return None
        value = float(text)
        if value <= 1:
            raise ValueError
        return value

    def _predict(self):
        if self.predictor is None:
            return
        try:
            a, b = self.predictor.resolve(self.a.get().strip()), self.predictor.resolve(self.b.get().strip())
            if a == b:
                raise ValueError('Choose two different players.')
        except ValueError as e:
            return self.message.configure(text=str(e), fg=BAD)
        try:
            day = pd.Timestamp(self.date.get().strip()).strftime('%Y%m%d')
        except ValueError:
            return self.message.configure(text='The date should look like 2026-10-22.', fg=BAD)
        try:
            odds_a, odds_b = self._odds(self.odds_a), self._odds(self.odds_b)
        except ValueError:
            return self.message.configure(text='Odds are decimal numbers above 1, for example 1.85.', fg=BAD)

        surface = self.surface.get()
        _, _, p = self.predictor.predict(a, b, surface, LEVELS[self.level.get()], ROUNDS[self.round.get()], day)
        self.a.set(a); self.b.set(b)
        self.p = p
        self.name_a.configure(text=a); self.name_b.configure(text=b)
        self.pct_a.configure(text=f'{p:.1%}'); self.pct_b.configure(text=f'{1 - p:.1%}')
        self._draw_bar()
        self._show_stats(a, b, surface, day)
        self._show_bets(a, b, p, odds_a, odds_b)
        self.message.configure(text='' if odds_a or odds_b else 'Add the odds you are offered to see each bet\'s chance and expected return.', fg=MUTED)
        fair_a = (1 / odds_a) / (1 / odds_a + 1 / odds_b) if odds_a and odds_b else None
        level, headline, warnings = self.predictor.assess(a, b, p, fair_a, surface, day)
        bg, fg = BADGE[level]
        self.badge.configure(text=headline, bg=bg, fg=fg)
        if warnings:
            self.warnings.configure(text='\n'.join('Note: ' + w for w in warnings))
            self.warnings.grid()
        else:
            self.warnings.grid_remove()
        self.current = dict(players=(a, b), chances=(p, 1 - p), odds=(odds_a, odds_b), info=f'{surface} · {self.round.get()}', level=level)
        for button in (self.add_a, self.add_b):
            button.configure(state='normal')

    # ---------- bet slip ----------
    def _slip_card(self, parent):
        card = self._card(parent, 'Bet slip  -  combined bet')
        card.columnconfigure(0, weight=1)
        self.picks = tk.Frame(card, bg=CARD)
        self.picks.grid(row=1, column=0, sticky='ew')
        odds = tk.Frame(card, bg=CARD)
        odds.grid(row=2, column=0, sticky='ew', pady=(12, 0))
        self.combo_odds = self._entry(odds, 'Combined odds offered', 12)
        self.combo_odds.bind('<KeyRelease>', lambda e: self._refresh_slip())
        ttk.Button(odds, text='Clear slip', style='Slip.TButton', command=self._clear_slip).pack(side='right', anchor='s')
        self.slip_summary = tk.Frame(card, bg=CARD)
        self.slip_summary.grid(row=3, column=0, sticky='ew', pady=(12, 0))
        self.slip_summary.columnconfigure(0, weight=1)
        self.slip_note = self._label(card, '', 9, color=MUTED, wraplength=330, justify='left')
        self.slip_note.grid(row=4, column=0, sticky='w', pady=(10, 0))
        self._label(card, NOTE, 9, color=MUTED, wraplength=330, justify='left').grid(row=5, column=0, sticky='w', pady=(14, 0))
        self._refresh_slip()

    def _add_pick(self, i):
        c = self.current
        pick, opponent = c['players'][i], c['players'][1 - i]
        self.slip = [s for s in self.slip if {s['pick'], s['opponent']} != {pick, opponent}]  # one pick per match
        level = c['level']
        if level == 'green' and c['chances'][i] < .5:
            level = 'amber'  # the underdog of a strong favourite is not the safe side
        self.slip.append(dict(pick=pick, opponent=opponent, chance=c['chances'][i], odds=c['odds'][i], info=c['info'], level=level))
        self._refresh_slip()

    def _remove_pick(self, i):
        del self.slip[i]
        self._refresh_slip()

    def _clear_slip(self):
        self.slip = []
        self.combo_odds.delete(0, 'end')
        self._refresh_slip()

    def _refresh_slip(self):
        for frame in (self.picks, self.slip_summary):
            for child in frame.winfo_children():
                child.destroy()
        if not self.slip:
            self._label(self.picks, 'No picks yet. Predict a match and press "+ Add ... to bet slip" under the player you back.',
                        10, color=MUTED, wraplength=330, justify='left').pack(anchor='w')
            self.slip_note.configure(text='')
            return
        for i, s in enumerate(self.slip):
            row = tk.Frame(self.picks, bg='#f7f8fa', padx=10, pady=6)
            row.pack(fill='x', pady=(0, 6))
            tk.Frame(row, bg=STRIP[s['level']], width=4).pack(side='left', fill='y', padx=(0, 8))
            remove = tk.Label(row, text='×', bg=row['bg'], fg=MUTED, font=(FONT, 13), cursor='hand2', padx=4)
            remove.pack(side='right')
            remove.bind('<Button-1>', lambda e, i=i: self._remove_pick(i))
            self._label(row, f"{s['chance']:.1%}", 11, True, ACCENT_DARK).pack(side='right', padx=(8, 2))
            text = tk.Frame(row, bg=row['bg'])
            text.pack(side='left', fill='x', expand=True)
            self._label(text, s['pick'], 10, True).pack(anchor='w')
            self._label(text, f"vs {s['opponent']}  ·  {s['info']}" + (f"  ·  @ {s['odds']:.2f}" if s['odds'] else ''), 9, color=MUTED).pack(anchor='w')

        chance = float(np.prod([s['chance'] for s in self.slip]))
        lines = [(f'All {len(self.slip)} picks win (model)', f'{chance:.1%}', INK), ("Model's fair combined odds", f'{1 / chance:.2f}', INK)]
        flagged = sum(s['level'] == 'amber' for s in self.slip)
        if flagged:
            lines.append(('Picks flagged amber', str(flagged), WARN))
        if all(s['odds'] for s in self.slip):
            lines.append(('Your single odds multiplied', f"{np.prod([s['odds'] for s in self.slip]):.2f}", MUTED))
        try:
            odds = self._odds(self.combo_odds)
        except ValueError:
            odds = None
            lines.append(('Combined odds', 'a number above 1, e.g. 4.50', BAD))
        if odds:
            ev = 10 * (chance * odds - 1)
            lines += [("Bookmaker's implied chance", f'{1 / odds:.1%}', INK), ('Expected return per 10 staked', f'{ev:+.2f}', GOOD if ev > 0 else BAD)]
        for r, (name, value, color) in enumerate(lines):
            self._label(self.slip_summary, name, 10, color=MUTED).grid(row=r, column=0, sticky='w', pady=1)
            self._label(self.slip_summary, value, 12 if r == 0 else 10, True, color).grid(row=r, column=1, sticky='e', pady=1)
        self.slip_note.configure(text=f"Assumes the matches are independent. A combined bet multiplies the bookmaker's margin: "
                                      f"{len(self.slip)} picks at a typical 6% each carry about {1.06 ** len(self.slip) - 1:.0%}.")


if __name__ == '__main__':
    App().mainloop()
