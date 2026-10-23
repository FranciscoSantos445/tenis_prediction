"""Daily update, started by predict_app.py when it opens (or by hand: python update.py; add --force to run again today).

1. atp-tennis-daily-pull.ipynb: the current tennis-data season, new Challengers, the ranking (past seasons and the
   Sackmann archive are cached, so only what changed is downloaded);
2. data_extraction.ipynb: player data and ratings rebuilt (the cells that only print, plot or analyse are skipped);
3. every MODEL_EVERY_DAYS: data_training.ipynb's final-model cell refits the model on every match with the tuned
   settings (re-tuning stays a manual step, see the README).

The notebooks' code cells run here directly (no Jupyter needed); their saved outputs are left untouched.
"""
import datetime, json, os, sys, time

SRC = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(SRC, '..', 'training_data_set'))
STAMP, LOCK = os.path.join(DATA, 'last_update.txt'), os.path.join(DATA, 'update.lock')
MODEL_EVERY_DAYS = 7
SKIP = {'4dca65b1', '5bb0046e', '36482241', 'b2c3d4e5', 'dea33792', '797fcf53', 'a63d9626', 'daf97cdc', 'ce219079'}  # data_extraction: prints/plots/analysis


def needed():
    try:
        return open(STAMP).read().strip() != datetime.date.today().isoformat()
    except FileNotFoundError:
        return True


def run(notebook, cells=None):
    """Run a notebook's code cells (all, or only those at the given indexes) in one namespace."""
    namespace = {'__name__': 'update', 'display': print}
    for i, cell in enumerate(json.load(open(os.path.join(SRC, notebook)))['cells']):
        if cell['cell_type'] != 'code' or cell.get('id') in SKIP or (cells is not None and i not in cells):
            continue
        code = ''.join(line for line in cell['source'] if not line.lstrip().startswith(('%', '!')))  # no shell/pip magics
        start = time.time()
        exec(compile(code, f'{notebook} cell {i}', 'exec'), namespace)
        print(f'{notebook} cell {i}: {time.time() - start:.0f}s', flush=True)


def lock():
    """One update at a time (two app windows): False if another live update holds the lock."""
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            os.kill(int(open(LOCK).read()), 0)
            return False
        except (ValueError, ProcessLookupError):  # left by an update that crashed
            os.remove(LOCK)
            return lock()
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def main(force=False):
    if not (force or needed()):
        return print('Already updated today.')
    if not lock():
        return print('Another update is running.')
    try:
        start = time.time()
        os.chdir(SRC)  # the notebooks use paths relative to src/
        sys.path.insert(0, SRC)
        os.environ['MPLBACKEND'] = 'Agg'  # plots in the cells don't open windows
        run('atp-tennis-daily-pull.ipynb')
        run('data_extraction.ipynb')
        if time.time() - os.path.getmtime(os.path.join(SRC, 'final_model.json')) > MODEL_EVERY_DAYS * 86400:
            run('data_training.ipynb', cells=[1, 5, 7])  # imports, functions (main() doesn't run here), final model
        open(STAMP, 'w').write(datetime.date.today().isoformat())
        print(f'Update done in {(time.time() - start) / 60:.1f} min', flush=True)
    finally:
        os.remove(LOCK)


if __name__ == '__main__':
    main(force='--force' in sys.argv)
