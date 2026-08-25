"""
Pre-game projections for all five starter props.

WHAT THIS IS FOR. Not to beat the market -- ten hypotheses tested against
real closing lines found no edge that survived scrutiny. This is a
REFERENCE NUMBER, so that a discrepancy against a posted line is a genuine
disagreement worth judging rather than an artefact of model error.

DESIGN DECISIONS, each forced by something that went wrong earlier:

  REGRESSION, NOT SIMULATION. Scored head to head on 900 held-out starts
  with no lookahead, the simulation failed to beat predicting the mean
  (RMSE 3.8967 against a 3.8964 baseline) while an eleven-feature
  regression reached 3.6506. All of the simulation's apparent skill came
  from being handed each start's realised pitches-per-PA.

  STRICT AS-OF DISCIPLINE. Every feature is computed from games STRICTLY
  BEFORE the projection date. There is one code path, so a backtest number
  is automatically achievable live -- the lookahead bug existed precisely
  because backtesting and projection used different harnesses.

  EMPIRICAL DISTRIBUTIONS, NOT SMOOTH ONES. Outs pile up at inning
  boundaries: 21.5% of starts end at exactly 18 outs against 6.4% at 17,
  and the spikes SURVIVE conditioning on pitcher quality. Converting a
  mean to P(over) through a normal misprices by 6-12 points. So the
  conversion uses the empirical residual distribution instead, which
  represents the lumpiness for free.

  CALIBRATION FITTED ON REALISED OUTCOMES. Thousands of starts, never on
  collected market lines -- those are scarce and must stay a clean test.

WHAT IT DELIBERATELY DOES NOT DO. It does not decide whether to bet. It
reports a probability and the fair price, and the discrepancy against a
posted line is left for human judgement, which is where the information
the model cannot see -- injury news, a manager's comments, weather at
first pitch -- actually enters.
"""
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from shim import storage

# prop -> (box-score column, None means derive outs from innings_pitched)
PROPS = {"outs": None, "strikeouts": "strikeouts", "hits": "hits_allowed",
         "walks": "walks_allowed", "runs": "runs_allowed"}

# Lines the market actually posts, so calibration is checked where it is used.
PROP_LINES = {
    "outs": [11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5],
    "strikeouts": [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
    "hits": [2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
    "walks": [0.5, 1.5, 2.5, 3.5],
    "runs": [0.5, 1.5, 2.5, 3.5, 4.5],
}

# `month` IS DELIBERATELY ABSENT. It was included, and it reproduced the
# exact failure that killed `season_index` in the hook model: a linear
# trend term fitted on months 3-5 and then EXTRAPOLATED to 6-8, dragging
# every projection down. Outs were predicted at 15.214 against a realised
# 15.631, a -0.418 bias, even though the target had not drifted (the fit
# period actually ran HIGHER at 15.782).
#
# Any feature that must be extrapolated to be used is a liability. If a
# seasonal effect is wanted later, encode it as a bounded categorical, and
# make it beat this on held-out data before it goes in.
# BULLPEN STATE IS DELIBERATELY ABSENT, and so is `month`.
#
# The bullpen idea was that a pen worn down over the previous few days
# buys the starter a longer leash. It was the strongest prior going in and
# it is dead, tested three ways:
#
#     vs the market residual, 136 starts      -1.41 sigma
#     vs outs, full population, 3,855 starts  +0.85 sigma
#     all variants (1d outs, 3d outs, 3d pitches)  under 0.9 sigma
#
# At 3,855 starts a real effect would be unmissable. Swinging the feature
# across its entire realistic range -- a rested pen at 13 outs to a
# hammered one at 46 -- moved projections by 0.15 outs against per-start
# noise of 3.6, roughly 4% of the uncertainty from the most extreme input
# possible. It was also the only field on the phone page that required
# looking something up, so it cost the user real effort for nothing.
#
# `month` was removed for the same reason `season_index` was removed from
# the old hook model: a linear trend fitted on months 3-5 and then
# EXTRAPOLATED to 6-8 dragged every projection down by 0.418 outs.
SHARED = ["rest", "n_starts", "home", "opp_runs"]


def _feature_names(prop):
    """Prop-specific form terms plus the shared context block.

    `recent_outs` is included for EVERY prop, not just outs: a start's
    length bounds every counting stat in it, and the empirical correlation
    between outs and strikeouts is +0.513.
    """
    names = [f"recent_{prop}", f"career_{prop}", "recent_outs",
             "recent_pitches"] + SHARED
    # dedupe: for the outs prop, recent_outs would otherwise appear twice --
    # once as the prop's own form term and once as the shared start-length
    # term. Least squares tolerates the duplicate column, but a rank
    # deficiency is not something to leave lying around on purpose.
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


class ProjectionModel:
    def __init__(self):
        self.fits = {}
        self.resid = {}

    # ---------------------------------------------------------------- data
    def build(self, conn, since="2026-01-01"):
        """One row per start, features strictly pre-game."""
        team_runs = defaultdict(list)
        for r in conn.execute(
                "SELECT team_id, game_date, runs_scored FROM team_game_stats"):
            d = storage.local_game_date(r["game_date"]) or str(r["game_date"])[:10]
            if r["runs_scored"] is not None:
                team_runs[r["team_id"]].append((d, r["runs_scored"]))
        for t in team_runs:
            team_runs[t].sort()

        starts = []
        for r in conn.execute(
                """SELECT p.game_id, p.player_id, p.player_name, p.team_id,
                          g.game_date, p.innings_pitched, p.pitch_count,
                          p.strikeouts, p.hits_allowed, p.walks_allowed,
                          p.runs_allowed, g.home_team_id, g.away_team_id
                   FROM pitcher_game_stats p JOIN games g USING(game_id)
                   WHERE p.is_starter = 1"""):
            d = storage.local_game_date(r["game_date"])
            o = storage.outs_from_innings_pitched(r["innings_pitched"])
            if not d or o is None:
                continue
            starts.append({
                "date": d, "gid": r["game_id"], "pid": r["player_id"],
                "name": r["player_name"], "team": r["team_id"],
                "outs": float(o), "pitches": float(r["pitch_count"] or 0),
                "strikeouts": float(r["strikeouts"] or 0),
                "hits": float(r["hits_allowed"] or 0),
                "walks": float(r["walks_allowed"] or 0),
                "runs": float(r["runs_allowed"] or 0),
                "opp": (r["away_team_id"] if r["home_team_id"] == r["team_id"]
                        else r["home_team_id"]),
                "home": 1.0 if r["home_team_id"] == r["team_id"] else 0.0})
        starts.sort(key=lambda x: x["date"])

        hist = defaultdict(list)
        rows = []
        for s in starts:
            h = hist[s["pid"]]
            if s["date"] >= since and len(h) >= 5:
                d0 = datetime.fromisoformat(s["date"])
                opp_r = [v for d, v in team_runs.get(s["opp"], []) if d < s["date"]]
                row = {"date": s["date"], "gid": s["gid"], "pid": s["pid"],
                       "name": s["name"],
                       "rest": float(min((d0 - datetime.fromisoformat(h[-1]["date"])).days, 12)),
                       "n_starts": float(len(h)),
                       "home": s["home"], "month": float(int(s["date"][5:7])),
                       "opp_runs": statistics.mean(opp_r[-30:]) if opp_r else 4.5,
                       "recent_pitches": statistics.mean(x["pitches"] for x in h[-5:]),
                       "recent_outs": statistics.mean(x["outs"] for x in h[-5:])}
                for p in PROPS:
                    row[f"recent_{p}"] = statistics.mean(x[p] for x in h[-5:])
                    row[f"career_{p}"] = statistics.mean(x[p] for x in h)
                    row[f"y_{p}"] = s[p]
                rows.append(row)
            hist[s["pid"]].append(s)
        return rows

    def features_for(self, conn, pid, team, opp, date, is_home):
        """Features for an UPCOMING start, from history strictly before
        `date`.

        Deliberately shares its definitions with build(): if projection and
        backtest ever diverge, a backtest number stops being achievable
        live, which is exactly how the pitches-per-PA lookahead survived
        undetected for so long.
        """
        h = []
        for r in conn.execute(
                """SELECT p.innings_pitched, p.pitch_count, p.strikeouts,
                          p.hits_allowed, p.walks_allowed, p.runs_allowed,
                          g.game_date
                   FROM pitcher_game_stats p JOIN games g USING(game_id)
                   WHERE p.player_id = ? AND p.is_starter = 1""", (pid,)):
            d = storage.local_game_date(r["game_date"])
            o = storage.outs_from_innings_pitched(r["innings_pitched"])
            if d and o is not None and d < date:
                h.append({"date": d, "outs": float(o),
                          "pitches": float(r["pitch_count"] or 0),
                          "strikeouts": float(r["strikeouts"] or 0),
                          "hits": float(r["hits_allowed"] or 0),
                          "walks": float(r["walks_allowed"] or 0),
                          "runs": float(r["runs_allowed"] or 0)})
        if len(h) < 5:
            return None
        h.sort(key=lambda x: x["date"])
        d0 = datetime.fromisoformat(date)
        opp_r = []
        for r in conn.execute(
                "SELECT game_date, runs_scored FROM team_game_stats WHERE team_id = ?",
                (opp,)):
            d = storage.local_game_date(r["game_date"]) or str(r["game_date"])[:10]
            if d < date and r["runs_scored"] is not None:
                opp_r.append((d, r["runs_scored"]))
        opp_r.sort()
        row = {"date": date, "pid": pid,
               "rest": float(min((d0 - datetime.fromisoformat(h[-1]["date"])).days, 12)),
               "n_starts": float(len(h)),
               "home": 1.0 if is_home else 0.0,
               "opp_runs": statistics.mean([v for _, v in opp_r[-30:]]) if opp_r else 4.5,
               "recent_pitches": statistics.mean(x["pitches"] for x in h[-5:]),
               "recent_outs": statistics.mean(x["outs"] for x in h[-5:])}
        for pr in PROPS:
            row[f"recent_{pr}"] = statistics.mean(x[pr] for x in h[-5:])
            row[f"career_{pr}"] = statistics.mean(x[pr] for x in h)
        return row

    # ----------------------------------------------------------------- fit
    def fit(self, rows):
        """Least squares per prop, plus the empirical residual distribution
        used to turn a mean into P(over)."""
        for prop in PROPS:
            F = _feature_names(prop)
            X = np.array([[r[f] for f in F] for r in rows], float)
            y = np.array([r[f"y_{prop}"] for r in rows], float)
            mu, sd = X.mean(0), X.std(0)
            sd[sd == 0] = 1.0
            beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), (X - mu) / sd],
                                       y, rcond=None)
            self.fits[prop] = (F, mu, sd, beta)
            # residuals kept RAW, so the spikes at inning boundaries survive
            self.resid[prop] = y - self.predict_mean(prop, rows)
        return self

    def predict_mean(self, prop, rows):
        F, mu, sd, beta = self.fits[prop]
        X = np.array([[r[f] for f in F] for r in rows], float)
        return np.c_[np.ones(len(X)), (X - mu) / sd] @ beta

    # --------------------------------------------------------------- price
    def fit_conditional(self, rows):
        """Empirical distribution of the ACTUAL stat, within bins of the
        projected mean.

        WHY NOT JUST ADD RESIDUALS TO THE MEAN. That was the first
        approach and it fails for outs specifically. The pile-ups sit at
        FIXED ABSOLUTE TOTALS -- 15 outs is five innings and 18 is six, for
        every pitcher alive -- but adding a pooled residual sample places
        them at a fixed DISTANCE FROM THE PROJECTION, so a pitcher
        projected at 13 gets a phantom spike at 13 and none at 15.

        That mattered most at exactly the lines it should. 15.5 and 18.5
        sit IMMEDIATELY ABOVE the two biggest spikes, so going over means
        clearing a full inning past the most common stopping point --
        making them the two lines on the board most sensitive to spike
        placement. They were the two that missed, by +0.057 and +0.077.

        Binning by projected mean and reading off the empirical
        distribution of real outcomes inside each bin keeps the spikes
        where they belong.
        """
        self.cond = {}
        for prop in PROPS:
            mu = self.predict_mean(prop, rows)
            y = np.array([r[f"y_{prop}"] for r in rows], float)
            order = np.argsort(mu)
            n_bins = max(4, min(10, len(rows) // 150))
            self.cond[prop] = []
            for chunk in np.array_split(order, n_bins):
                self.cond[prop].append((float(mu[chunk].mean()), y[chunk]))
        return self

    def p_over(self, prop, mean, line):
        """P(stat > line) from the empirical outcome distribution of the
        nearest projected-mean bin, so absolute spike positions survive."""
        bins = getattr(self, "cond", {}).get(prop)
        if not bins:
            sim = mean + self.resid[prop]
            return float((np.round(sim) > line).mean())
        centre, sample = min(bins, key=lambda b: abs(b[0] - mean))
        # shift the bin's outcomes by how far this start sits from the bin
        # centre, then ROUND -- the shift moves the level, the rounding
        # keeps outcomes on the integer grid the market settles on
        return float((np.round(sample + (mean - centre)) > line).mean())
