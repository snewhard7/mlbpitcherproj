"""
Pull the last few days of completed games into data.db.

Runs in GitHub Actions every morning. Deliberately narrow: box scores,
games and probable pitchers from BALLDONTLIE only.

WHY NO STATCAST. The projection model uses none of it -- tested three
separate ways (inside the simulation, as a k_rate correction, and as
direct regression columns) and it moved every prop by under 0.35%. It is
also the slowest and least reliable step, which makes it the most likely
thing to break an unattended job. The xwOBA luck context DOES come from
Statcast, but it is context rather than an input, and it refreshes
whenever a full local export is pushed.

Safe to re-run over dates already present -- rows are replaced, not
duplicated.

    python scripts/update_data.py            # last 4 days
    python scripts/update_data.py 2026-08-01 2026-08-25
"""
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

API = "https://api.balldontlie.io/mlb/v1"
DB = Path(__file__).parent.parent / "data.db"
KEY = os.environ.get("BALLDONTLIE_API_KEY", "")


def get(path, **params):
    """One GET with retries. The free tier rate-limits, so back off rather
    than failing the whole run on a single 429."""
    out, cursor = [], None
    while True:
        p = dict(params)
        p["per_page"] = 100
        if cursor:
            p["cursor"] = cursor
        url = f"{API}/{path}?" + urllib.parse.urlencode(p, doseq=True)
        req = urllib.request.Request(url, headers={"Authorization": KEY})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    import json
                    body = json.load(r)
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 2 ** attempt
                print(f"    retry in {wait}s ({type(exc).__name__})", flush=True)
                time.sleep(wait)
        out.extend(body.get("data") or [])
        cursor = (body.get("meta") or {}).get("next_cursor")
        if not cursor:
            return out


def main():
    if not KEY:
        print("BALLDONTLIE_API_KEY is not set")
        return 1
    if len(sys.argv) > 2:
        start, end = sys.argv[1], sys.argv[2]
    else:
        end = date.today().isoformat()
        start = (date.today() - timedelta(days=4)).isoformat()
    print(f"updating {start} .. {end}")

    conn = sqlite3.connect(DB)
    dates = []
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    while d0 <= d1:
        dates.append(d0.isoformat())
        d0 += timedelta(days=1)

    games = get("games", **{"dates[]": dates})
    print(f"  {len(games)} games")
    gid = []
    n_bad = 0
    for g in games:
        if not g.get("date") or not g.get("home_team") or not g.get("away_team"):
            n_bad += 1
            continue
        g_id = f"bdl_{g['id']}"
        gid.append(g['id'])
        conn.execute(
            "INSERT OR REPLACE INTO games (game_id, league, game_date, "
            "home_team_id, away_team_id, home_score, away_score, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (g_id, "MLB", g.get("date"), f"bdl_{g['home_team']['id']}",
             f"bdl_{g['away_team']['id']}", g.get("home_team_score"),
             g.get("away_team_score"), (g.get("status") or "").lower()))

    if gid:
        # game_id -> date, from the games we just pulled. The stats endpoint
        # does not reliably nest a game object on every row, and
        # pitcher_game_stats.game_date is NOT NULL -- so the date is taken
        # from the games response rather than trusted to be present here.
        gdate = {f"bdl_{g['id']}": g.get("date") for g in games}

        stats = get("stats", **{"game_ids[]": gid})
        n_p = n_skip = 0
        for s in stats:
            pl, gm = s.get("player") or {}, s.get("game") or {}
            ip = s.get("ip")
            if ip is None:
                continue
            g_id = f"bdl_{gm.get('id')}" if gm.get("id") else None
            p_id = f"bdl_{pl.get('id')}" if pl.get("id") else None
            when = gm.get("date") or gdate.get(g_id)
            if not (g_id and p_id and when):
                n_skip += 1
                continue
            n_p += 1
            conn.execute(
                "INSERT OR REPLACE INTO pitcher_game_stats (game_id, player_id, "
                "player_name, team_id, game_date, innings_pitched, hits_allowed, "
                "runs_allowed, earned_runs, walks_allowed, strikeouts, "
                "home_runs_allowed, hit_by_pitch, is_starter, pitch_count) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (g_id, p_id,
                 f"{pl.get('first_name','')} {pl.get('last_name','')}".strip(),
                 f"bdl_{(s.get('team') or {}).get('id')}", when, ip,
                 s.get("h"), s.get("r"), s.get("er"), s.get("bb"), s.get("k"),
                 s.get("hr"), s.get("hb"), 1 if s.get("gs") else 0, s.get("pitch_count")))
        print(f"  {n_p} pitcher lines" +
              (f" ({n_skip} skipped, missing ids or date)" if n_skip else ""))

    conn.commit()
    latest = conn.execute(
        "SELECT MAX(date(game_date)) FROM pitcher_game_stats").fetchone()[0]
    conn.close()
    print(f"  data now runs through {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
