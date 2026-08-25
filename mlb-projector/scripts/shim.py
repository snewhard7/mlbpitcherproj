"""
Minimal stand-ins for the two helpers projection.py needs, so this repo
does not depend on the full research project.
"""
from datetime import datetime, timedelta


class storage:
    @staticmethod
    def local_game_date(ts):
        """LOCAL date of a UTC games.game_date timestamp.

        games.game_date is UTC while Statcast dates are local, so an 8pm
        Eastern first pitch is stored as the NEXT day. Joining the two on
        the raw date silently drops 24% of the schedule, all of it late
        games. No game starts later than 02:00 UTC, so subtracting five
        hours moves after-midnight games back without dragging afternoon
        games into the previous day.
        """
        try:
            return str(datetime.fromisoformat(ts) - timedelta(hours=5))[:10]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def outs_from_innings_pitched(ip):
        """6.2 innings is 6 innings and 2 outs = 20. The decimal counts
        outs; it is not a fraction."""
        if ip is None:
            return None
        try:
            whole = int(float(ip))
            return whole * 3 + int(round((float(ip) - whole) * 10))
        except (TypeError, ValueError):
            return None
