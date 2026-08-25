"""
Export a SELF-CONTAINED projector you can use on a phone, offline.

THE PROBLEM THIS SOLVES. tests/project_slate.py needs a command line and
the 1.3 GB database, so it only runs at a computer. A scheduled task plus
cloud sync would work but still depends on that machine being awake.

THE APPROACH. The model is a linear regression per prop plus an empirical
outcome distribution -- small enough to EMBED. This writes one HTML file
containing the fitted coefficients, every starter's current form, and each
team's scoring rate. You pick a pitcher and opponent on the phone and the
ladder is computed in the browser. No server, no network, no command.

IT ALSO REMOVES THE DEPENDENCE ON PROBABLE PITCHERS, which the slate
version needs and which are often not stored until shortly before games.
You are reading the matchup off the book anyway.

WHAT GOES STALE, AND HOW FAST. A pitcher's recent form only changes when
he starts, so roughly every five days. Regenerating weekly is ample;
regenerate after any data pull if you want it exact. The file states the
date it was built from so a stale copy is obvious.

WHY THE DISTRIBUTION IS SHIPPED AS A HISTOGRAM. Outs pile up at inning
boundaries -- 15 outs is five innings and 18 is six, and roughly a fifth
of starts stop at each. A line at 15.5 sits IMMEDIATELY ABOVE the first
pile, so the over must clear a whole further inning past the most common
stopping point. No smooth curve reproduces that, so the empirical
distribution travels with the model rather than being approximated in the
browser.

Run:  python -m tests.export_phone_projector
"""
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from shim import storage
from projection import ProjectionModel, PROPS, PROP_LINES, _feature_names

DB = Path(__file__).parent.parent / "data.db"
LABEL = {"outs": "Outs", "strikeouts": "Ks", "hits": "Hits",
         "walks": "Walks", "runs": "Runs"}


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    print("fitting...", flush=True)
    model = ProjectionModel()
    rows = model.build(conn, since="2026-01-01")
    model.fit(rows).fit_conditional(rows)
    asof = max(r["date"] for r in rows)
    print(f"  {len(rows)} starts, data through {asof}", flush=True)

    # --- CONTEXT ONLY: xwOBA against actual wOBA -------------------------
    #
    # This does NOT enter the projection, and that is a deliberate choice
    # backed by a test. Proper xwOBA -- expected value on contact, actual
    # value on strikeouts and walks -- IS more predictive than past
    # results: across 353 pitchers with 16+ games, first-half xwOBA
    # forecast second-half wOBA at r=+0.459 against +0.407 for first-half
    # wOBA. The classic claim holds.
    #
    # But adding it to the regression moved every prop by less than 0.35%,
    # several of them negative. Two reasons. These props are counting
    # stats dominated by WORKLOAD -- outs correlate +0.28 with batters
    # faced and only +0.12 with per-PA rate -- so xwOBA sharpens a term
    # that is not the binding one. And its advantage over plain wOBA is
    # largest at SMALL samples, while a five-start average has mostly
    # caught up already.
    #
    # Where it does matter is exactly where results have not caught up to
    # mechanism: a pitcher back from injury, a mid-season stuff change, a
    # rookie with three starts. That is a judgement call rather than a
    # regression term, so it is surfaced as context and left to the reader.
    xwoba = {r["player_id"]: (r["xwoba"], r["woba"])
             for r in conn.execute("SELECT * FROM pitcher_xwoba")}

    # --- per pitcher current form, from his most recent starts -----------
    hist = defaultdict(list)
    for r in conn.execute(
            """SELECT p.player_id, p.player_name, p.innings_pitched, p.pitch_count,
                      p.strikeouts, p.hits_allowed, p.walks_allowed,
                      p.runs_allowed, g.game_date
               FROM pitcher_game_stats p JOIN games g USING(game_id)
               WHERE p.is_starter = 1"""):
        d = storage.local_game_date(r["game_date"])
        o = storage.outs_from_innings_pitched(r["innings_pitched"])
        if d and o is not None:
            hist[(r["player_id"], r["player_name"])].append({
                "date": d, "outs": float(o), "pitches": float(r["pitch_count"] or 0),
                "strikeouts": float(r["strikeouts"] or 0),
                "hits": float(r["hits_allowed"] or 0),
                "walks": float(r["walks_allowed"] or 0),
                "runs": float(r["runs_allowed"] or 0)})

    pitchers = []
    for (pid, name), h in hist.items():
        h.sort(key=lambda x: x["date"])
        # only pitchers who have started recently enough to still be starting
        if len(h) < 5 or h[-1]["date"] < "2026-06-15":
            continue
        f = {"recent_pitches": statistics.mean(x["pitches"] for x in h[-5:]),
             "recent_outs": statistics.mean(x["outs"] for x in h[-5:]),
             "n_starts": float(len(h))}
        for p in PROPS:
            f[f"recent_{p}"] = statistics.mean(x[p] for x in h[-5:])
            f[f"career_{p}"] = statistics.mean(x[p] for x in h)
        rec = {"n": name, "d": h[-1]["date"],
               "f": {k: round(v, 4) for k, v in f.items()}}
        if pid in xwoba:
            rec["x"], rec["w"] = xwoba[pid]
        pitchers.append(rec)
    pitchers.sort(key=lambda x: x["n"])
    print(f"  {len(pitchers)} active starters", flush=True)

    teams = {}
    for r in conn.execute("SELECT team_id, abbreviation FROM teams"):
        runs = [x["runs_scored"] for x in conn.execute(
            "SELECT game_date, runs_scored FROM team_game_stats WHERE team_id=? "
            "ORDER BY game_date DESC LIMIT 30", (r["team_id"],))
            if x["runs_scored"] is not None]
        teams[r["abbreviation"]] = round(statistics.mean(runs), 3) if runs else 4.5

    # --- model payload ---------------------------------------------------
    fits = {}
    for p in PROPS:
        F, mu, sd, beta = model.fits[p]
        # ship the conditional distribution as (bin centre, outcome counts)
        bins = [{"c": round(c, 3),
                 "h": {str(int(k)): int(v) for k, v in
                       zip(*np.unique(np.round(s), return_counts=True))}}
                for c, s in model.cond[p]]
        fits[p] = {"F": F, "mu": [round(x, 6) for x in mu],
                   "sd": [round(x, 6) for x in sd],
                   "b": [round(x, 6) for x in beta],
                   "bins": bins, "lines": PROP_LINES[p], "label": LABEL[p]}

    payload = {"asof": asof, "fits": fits, "pitchers": pitchers, "teams": teams}
    out = Path(__file__).parent.parent / "index.html"
    out.write_text(render(payload), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"\nwrote {out.resolve()}  ({kb:.0f} KB)")
    return 0


def render(payload):
    data = json.dumps(payload, separators=(",", ":"))
    return """<!doctype html><meta charset=utf-8>
<meta name=viewport content='width=device-width,initial-scale=1'>
<title>Prop Projector</title><style>
*{box-sizing:border-box}
body{margin:0;padding:12px;background:#12141a;color:#e8eaf0;
 font:16px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
h1{font-size:18px;margin:0 0 2px}
.sub{color:#8a91a3;font-size:12px;margin-bottom:14px}
label{display:block;font-size:11px;color:#7a8196;text-transform:uppercase;
 letter-spacing:.4px;margin:10px 0 3px}
select,input{width:100%;padding:11px;font-size:16px;background:#1b1e26;
 color:#e8eaf0;border:1px solid #2b3040;border-radius:9px}
.row{display:flex;gap:9px}.row>div{flex:1}
.prop{background:#1b1e26;border-radius:11px;margin-top:13px;padding:12px 13px}
.plabel{font-size:13px;color:#9aa3b8;margin-bottom:6px}
.proj{color:#6ee7a8;font-weight:700;font-size:17px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{color:#7a8196;font-weight:500;font-size:10px;text-align:right;padding:3px 4px;
 text-transform:uppercase;letter-spacing:.4px}
th:first-child{text-align:left}
td{padding:5px 4px;text-align:right;border-top:1px solid #262b38}
td:first-child{text-align:left;color:#c3cad9}
.o{color:#7fb8ff}.u{color:#ffb37f}.near{background:#222a38}
.foot{color:#6b7285;font-size:11px;margin-top:18px;line-height:1.6}
.ctx{background:#1b1e26;border-radius:11px;margin-top:13px;padding:11px 13px;
 font-size:13px;color:#9aa3b8}
.ctx b{color:#e8eaf0;font-weight:600}
.rest{color:#8a91a3;font-size:12px;margin-top:5px}
.rest b{color:#6ee7a8}
.luck{float:right;font-weight:600}
.lucky{color:#ffb37f}   /* results flattering -- expect regression */
.unlucky{color:#7fb8ff} /* results harsh -- expect improvement */
</style>
<h1>Prop Projector</h1>
<div class=sub id=asof></div>
<label>Pitcher</label><select id=p></select>
<div class=row>
 <div><label>Opponent</label><select id=o></select></div>
 <div><label>Side</label><select id=h><option value=1>Home</option>
   <option value=0>Away</option></select></div>
</div>
<label>Game date</label><input id=g type=date>
<div class=rest id=restnote></div>
<label style="margin-top:12px">Context</label>
<select id=po><option value=0>Regular season</option>
 <option value=1>Playoffs</option></select>
<div class=rest id=ponote></div>
<div id=out></div>
<div class=foot>Fair odds, no vig &mdash; a posted &minus;110 against a fair
&minus;110 is a 4.5% loser. Calibrated against realised outcomes, not against
the market: a gap is a genuine disagreement, not proof the price is wrong.
No prop has been shown to beat a closing line. Walks carries the least
signal; treat its gaps with the most suspicion.</div>
<script>
const D=__DATA__;
document.getElementById('asof').textContent='data through '+D.asof+
  ' \\u00b7 '+D.pitchers.length+' starters';
const P=document.getElementById('p'),O=document.getElementById('o');
D.pitchers.forEach((x,i)=>P.add(new Option(x.n+'  ('+x.d+')',i)));
Object.keys(D.teams).sort().forEach(t=>O.add(new Option(t,t)));
function amer(p){if(p<=.001)return'+99999';if(p>=.999)return'-99999';
 return p>.5?'-'+Math.round(100*p/(1-p)):'+'+Math.round(100*(1-p)/p);}
function calc(){
 const pit=D.pitchers[+P.value], opp=O.value;
 // REST IS DERIVED, not typed. The file already carries each pitcher's
 // last start date, so subtracting is the machine's job. Capped at 12 to
 // match how the model was fitted -- beyond that the feature was clipped
 // in training and extrapolating it would be inventing a value the
 // regression never saw.
 let rest=5, note='';
 const gd=document.getElementById('g').value;
 if(gd && pit.d){
  const ms=(new Date(gd)-new Date(pit.d))/86400000;
  if(ms>0){ rest=Math.min(Math.round(ms),12);
    note='last start '+pit.d+' \u00b7 <b>'+Math.round(ms)+' days rest</b>'+
      (ms>12?' (capped at 12 for the model)':'');
  } else { note='game date is not after the last start \u2014 using 5 days'; }
 } else { note='pick a game date to derive rest \u00b7 assuming 5 days'; }
 document.getElementById('restnote').innerHTML=note;
 const ctx={rest:rest,
   home:+document.getElementById('h').value, opp_runs:D.teams[opp]};
 // the strikeout tier is keyed off the pitcher's projected OUTS, not his
 // projected strikeouts -- the tiers were measured on start length.
 let base_outs=0;
 {const f=D.fits['outs']; let z=f.b[0];
  f.F.forEach((n,i)=>{const v=(n in pit.f)?pit.f[n]:ctx[n];
   z+=f.b[i+1]*((v-f.mu[i])/f.sd[i]);});
  base_outs=z;}
 let html='';
 if(pit.x!==undefined){
  const d=pit.w-pit.x;
  // SIGN CARE: for a PITCHER, a HIGHER wOBA allowed is worse. Actual
  // above expected means he was hit harder than his contact quality
  // warranted -- unlucky, and an argument his results should improve.
  const tag=d>0.02?'<span class="luck unlucky">unlucky &middot; hit harder than contact quality</span>'
      :(d<-0.02?'<span class="luck lucky">lucky &middot; better results than contact quality</span>':'');
  html+='<div class=ctx>last 5 starts &nbsp; xwOBA <b>'+pit.x.toFixed(3)+
    '</b> &nbsp; actual wOBA <b>'+pit.w.toFixed(3)+'</b>'+tag+'</div>';
 }
 for(const key in D.fits){
  const f=D.fits[key];
  let z=f.b[0];
  f.F.forEach((name,i)=>{
   const v=(name in pit.f)?pit.f[name]:ctx[name];
   z+=f.b[i+1]*((v-f.mu[i])/f.sd[i]);
  });
  // PLAYOFF ADJUSTMENT -- measured by hand, not fitted into the model.
  //
  // Within pitcher, same season, 383 postseason starts: outs run 2.85
  // shorter overall, at -11.63 sigma. But the effect is NOT uniform, and
  // the difference between mid-tier starters (-3.69) and aces (-1.91) is
  // 3.6 sigma. Applying one blanket number would be nearly a full out too
  // aggressive on exactly the arms most likely to be worth betting, so
  // the tiers are used even though they barely beat flat on RMSE --
  // RMSE is dominated by per-start variance and cannot see a systematic
  // subgroup error that matters enormously for pricing one pitcher.
  //
  // HITS take a flat -0.50 (5.12 sigma). WALKS (-1.35 sigma) and RUNS
  // (-0.03) never showed a real effect and are left alone.
  //
  // A fitted line scored slightly better on hits, walks and runs -- but
  // its slopes were 0.348, 0.474 and -0.075, meaning it was largely
  // DISCARDING the pitcher and substituting a constant. That is a strong
  // claim from 383 starts for a 3-5% RMSE gain, so it is not used.
  if(document.getElementById('po').value==='1'){
   if(key==='outs')       z += (z<15.5? -2.26 : (z<17.5? -3.69 : -1.91));
   else if(key==='strikeouts') z += (base_outs<15.5? -1.09 : (base_outs<17.5? -1.34 : -0.74));
   else if(key==='hits')  z += -0.50;
  }
  // nearest conditional bin, shifted to this projection
  let best=f.bins[0];
  f.bins.forEach(b=>{if(Math.abs(b.c-z)<Math.abs(best.c-z))best=b;});
  const shift=z-best.c;
  let tot=0;const pts=[];
  for(const k in best.h){const c=best.h[k];tot+=c;pts.push([+k+shift,c]);}
  const near=f.lines.map(l=>Math.abs(l-z)).reduce((a,b,i,arr)=>arr[a]<b?a:i,0);
  const lo=Math.max(0,near-3),hi=Math.min(f.lines.length,near+4);
  let rows='';
  f.lines.slice(lo,hi).forEach(l=>{
   let over=0;pts.forEach(([v,c])=>{if(Math.round(v)>l)over+=c;});
   const po=over/tot;
   rows+='<tr'+(Math.abs(l-z)<1?' class=near':'')+'><td>'+l+'</td>'+
    '<td class=o>'+Math.round(po*100)+'%</td><td class=o>'+amer(po)+'</td>'+
    '<td class=u>'+Math.round((1-po)*100)+'%</td><td class=u>'+amer(1-po)+'</td></tr>';
  });
  html+='<div class=prop><div class=plabel>'+f.label+
   ' <span class=proj>'+z.toFixed(2)+'</span></div><table>'+
   '<tr><th>Line</th><th>Over</th><th>Fair</th><th>Under</th><th>Fair</th></tr>'+
   rows+'</table></div>';
 }
 document.getElementById('out').innerHTML=html;
}
[P,O,'h','g','po'].forEach(e=>{const el=typeof e==='string'?document.getElementById(e):e;
 el.addEventListener('change',calc);el.addEventListener('input',calc);});
(function(){const d=new Date();
 document.getElementById('g').value=
  d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+
  String(d.getDate()).padStart(2,'0');})();
 document.getElementById('ponote').innerHTML =
   document.getElementById('po').value==='1'
   ? 'outs and strikeouts adjusted by tier, hits by \u22120.50 \u00b7 '+
     'walks and runs unchanged \u00b7 <b>treat hits/walks/runs gaps with '+
     'extra suspicion in October</b>'
   : '';
calc();
</script>""".replace("__DATA__", data)


if __name__ == "__main__":
    sys.exit(main())
