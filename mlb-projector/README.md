# Pitcher prop projector

A page that rebuilds itself every morning and reads on a phone.

Pick a pitcher and opponent, get projected outs, strikeouts, hits, walks
and runs, with a ladder of lines around each projection and fair odds
(no vig) on both sides.

---

## What this is, and what it is not

It is a **reference number**, not a bet signal.

Ten hypotheses were tested against real collected closing lines — the
simulation, opening-line value, bullpen state, manager identity,
cross-prop consistency, public-bias proxies, distributional discreteness,
moneyline movement, workload limits, and cross-book dispersion. **None
produced an edge that survived scrutiny.** Nothing here claims to beat the
market.

What it does claim is **calibration**. On held-out 2026 starts, all 32
posted lines across the five props came in within 0.05 of reality, most
within 0.02. So when this number disagrees with a price, that is a genuine
disagreement rather than model error — which is what makes it worth a
human second look.

Signal strength varies a lot by prop:

| prop | improvement over predicting the mean |
|---|---|
| outs | +11.2% |
| strikeouts | +9.1% |
| hits | +4.1% |
| walks | +2.6% |
| runs | +1.2% |

**Walks and runs barely beat a constant.** Treat their disagreements with
the most suspicion.

---

## One-time setup (~20 minutes, mostly clicking)

### 1. Create the repo

On GitHub, make a new repository. **Public** is simplest — GitHub Pages is
free on public repos, and there is nothing sensitive here. Upload every
file in this folder, keeping the structure:

```
data.db
index.html
scripts/
  projection.py
  shim.py
  update_data.py
  build_page.py
.github/workflows/daily.yml
```

### 2. Add your API key as a secret

**Settings → Secrets and variables → Actions → New repository secret**

- Name: `BALLDONTLIE_API_KEY`
- Value: your key

A secret is never visible in the repo or in logs. Do not put the key in a
file.

### 3. Turn on Pages

**Settings → Pages → Source: Deploy from a branch → Branch: main, folder:
/ (root)**

After a minute your page is at:

```
https://<your-username>.github.io/<repo-name>/
```

Bookmark that on your phone. Add it to your home screen and it behaves
like an app.

### 4. Check the job runs

**Actions → Daily projections → Run workflow.** It should finish in about
two minutes. If it goes green, you are done — it will now run every
morning by itself.

---

## Daily use

Open the bookmark. That is the whole workflow.

The page rebuilds around 9am Eastern, after the previous night's games are
final. It shows the date its data runs through, so a stale copy is
obvious rather than silent.

---

## Occasional maintenance

**Nothing is required.** But two things only refresh when you push from
your computer:

- **The xwOBA luck context** (expected vs actual wOBA over the last five
  starts), stored precomputed in the `pitcher_xwoba` table. It derives
  from Statcast, which the daily job deliberately does not pull — Statcast
  is slow and unreliable to automate, and the projection model uses none
  of it. Keeping only the 243 precomputed rows instead of 155,000 raw
  plate appearances is also what brings `data.db` under GitHub's 25 MB
  web-upload limit. The context ages usefully for weeks.
- **The model coefficients**, which refit from whatever is in `data.db`.

To refresh both, run the full local export and push the new `data.db`.

---

## Costs

Free. GitHub Actions is unlimited on public repos; this uses about two
minutes a day. BALLDONTLIE gets one small request per day.

---

## If something breaks

**Job fails on the API step** — check the secret name is exactly
`BALLDONTLIE_API_KEY`. Rate limits are retried automatically with backoff.

**Page does not update** — look at the Actions tab. A red run shows the
error. The job is safe to re-run; data pulls replace rows rather than
duplicating them.

**Page shows an old date** — the job runs but found nothing new, which is
normal on an off day.

**You want it now** — Actions → Daily projections → Run workflow. That
button works from a phone.
