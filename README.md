# Google Play Store — Hexbin Density Dashboard

A Dash / Plotly dashboard over two local CSV files (`data_apps.csv`,
`data_reviews.csv`). The CSVs are read locally with pandas; nothing is
hard-coded or pasted into the app.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:8050

## Time gate (5 PM – 7 PM IST)

The dashboard is **only visible between 5:00 PM and 7:00 PM IST**.
The time zone is handled with `ZoneInfo("Asia/Kolkata")`. Outside that
window a locked screen shows the live IST clock and a countdown to the
next opening.

**Automatic time refresh:** a `dcc.Interval` re-evaluates the time gate
and the clock every second, so the dashboard appears automatically at
5:00 PM IST with no manual reload. While the dashboard is open, the page
also auto-reloads every 5 minutes.

## Data files

The app detects which file is which by column content (the two file
names are swapped in this workspace):

- **Store data** (has `Category`, `Rating`, `Reviews`, `Size`,
  `Installs`) → `data_reviews.csv`
- **Reviews / sentiment data** (has `Sentiment_Subjectivity`) →
  `data_apps.csv`

Average `Sentiment_Subjectivity` is computed per App from the reviews
file and merged into the store data by App name (case-insensitive,
whitespace-trimmed).

## Filters (all applied together)

- Categories: Game, Beauty, Business, Comics, Communication, Dating,
  Entertainment, Social, Events
- Rating > 3.5
- Installs > 50,000
- Reviews > 500
- Size between 10 and 100 MB (`M`/`K`/`G` suffixes and
  "Varies with device" are handled)
- Average Sentiment Subjectivity > 0.5
- App names containing the letter "s" (case-insensitive) are excluded

## Visuals

- **True hexbin** of Size (X) vs Rating (Y) using an offset (odd-r)
  hexagonal grid; hexagon colour encodes the **average Installs** per
  hex (interactive bin-size sliders and colour-scale selector).
- **Marginal distributions**: App Size histogram (top) and Rating
  histogram (right) of the filtered dataset.
- **Game** apps highlighted in **pink** over the hexbin.
- **Category-level IQR outliers** on Installs (Q1 − 1.5·IQR /
  Q3 + 1.5·IQR computed per category), drawn as labeled black diamonds
  (app names printed next to the points) plus an outliers table and an
  IQR-bounds table per category.
- Installs box plot per category (log scale).
- KPI cards (filtered apps, avg rating, avg installs, categories,
  Game apps, outliers).

## Category display names

- Beauty → **सुंदरता** (Hindi)
- Business → **வணிகம்** (Tamil)
- Dating → **Dating**

All other categories use standard English title case. Underlying data
values are unchanged; only the display labels are translated.

## Files

- `app.py` — the dashboard
- `requirements.txt` — Python dependencies
- `README.md` — this file
