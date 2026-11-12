# STORM SSI: Streamlit mock-up of the Replit app

What the planned Replit product would look like: users upload data and download results. The code stays on the server.

## Run it (JupyterHub or your laptop)
```
pip install -r requirements.txt
python -m textblob.download_corpora lite
streamlit run app.py
```
Then open the link it prints. Switch on **Use the demo data** to try it without uploading anything.

## Files
- `app.py`: the app (screens only; every calculation is in `ssi_pipeline.py`)
- `ssi_pipeline.py`: the same pipeline as in `ssi_pipeline/`. Keep both copies in step.
- `demo/`: 30,472 Bitcoin posts from 2024 plus the 2024 Google Trends ISB
- `.streamlit/config.toml`: allows uploads up to 500 MB

## Flow
1. Data: upload a microblog CSV (text + time stamp) and, optionally, an ISB CSV. Text and time columns are detected and can be changed.
2. Settings (sidebar): libraries (TextBlob and VADER; the BERT models are shown greyed out for v2), time unit (auto-detect by default), date window, minimum posts per bucket, data voids.
3. Results: headline numbers, flagged issues, an interactive SSI chart, the pipeline's graphs, the data table and dictionary, the QC summary, and downloads (CSV, dictionary, QC, or everything as a zip).

## Notes
- The demo ISB is rescaled to its own 2024 peak, so demo SSI values are higher than in the full 2015 to 2025 run.
- No data voids are set by default. The Bitcoin 2023 gap (2023-06-14 to 2023-10-12) is entered in the sidebar table only for that dataset.
- Mock-up only: no log-in, and uploads are kept in a temporary folder for the session.
