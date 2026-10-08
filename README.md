---
title: Waypoint Planning Console
emoji: 🚚
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# Waypoint Planning Console

Team Synapse, Tech Triathlon 2026 Datathon. A Streamlit app that runs our trained models:

- **Tonight's run**: every planned stop scored for handling time and late risk.
- **Route risk**: one route against its delivery windows, with a live what if (late departure, road conditions) scored by the LightGBM models.
- **Demand outlook**: the 10 week volume forecast per depot and brand.
- **Peak day plan**: the integer programming allocation, the reasons for each deferral, and a live check of all seven feasibility rules.
- **How it works**: label construction, the expected slack feature and the architecture.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the app data

`app_data/` is built once from the competition data and our saved models:

```
python tools/build_app_data.py <path to data folder> <path to Synapse_Datathon folder>
```

The competition data is confidential. Keep this repository and any Space **private**.
