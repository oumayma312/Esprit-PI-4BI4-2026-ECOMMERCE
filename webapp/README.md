# Flask Web App (3 models)

This app displays the notebook visualizations for the 3 models by extracting plots from the `.ipynb` cell outputs.

## 1) Add the background image

Save your image as:

- `webapp/img/back.jpg`

The app serves this file at:

- `/img/back.jpg`

## 2) Make sure notebooks contain outputs

The app can only show plots that exist in the notebook outputs.

If a section is empty:

- Open the notebook and run the cells that generate the plots
- Refresh the web page

## 3) Run

From the repo root:

- `pip install -r requirements.txt`
- `python webapp/app.py`

Then open:

- `http://127.0.0.1:5000`
