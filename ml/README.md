## ML structure

- `database/`: PostgreSQL connectivity and warehouse snapshot loading.
- `models/`: API request schemas used by FastAPI.
- `services/`: production prediction and recommendation services.
- `notebooks/`: research notebooks kept as references for the ML workflows.

Production traffic goes through the FastAPI routes exposed from `ml/api.py`.
