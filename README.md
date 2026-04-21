# README - n8n Workflow `ml automsation`

This document explains each node in the n8n workflow defined in `ml automsation.json`.

## Overview

The workflow orchestrates 3 ML pipelines:
- Supplier Clustering
- Best Time To Sell
- Best Time To Promote

The workflow can be triggered in two ways:
- automatically every day at 07:00 (Schedule Trigger1)
- via webhook `POST /trigger-ml-pipeline` (Webhook), typically called when a new supplier is added

If everything succeeds, a confirmation email is sent.
If one model fails, a model-specific alert email is sent.

## Node Details

### 1) Schedule Trigger1
- **Type** : `n8n-nodes-base.scheduleTrigger`
- **Role** : Triggers the scheduled execution of the pipeline.
- **Key parameter** : `triggerAtHour: 7`
- **Outputs** :
  - to **Fetch Sales Data**
  - to **Fetch Promotion Data**
  - to **Fetch Supplier Data**

### 2) Webhook
- **Type** : `n8n-nodes-base.webhook`
- **Role** : Triggers the pipeline from an external event, in practice when a new supplier is added.
- **Key parameters** :
  - `httpMethod: POST`
  - `path: trigger-ml-pipeline`
  - header authentication (`httpHeaderAuth`)
- **Output** : to **Fetch Supplier Data** (starts the supplier branch)

### 3) Fetch Supplier Data
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Fetches the supplier dataset from the ML API.
- **Endpoint** : `GET http://127.0.0.1:8000/train/supplier/dataset`
- **Query** : `limit=90`
- **Output** : to **Predict Supplier Clustering**

### 4) Predict Supplier Clustering
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Runs supplier clustering training/prediction.
- **Endpoint** : `POST http://127.0.0.1:8000/train/supplier/train`
- **Query** : `n_clusters=3`, `random_state=42`
- **Error handling** : `onError: continueErrorOutput`
- **Outputs** :
  - success to **Store Clustering Results**
  - error to **mail echec supplier**

### 5) Store Clustering Results
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Saves clustering results to the database (lookup table).
- **Endpoint** : `POST http://127.0.0.1:8000/lookup/supplier-clusters/save`
- **Output** : to **Merge** (input 0)

### 6) Fetch Sales Data
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Fetches the sales dataset.
- **Endpoint** : `GET http://127.0.0.1:8000/train/sell/dataset`
- **Query** : `limit=2000`
- **Output** : to **Predict Best Time To Sell**

### 7) Predict Best Time To Sell
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Runs the model to predict the best time to sell.
- **Endpoint** : `POST http://127.0.0.1:8000/train/sell/train`
- **Query** : `random_state=42`
- **Error handling** : `onError: continueErrorOutput`
- **Outputs** :
  - success to **Store Timing Results**
  - error to **mail echec sell**

### 8) Store Timing Results
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Saves sales timing results to a monthly lookup.
- **Endpoint** : `POST http://127.0.0.1:8000/lookup/sell-monthly/save`
- **Output** : to **Merge** (input 1)

### 9) Fetch Promotion Data
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Fetches the promotions dataset.
- **Endpoint** : `GET http://127.0.0.1:8000/train/promote/dataset`
- **Query** : `limit=3000`
- **Output** : to **Predict Best Time To Promote**

### 10) Predict Best Time To Promote
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Runs the model to predict the best time to promote.
- **Endpoint** : `POST http://127.0.0.1:8000/train/promote/train`
- **Query** : `random_state=42`
- **Error handling** : `onError: continueErrorOutput`
- **Outputs** :
  - success to **Store Promotion Results**
  - error to **mail echec promote**

### 11) Store Promotion Results
- **Type** : `n8n-nodes-base.httpRequest`
- **Role** : Saves promotion results to a monthly lookup.
- **Endpoint** : `POST http://127.0.0.1:8000/lookup/promote-monthly/save`
- **Output** : to **Merge** (input 2)

### 12) Merge
- **Type** : `n8n-nodes-base.merge`
- **Role** : Waits for the 3 success branches before continuing.
- **Key parameter** : `numberInputs: 3`
- **Expected inputs** :
  - supplier results
  - sell results
  - promote results
- **Output** : to **Log Pipeline Execution**

### 13) Log Pipeline Execution
- **Type** : `n8n-nodes-base.executeCommand`
- **Role** : Writes a local success log through a shell command.
- **Command** : `python -c "print('Pipeline ML executed successfully')"`
- **Output** : to **mail succes**

### 14) mail succes
- **Type** : `n8n-nodes-base.gmail`
- **Role** : Sends a global success email when all 3 models are complete.
- **Recipient** : `maysa.guesmi463@gmail.com`
- **Subject** : `Pipeline ML Sougui - Success`
- **Content** : execution summary + date + list of executed models.

### 15) mail echec supplier
- **Type** : `n8n-nodes-base.gmail`
- **Role** : Sends an alert email when the Supplier Clustering model fails.
- **Trigger** : error output of **Predict Supplier Clustering**
- **Recipient** : `maysa.guesmi463@gmail.com`

### 16) mail echec sell
- **Type** : `n8n-nodes-base.gmail`
- **Role** : Sends an alert email when the Best Time To Sell model fails.
- **Trigger** : error output of **Predict Best Time To Sell**
- **Recipient** : `maysa.guesmi463@gmail.com`

### 17) mail echec promote
- **Type** : `n8n-nodes-base.gmail`
- **Role** : Sends an alert email when the Best Time To Promote model fails.
- **Trigger** : error output of **Predict Best Time To Promote**
- **Recipient** : `maysa.guesmi463@gmail.com`

## Simplified Global Sequence

1. Trigger (schedule or webhook).
2. Load the 3 datasets (supplier, sell, promote).
3. Train/predict each model.
4. Save results from each branch.
5. Synchronize the 3 branches through **Merge**.
6. Write success log.
7. Send success email.
8. If one branch fails, send the branch-specific failure email.


