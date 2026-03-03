# Esprit-PI-4BI4-2026-ECOMMERCE
Developed as part of the PI track (4th Year Engineering Program) at Esprit School of Engineering – Tunisia (Academic Year 2025–2026). Tech: Talend, PostgreSQL, Power BI, Apache Airflow, Python, n8n.

# E-COMMERCE BI Platform
## Overview
This project was developed as part of the **PI – 4th Year Engineering Program** at **Esprit School of Engineering** (Academic Year 2025–2026).  
It delivers an end-to-end Business Intelligence workflow for an e-commerce context: data integration, data warehouse modeling, orchestration, and decision dashboards.

## Features
- Integration of multiple data sources (sales, orders, payments, deliveries, etc.)
- Data cleaning and normalization (types, duplicates, null values)
- Data Warehouse implementation (dimensions, facts, surrogate keys, constraints)
- SCD handling for historical tracking (where applicable)
- Automated orchestration with Apache Airflow (master/sub-jobs, monitoring)
- KPI dashboards and reporting

## Tech Stack
### Frontend
- Power BI (Dashboards & Reporting)

### Backend
- PostgreSQL (Data Warehouse)
- Talend (ETL / Data Integration)
- Apache Airflow (Orchestration)

## Architecture
- **Staging Area (SA):** raw landing tables loaded from source files
- **ODS (optional):** cleaned and normalized intermediate layer
- **Data Warehouse (DW):** star/constellation schema (Facts + Dimensions)
- **Orchestration:** Airflow DAGs triggering Talend jobs and SQL scripts
- **Analytics:** Power BI dashboards connected to DW

## Contributors
- [Your Name / GitHub Username]

## Academic Context
Developed at **Esprit School of Engineering – Tunisia**  
PI – 4th Year Engineering Program | Academic Year 2025–2026

## Getting Started
1. Clone the repository
2. Prepare the database (PostgreSQL) and run SQL scripts in `04-DataWarehouse/`
3. Configure Talend connections and contexts, then run jobs in `03-ETL/`
4. Configure Airflow connection variables and deploy DAGs from `06-Airflow/`
5. Open Power BI dashboards from `05-Dashboards/` and refresh data

## Acknowledgments
- Esprit School of Engineering
- Course instructors and supervisors
