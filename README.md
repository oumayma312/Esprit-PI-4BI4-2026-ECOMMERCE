# Esprit-PI-4BI4-2026-ECOMMERCE
Developed as part of the PI track (4th Year Engineering Program) at Esprit School of Engineering – Tunisia (Academic Year 2025–2026). Tech: Talend, PostgreSQL, Power BI, Apache Airflow, Python, n8n.

# E-COMMERCE BI Platform
## Overview
This project was developed as part of the **PI – 4th Year Engineering Program** at **Esprit School of Engineering** (Academic Year 2025–2026).  
It delivers an end-to-end Business Intelligence workflow for an e-commerce context: data integration, data warehouse modeling, orchestration, and decision dashboards.

## Features
- Integration of multiple data sources 
- Data cleaning and normalization
- Data Warehouse implementation
- SCD handling for historical tracking 
- Automated orchestration with Apache Airflow
- KPI dashboards and reporting

## Tech Stack
### Frontend
- Angular 

### Backend
- PostgreSQL (Data Warehouse)
- Talend (ETL / Data Integration)
- Apache Airflow (Orchestration)


## Architecture
- **Staging Area (SA):** raw landing tables loaded from source files
- **Data Warehouse (DW):** constellation schema (Facts + Dimensions)
- **Orchestration:** Airflow DAGs triggering Talend jobs and SQL scripts
- **Analytics:** Power BI dashboards connected to DW

## Contributors
- Datoops Team 

## Academic Context
Developed at **Esprit School of Engineering – Tunisia**  
PI – 4th Year Engineering Program | Academic Year 2025–2026

## Acknowledgments
- Esprit School of Engineering
- Course instructors and supervisors
