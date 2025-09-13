# mlops_book_recommendation_system

**Author:** Teighan Miller  
**Status:** 🚧 In Development (Capstone Project for MLOps Zoomcamp)

---

## 📌 Table of Contents

1. [Project Overview](#project-overview)  
2. [Motivation & Objectives](#motivation--objectives)  
3. [What’s Built So Far](#whats-built-so-far)  
4. [Areas Under Development / Planned Improvements](#areas-under-development--planned-improvements)  
5. [Tech Stack & Architecture](#tech-stack--architecture)  
6. [How to Run / Use It](#how-to-run--use-it)  
7. [Testing, Monitoring & MLOps Practices](#testing-monitoring--mlops-practices)  
8. [Why This Project Matters](#why-this-project-matters)  
9. [Contribution Opportunities](#contribution-opportunities)  
10. [Contact](#contact)

---

## Project Overview

The **mlops_book_recommendation_system** is a book recommendation system built as the capstone project for the [MLOps Zoomcamp] (https://github.com/DataTalksClub/mlops-zoomcamp).  

The goal is not only to build a working recommendation model, but also to **deploy and maintain it using modern MLOps practices**: versioning, CI/CD, orchestration, monitoring, and reproducibility.

---

## 🎯 Motivation & Objectives

- Gain hands-on experience with MLOps workflows end-to-end.  
- Demonstrate the ability to design, build, deploy, and maintain a production-ready ML system.  
- Learn tools such as Docker, Airflow, MLflow, and experiment tracking.  
- Emphasize reproducibility, scalability, monitoring, and testing.  

---

## What’s Built So Far

| Component | Status | Key Features |
|-----------|--------|--------------|
| **Data Ingestion & Preprocessing** | Done | Scripts for cleaning, feature engineering, datasets managed under `data/` & `scripts/`. |
| **Model Training & Experimentation** | Done | Prototyped recommendation models; experiment tracking included. |
| **API / Serving Layer** | In Progress | Flask service scaffolded for serving recommendations; basic functionality available. |
| **Orchestration / Workflow Management** | In Progress | Early DAGs/pipelines for scheduling jobs; containerized with Docker. |
| **Versioning & Environment Management** | Partial | `requirements.txt` provided; Dockerfiles available; linting & formatting tools included. |
| **Logging & Monitoring** | Initial | Basic logging enabled; monitoring/drift detection under construction. |

---

## Areas Under Development / Planned Improvements

- **Model performance improvements**  
  - Hyperparameter tuning, algorithm comparison, and ensembling.  
  - Advanced evaluation metrics (precision, recall, diversity).

- **Scaling & Production Readiness**  
  - Harden API with authentication, error handling, and rate limiting.  
  - Improve deployment strategy for high availability.  

- **Workflow Orchestration & Scheduling**  
  - Complete Airflow DAGs for ingestion → training → validation → deployment.  
  - Automate retraining pipelines.  

- **CI/CD Pipelines**  
  - Unit & integration tests.  
  - Automated deployment to staging/production environments.  

- **Monitoring, Logging & Alerting**  
  - Structured logging.  
  - Model drift and data drift detection.  
  - Alerting when performance degrades.  

- **Data Versioning & Governance**  
  - Dataset version tracking for reproducibility.  
  - Metadata management for datasets.  

- **User Interface (Future Work)**  
  - Web UI/dashboard to request recommendations.  
  - Filtering and personalization features.  

---

## Tech Stack & Architecture

- **Language & Libraries:** Python (Pandas, NumPy, Scikit-learn, etc.)  
- **Experiment Tracking:** MLflow  
- **Workflow Orchestration:** Airflow (DAG-based pipelines)  
- **Containerization:** Docker  
- **Model Serving:** Flask API  (May change to Streamlit)
- **Data Storage:** Currently using a mix of local and S3 data storage. Development will result in all permanent data being stored in S3.
- **Testing / Linting:** Unit tests, pre-commit hooks, linting tools  

---

## Development Status & Limitations

This project is **still in active development** and not yet runnable as a standalone application. Many components, including the API and orchestration pipelines, depend on secure cloud infrastructure such as **Amazon S3 for dataset storage and model artifacts**. Without access keys and environment configurations, the system cannot function end-to-end.  

Current focus areas include refining the pipelines, improving model performance, and establishing CI/CD workflows. Once cloud integration is finalized, instructions for setup and usage will be added here.

---

## Contact

**Teighan Miller**  
- GitHub: [teighanmiller](https://github.com/teighanmiller)  
- Email: teighanspiermiller@gmail.com  
- LinkedIn: www.linkedin.com/in/teighan-miller-b384a1231  
