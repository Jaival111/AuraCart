# Product Recommendation System for E-commerce

![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-brightgreen)
![Jupyter](https://img.shields.io/badge/Model-Jupyter_Notebook-orange)

A machine learning-based product recommendation system implemented in an e-commerce shopping website using FastAPI and MongoDB.

## Features

- Personalized product recommendations based on user behavior
- FastAPI backend for efficient API responses
- MongoDB for flexible data storage
- Jupyter Notebook for model development and experimentation
- Easy-to-run development server

## Prerequisites

- Python 3.7+
- MongoDB (local or Atlas connection)
- pip package manager

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Jaival111/AuraCart.git
   cd your-repo-name

2. Create and activate a virtual environment (recommended).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate'

3. Install the required dependencies
    ```bash
    pip install -r requirements.txt

## Configuration

Set up your MongoDB connection:
- Create a .env file in the root directory
- Add your MongoDB connection string and JWT Secret Key in .env

## Running the Application

To start the FastAPI development server
```bash
    fastapi dev server.py
```

The application will be available at http://localhost:8000.

## Model Development

The main recommendation model is developed in model.ipynb. This Jupyter notebook contains:

- Data preprocessing
- Model training
- Evaluation metrics
- Recommendation logic

## Contact

Email me here: [email](contactjaival@gmail.com)