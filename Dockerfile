FROM python:3.13.5

WORKDIR /app

# Install system dependencies for numpy, plotly, cvxpy, solvers
RUN apt-get update && apt-get install -y \
    gcc g++ gfortran \
    libblas-dev liblapack-dev \
    libfreetype6-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Try to add --upgrade if you face any issues with package versions
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]