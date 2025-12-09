# Australian Aurora Watch API



## Install Locally

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run Locally

```
uvicorn app.main:app --reload --host
```

## Docker Build and Run

```
docker build -t spacer-weather-api-wrapper .
docker run -p 8200:8200 space-weather-api-wrapper
```

## API Documentation

Once the server is running, navigate to `http://localhost:8200/docs` to access the interactive API documentation provided by Swagger UI.


## Dev Notes

### TODO

- [x] Implement Pydantic response_models
- [x] Add telegram bot integration for alerts and define when alerts should be sent
- [x] Add Dockerfile for deployment including mounted volume for logs
- Deploy to portainer 
- Configure fluent-bit on pi for shipping logs to my logflow service
