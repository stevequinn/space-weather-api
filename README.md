# Australian Aurora Watch API

This is a FastAPI application that serves as a wrapper for the Australian Aurora Watch API. It provides endpoint to fetch space weather data, including auroral activity, solar wind parameters, and geomagnetic indices.

It also alerts via telegram bot when auroral activity is high.

## Why?

I wanted to be pinged when an aurora may be visible from my location in Melbourne, Australia so I could go and take silly photos of it. 

Please don't complain about the lack of tests...I've just been using Bruno and haven't added any unit tests, oh no!!

## Install Locally

```
cp dot.env .env
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

Or use the provided `docker-compose.yml`:
```
docker-compose up --build -d
docker-compose logs -f
docker-compose down
```

## API Documentation

Once the server is running, navigate to `http://localhost:8200/docs` to access the interactive API documentation provided by Swagger UI.


## Dev Notes

### TODO

- [x] Implement Pydantic response_models
- [x] Add telegram bot integration for alerts and define when alerts should be sent
- [x] Add Dockerfile for deployment including mounted volume for logs
- [x] Deploy to portainer 
- Configure fluent-bit on pi for shipping logs to my logflow service
