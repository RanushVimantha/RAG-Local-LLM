# Weather Monitoring API Documentation

## Overview

The Weather Monitoring API provides real-time weather data for cities worldwide. All endpoints return JSON responses and require an API key for authentication.

## Base URL

```
https://api.weather-monitor.example.com/v1
```

## Authentication

Include your API key in the `X-API-Key` header:

```
X-API-Key: your_api_key_here
```

## Endpoints

### GET /current/{city}

Returns current weather conditions for a specified city.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| city | string | Yes | City name or city ID |
| units | string | No | Temperature units: "metric" (default), "imperial" |

**Response:**
```json
{
  "city": "Colombo",
  "temperature": 31.5,
  "humidity": 78,
  "wind_speed": 12.3,
  "condition": "Partly Cloudy",
  "updated_at": "2026-04-06T10:30:00Z"
}
```

### GET /forecast/{city}

Returns a 7-day weather forecast.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| city | string | Yes | City name or city ID |
| days | integer | No | Number of forecast days (1-14, default: 7) |

**Response:**
```json
{
  "city": "Colombo",
  "forecast": [
    {
      "date": "2026-04-07",
      "high": 33.0,
      "low": 25.5,
      "condition": "Thunderstorms",
      "precipitation_chance": 85
    }
  ]
}
```

## Rate Limits

- Free tier: 100 requests per hour
- Pro tier: 10,000 requests per hour
- Enterprise tier: Unlimited

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad request — invalid parameters |
| 401 | Unauthorized — invalid or missing API key |
| 404 | City not found |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
