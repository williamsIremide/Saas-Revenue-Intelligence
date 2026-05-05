import os, json
from upstash_redis import Redis
from dotenv import load_dotenv

load_dotenv()

USE_CACHE = True

CACHE_VERSION = "v1"

upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

if not upstash_url or not upstash_token:
    raise ValueError("UPSTASH_REDIS_REST_URL is not set")

cache =  Redis(url=upstash_url, token=upstash_token)
CACHE_TTL = 86400


def get_cache(key: str) -> dict | None:
    val = cache.get(key)
    if val is None:
        return None
    return val if isinstance(val, dict) else json.loads(val)


def set_cache(key: str, value: dict) -> None:
    cache.setex(key, CACHE_TTL, json.dumps(value))