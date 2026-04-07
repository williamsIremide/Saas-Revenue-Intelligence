import asyncio
import httpx
from bs4 import BeautifulSoup
from time import time

# -------------------------
# Simple Cache (Redis-like)
# -------------------------
CACHE = {}
CACHE_TTL = 300  # 5 minutes


def get_cache(key):
    if key in CACHE:
        data, expiry = CACHE[key]
        if time() < expiry:
            return data
        del CACHE[key]
    return None


def set_cache(key, value):
    CACHE[key] = (value, time() + CACHE_TTL)


# -------------------------
# ATS Detection
# -------------------------
async def detect_ats(client, domain: str) -> str:
    urls = [
        f"https://{domain}",
        f"https://{domain}/careers",
        f"https://{domain}/jobs",
    ]

    for url in urls:
        try:
            res = await client.get(url)
            text = res.text[:5000].lower()  # Check only the first 5k chars for efficiency

            if "greenhouse.io" in text:
                return "greenhouse"
            if "lever.co" in text:
                return "lever"
            if "ashbyhq.com" in text:
                return "ashby"
            if "workable.com" in text:
                return "workable"
            if "bamboohr.com" in text:
                return "bamboohr"
            if "rippling.com" in text:
                return "rippling"

        except:
            continue

    return "unknown"


# -------------------------
# Fetchers
# -------------------------
async def fetch_greenhouse(client, slug):
    try:
        url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=false"
        res = await client.get(url)
        if res.status_code == 200:
            jobs = res.json().get("jobs", [])
            return [j.get("title", "") for j in jobs], "greenhouse"
        return [], None
    except:
        return [], None


async def fetch_lever(client, slug):
    try:
        url = f"https://jobs.lever.co/{slug}"
        res = await client.get(url)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            jobs = [el.get_text(strip=True) for el in soup.find_all(class_="posting-name")]
            return jobs, "lever"
        return [], None
    except:
        return [], None


async def fetch_workable(client, slug):
    try:
        url = f"https://{slug}.workable.com/api/v3/jobs"
        res = await client.get(url)
        if res.status_code == 200:
            jobs = res.json().get("jobs", [])
            return [j.get("title", "") for j in jobs], "workable"
        return [], None
    except:
        return [], None
    

async def fetch_ashby(client, slug):
    try:
        url = f"https://jobs.ashbyhq.com/{slug}"
        res = await client.get(url)

        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")

            # Ashby usually uses these classes
            job_elements = soup.find_all("a", href=True)
            jobs = [
                el.get_text(strip=True)
                for el in job_elements
                if "/jobs/" in (el.get("href") or "").lower()
            ]

            return jobs, "ashby"

    except:
        pass

    return [], None


async def fetch_bamboohr(client, slug):
    try:
        url = f"https://{slug}.bamboohr.com/jobs/"
        res = await client.get(url)

        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")

            job_elements = soup.find_all(class_="BambooHR-ATS-Jobs-Item")
            jobs = [el.get_text(strip=True) for el in job_elements]

            return jobs, "bamboohr"

    except:
        pass

    return [], None


async def fetch_rippling(client, slug):
    try:
        url = f"https://ats.rippling.com/{slug}/jobs"
        res = await client.get(url)

        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")

            job_elements = soup.find_all("a")
            jobs = [el.get_text(strip=True) for el in job_elements if el.get_text(strip=True)]

            return jobs, "rippling"

    except:
        pass

    return [], None


# -------------------------
# Main Engine
# -------------------------
async def get_jobs(domain: str):
    normalized_domain = domain.replace("https://", "").replace("http://", "").replace("www.", "")

    # 🔹 Cache check
    cached = get_cache(normalized_domain)
    if cached:
        print("Cache hit:", domain)
        return cached

    slug = domain.replace("https://", "").replace("http://", "").replace("www.", "").split(".")[0]

    async with httpx.AsyncClient(
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0"}
    ) as client:

        # 1. Detect ATS
        ats = await detect_ats(client, domain)
        print("Detected ATS:", ats)

        # 2. If confident → call ONE source
        if ats == "greenhouse":
            jobs, source = await fetch_greenhouse(client, slug)
            if not jobs:
                ats = "unknown"

        elif ats == "lever":
            jobs, source = await fetch_lever(client, slug)
            if not jobs:
                ats = "unknown"

        elif ats == "workable":
            jobs, source = await fetch_workable(client, slug)
            if not jobs:
                ats = "unknown"
        
        elif ats == "ashby":
            jobs, source = await fetch_ashby(client, slug)
            if not jobs:
                ats = "unknown"

        elif ats == "bamboohr":
            jobs, source = await fetch_bamboohr(client, slug)
            if not jobs:
                ats = "unknown"

        elif ats == "rippling":
            jobs, source = await fetch_rippling(client, slug)
            if not jobs:
                ats = "unknown"

        else:
            # 3. Parallel fallback
            tasks = [
                fetch_greenhouse(client, slug),
                fetch_lever(client, slug),
                fetch_workable(client, slug),
                fetch_ashby(client, slug),
                fetch_bamboohr(client, slug),
                fetch_rippling(client, slug)
            ]

            for coro in asyncio.as_completed(tasks):
                jobs, source = await coro
                if jobs:
                    break

    result = {
        "titles": jobs,
        "source": source
    }

    # 🔹 Cache result
    set_cache(normalized_domain, result)

    return result


async def get_hiring_signal(domain: str) -> dict:
    empty_result = {
        "open_roles": 0,
        "engineering_roles": 0,
        "sales_roles": 0,
        "source": "none",
        "velocity_score": 0.0,
        "raw_titles": []
    }

    try:
        jobs_data = await get_jobs(domain)

        job_titles = jobs_data["titles"]
        source = jobs_data["source"]

        if not job_titles:
            return empty_result

        # 4. Counts
        open_roles = len(job_titles)

        engineering_keywords = [
            "engineer", "developer", "backend", "frontend",
            "full stack", "software", "data", "machine learning", "ai"
        ]

        sales_keywords = [
            "sales", "account executive", "account manager",
            "sales manager", "revenue"
        ]

        engineering_roles = sum(
            1 for title in job_titles
            if any(k in title.lower() for k in engineering_keywords)
        )

        sales_roles = sum(
            1 for title in job_titles
            if any(k in title.lower() for k in sales_keywords)
        )

        # 5. Velocity score
        if open_roles == 0:
            velocity_score = 0.0
        elif 1 <= open_roles <= 5:
            velocity_score = 0.2
        elif 6 <= open_roles <= 15:
            velocity_score = 0.4
        elif 16 <= open_roles <= 30:
            velocity_score = 0.6
        elif 31 <= open_roles <= 60:
            velocity_score = 0.8
        else:
            velocity_score = 1.0

        return {
            "open_roles": open_roles,
            "engineering_roles": engineering_roles,
            "sales_roles": sales_roles,
            "source": source,
            "velocity_score": velocity_score,
            "raw_titles": job_titles
        }

    except Exception:
        return empty_result
    

