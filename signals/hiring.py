import asyncio, os, json
import httpx
from bs4 import BeautifulSoup
from signals.cache import get_cache, set_cache, USE_CACHE, CACHE_VERSION


# ── Fetchers ──────────────────────────────────────────────────────────────────

async def fetch_greenhouse(client: httpx.AsyncClient, slug: str) -> tuple[list, str | None]:
    try:
        url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=false"
        res = await client.get(url)
        if res.status_code == 200:
            jobs = res.json().get("jobs", [])
            return [j.get("title", "") for j in jobs], "greenhouse"
        return [], None
    except Exception as e:
        print(f"[hiring/greenhouse] {slug}: {e}")
        return [], None


async def fetch_lever(client: httpx.AsyncClient, slug: str) -> tuple[list, str | None]:
    try:
        url = f"https://jobs.lever.co/{slug}"
        res = await client.get(url)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            jobs = [el.get_text(strip=True) for el in soup.find_all(class_="posting-name")]
            return jobs, "lever"
        return [], None
    except Exception as e:
        print(f"[hiring/lever] {slug}: {e}")
        return [], None


async def fetch_workable(client: httpx.AsyncClient, slug: str) -> tuple[list, str | None]:
    try:
        url = f"https://{slug}.workable.com/api/v3/jobs"
        res = await client.get(url)
        if res.status_code == 200:
            jobs = res.json().get("jobs", [])
            return [j.get("title", "") for j in jobs], "workable"
        return [], None
    except Exception as e:
        print(f"[hiring/workable] {slug}: {e}")
        return [], None


async def fetch_ashby(client: httpx.AsyncClient, slug: str) -> tuple[list, str | None]:
    try:
        res = await client.post(
            "https://jobs.ashbyhq.com/api/non-user-graphql",
            json={
                "operationName": "ApiJobBoardWithTeams",
                "variables": {"organizationHostedJobsPageName": slug},
                "query": """query ApiJobBoardWithTeams($organizationHostedJobsPageName: String!) {
                    jobBoard: jobBoardWithTeams(organizationHostedJobsPageName: $organizationHostedJobsPageName) {
                        jobPostings { title }
                    }
                }""",
            },
        )
        if res.status_code == 200:
            data = res.json()
            job_board = (data.get("data") or {}).get("jobBoard") or {}
            postings = job_board.get("jobPostings") or []
            jobs = [p.get("title", "") for p in postings if p.get("title")]
            return jobs, "ashby"
    except Exception as e:
        print(f"[hiring/ashby] {slug}: {e}")
    return [], None


async def fetch_bamboohr(client: httpx.AsyncClient, slug: str) -> tuple[list, str | None]:
    try:
        url = f"https://{slug}.bamboohr.com/jobs/"
        res = await client.get(url)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            job_elements = soup.find_all(class_="BambooHR-ATS-Jobs-Item")
            jobs = [el.get_text(strip=True) for el in job_elements]
            return jobs, "bamboohr"
    except Exception as e:
        print(f"[hiring/bamboohr] {slug}: {e}")
    return [], None


async def fetch_rippling(client: httpx.AsyncClient, slug: str) -> tuple[list, str | None]:
    try:
        url = f"https://ats.rippling.com/{slug}/jobs"
        res = await client.get(url)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            job_elements = soup.find_all("a")
            jobs = [el.get_text(strip=True) for el in job_elements if el.get_text(strip=True)]
            return jobs, "rippling"
    except Exception as e:
        print(f"[hiring/rippling] {slug}: {e}")
    return [], None


# ── Main engine ───────────────────────────────────────────────────────────────

async def get_jobs(domain: str) -> dict:
    normalized_domain = (
        domain.strip()
        .replace("https://", "")
        .replace("http://", "")
        .replace("www.", "")
        .lower()
    )

    cache_key = f"hiring:{CACHE_VERSION}:{normalized_domain}"
    cached = get_cache(cache_key)
    if USE_CACHE and cached:
        print(f"[hiring] cache hit: {normalized_domain}")
        return cached

    slug = normalized_domain.split(".")[0]

    async with httpx.AsyncClient(
        timeout=10,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as client:
        fetchers = [
            fetch_ashby,
            fetch_greenhouse,
            fetch_lever,
            fetch_workable,
            fetch_bamboohr,
            fetch_rippling,
        ]

        tasks = [asyncio.create_task(fetcher(client, slug)) for fetcher in fetchers]

        jobs, source = [], "none"

        try:
            for task in asyncio.as_completed(tasks):
                try:
                    result_jobs, result_source = await task
                    if result_jobs:
                        jobs, source = result_jobs, result_source
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        break
                except Exception as e:
                    print(f"[hiring] task error: {e}")
                    continue
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

    result = {"titles": jobs, "source": source}

    if jobs:
        set_cache(cache_key, result)

    return result


async def get_hiring_signal(domain: str) -> dict:
    empty_result = {
        "open_roles":       0,
        "engineering_roles": 0,
        "sales_roles":      0,
        "source":           "none",
        "velocity_score":   0.0,
        "raw_titles":       [],
    }

    try:
        jobs_data  = await get_jobs(domain)
        job_titles = jobs_data["titles"]
        source     = jobs_data["source"]

        if not job_titles:
            return empty_result

        open_roles = len(job_titles)

        engineering_keywords = [
            "engineer", "developer", "backend", "frontend",
            "full stack", "software", "data", "machine learning", "ai",
        ]
        sales_keywords = [
            "sales", "account executive", "account manager",
            "sales manager", "revenue",
        ]

        engineering_roles = sum(
            1 for title in job_titles
            if any(k in title.lower() for k in engineering_keywords)
        )
        sales_roles = sum(
            1 for title in job_titles
            if any(k in title.lower() for k in sales_keywords)
        )

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
            "open_roles":       open_roles,
            "engineering_roles": engineering_roles,
            "sales_roles":      sales_roles,
            "source":           source,
            "velocity_score":   velocity_score,
            "raw_titles":       job_titles,
        }

    except Exception as e:
        print(f"[hiring] get_hiring_signal error: {e}")
        return empty_result