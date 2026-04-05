import httpx
from bs4 import BeautifulSoup


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
        # 1. Extract slug
        slug = domain.replace("https://", "").replace("http://", "").split(".")[0]
        print("Slug:", slug)

        greenhouse_url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=false"
        lever_url = f"https://jobs.lever.co/{slug}"

        async with httpx.AsyncClient(
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        ) as client:

            job_titles = []
            source = "none"

            # 2. Try Greenhouse
            response = await client.get(greenhouse_url)
            print("Greenhouse URL:", greenhouse_url)
            print("Greenhouse status:", response.status_code)
            print("Greenhouse text (first 200):", response.text[:200])

            if response.status_code == 200:
                data = response.json()
                jobs = data.get("jobs", [])
                job_titles = [job.get("title", "") for job in jobs]
                source = "greenhouse"

            # 3. Fallback to Lever
            elif response.status_code == 404:
                print("Falling back to Lever:", lever_url)
                response = await client.get(lever_url)
                print("Lever status:", response.status_code)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    job_elements = soup.find_all(class_="posting-name")
                    job_titles = [el.get_text(strip=True) for el in job_elements]
                    source = "lever"
                else:
                    return empty_result
            else:
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
    

