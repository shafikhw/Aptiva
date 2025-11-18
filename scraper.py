import json
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv
import pandas as pd
from apify_client import ApifyClient


load_dotenv()
DEFAULT_ACTOR_ID = "BvepfniODI2AixuNN"


def _get_client() -> ApifyClient:
    token = os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN")
    if not token:
        raise RuntimeError("APIFY_API_TOKEN (or APIFY_TOKEN) is required to run the Apartments.com scraper.")
    return ApifyClient(token)


def fetch_listings_from_actor(
    run_input: Dict,
    *,
    actor_id: Optional[str] = None,
) -> List[Dict]:
    """Run the configured Apify actor and return listing items as dictionaries."""

    client = _get_client()
    actor_client = client.actor(actor_id or os.getenv("APIFY_ACTOR_ID", DEFAULT_ACTOR_ID))
    run_result = actor_client.call(run_input=run_input)

    dataset_id = run_result.get("defaultDatasetId")
    if not dataset_id:
        raise RuntimeError("Actor did not return defaultDatasetId; check the actor output/storage method.")

    dataset_client = client.dataset(dataset_id)
    items_result = dataset_client.list_items()
    return items_result.items


# def run_actor_and_save_outputs(run_input, json_path="output.json", csv_path="output.csv"):
def run_actor_and_save_outputs(run_input, json_path="output.json"):
    """Backward-compatible helper that also saves scraped items to disk."""

    items = fetch_listings_from_actor(run_input)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} items to {json_path}")

    # df = pd.DataFrame(items)
    # df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    # print(f"Saved {len(items)} items to {csv_path}")

    # xlsx_path = csv_path.replace(".csv", ".xlsx")
    # df.to_excel(xlsx_path, index=False)
    # print(f"Saved {len(items)} items to {xlsx_path}")


if __name__ == "__main__":
    # Example input for your actor — change to whatever the actor expects
    example_input = {
        "search_url": "https://www.apartments.com/off-campus-housing/ca/los-angeles/university-of-southern-california-university-park-campus/",
        "max_pages": 1,  # 1 to 5
        "filter_option": "all",  # "all", "bed0" , "bed1" , "bed2" , "bed3" , "bed4" , "bed5"
    }
    # run_actor_and_save_outputs(example_input, json_path="actor_output.json", csv_path="actor_output.csv")
    run_actor_and_save_outputs(example_input, json_path="actor_output.json")
