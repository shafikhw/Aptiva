import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Dict, List, Optional
from dotenv import load_dotenv
import pandas as pd
from apify_client import ApifyClient
from telemetry.metrics import log_metric
from telemetry.logging_utils import get_logger
from telemetry.retry import retry_with_backoff

logger = get_logger(__name__)
APIFY_TIMEOUT_SECONDS = int(os.getenv("APIFY_TIMEOUT_SECONDS", "120"))
APIFY_MAX_RETRIES = int(os.getenv("APIFY_MAX_RETRIES", "2"))


load_dotenv()
DEFAULT_ACTOR_ID = "BvepfniODI2AixuNN"



APIFY_API_TOKEN= os.getenv("APIFY_API_TOKEN")
ACTOR_ID = "BvepfniODI2AixuNN" 
def _get_client() -> ApifyClient:
    token = os.getenv("APIFY_API_TOKEN") or os.getenv("APIFY_TOKEN")
    if not token:
        raise RuntimeError("APIFY_API_TOKEN (or APIFY_TOKEN) is required to run the Apartments.com scraper.")
    return ApifyClient(token)


def fetch_listings_from_actor(
    run_input: Dict,
    *,
    actor_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> List[Dict]:
    """Run the configured Apify actor and return listing items as dictionaries."""

    start = time.perf_counter()
    actor_name = actor_id or os.getenv("APIFY_ACTOR_ID", DEFAULT_ACTOR_ID)
    logger.info(
        "apify_call_start",
        extra={"actor_id": actor_name, "conversation_id": conversation_id, "run_input_keys": list(run_input.keys())},
    )
    try:
        def _run_once():
            client = _get_client()
            actor_client = client.actor(actor_name)
            with ThreadPoolExecutor(max_workers=1) as executor:
                run_result = executor.submit(lambda: actor_client.call(run_input=run_input)).result(
                    timeout=APIFY_TIMEOUT_SECONDS
                )

            dataset_id = run_result.get("defaultDatasetId")
            if not dataset_id:
                raise RuntimeError("Actor did not return defaultDatasetId; check the actor output/storage method.")

            dataset_client = client.dataset(dataset_id)
            with ThreadPoolExecutor(max_workers=1) as executor:
                items_result = executor.submit(lambda: dataset_client.list_items()).result(
                    timeout=APIFY_TIMEOUT_SECONDS
                )
            return items_result.items

        items = retry_with_backoff(
            _run_once,
            retries=APIFY_MAX_RETRIES,
            retry_exceptions=(Exception, TimeoutError),
        )
        logger.info(
            "apify_call_complete",
            extra={"actor_id": actor_name, "conversation_id": conversation_id, "items": len(items)},
        )
        return items
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        log_metric(
            "apify",
            actor_name,
            latency_ms=latency_ms,
            conversation_id=conversation_id,
        )


# def run_actor_and_save_outputs(run_input, json_path="output.json", csv_path="output.csv"):
def run_actor_and_save_outputs(run_input, json_path="output.json", conversation_id: Optional[str] = None):
    """Backward-compatible helper that also saves scraped items to disk."""

    items = fetch_listings_from_actor(run_input, conversation_id=conversation_id)

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
