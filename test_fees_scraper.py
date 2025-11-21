import os
import json
from apify_client import ApifyClient

# ----------------------------
# CONFIG
# ----------------------------
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
ACTOR_ID = "BvepfniODI2AixuNN"

# IMPORTANT:
# Use a *search results* URL, not a single-property URL.
# You can paste any Apartments.com search URL you like.
TEST_SEARCH_URL = "https://www.apartments.com/chicago-il/1-bedrooms/"


def main():
    print("Running test scraper...\n")

    client = ApifyClient(APIFY_API_TOKEN)
    actor = client.actor(ACTOR_ID)

    # This matches how your real project calls the actor: `search_url`, `max_pages`, `filter_option`
    run_input = {
        "search_url": TEST_SEARCH_URL,
        "max_pages": 1,
        "filter_option": "all",
        # NOTE: We *could* add extendOutputFunction here,
        # but the actor already includes `feesAndPolicies` by default,
        # according to its official docs.
    }

    print("Starting Apify actor run...\n")
    run = actor.call(run_input=run_input)

    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        print("Actor returned no dataset (no defaultDatasetId).")
        return

    dataset = client.dataset(dataset_id)
    items = dataset.list_items().items

    print(f"Scraped {len(items)} items.\n")

    if not items:
        print("No items returned. Try a different Apartments.com SEARCH url.")
    else:
        # Look at the first item only (to keep output readable)
        first = items[0]
        print("First listing keys:\n", sorted(first.keys()))
        print("\nURL:", first.get("url"))

        # This is the field we care about:
        print("\n==== RAW feesAndPolicies FIELD ====\n")
        print(json.dumps(first.get("feesAndPolicies"), indent=2, ensure_ascii=False))

    # Save everything so you can inspect in an editor
    with open("test_output.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print("\nSaved full data to test_output.json")
    print("Test complete.")


if __name__ == "__main__":
    main()
