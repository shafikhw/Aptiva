# test_scrape_carson_crossing.py
from scraper import run_actor_and_save_outputs

if __name__ == "__main__":
    run_actor_and_save_outputs(
        {
            "search_url": "https://www.apartments.com/carson-crossing-austin-tx/vj5zelf/",
            "max_pages": 1,
            "filter_option": "all",
        },
        json_path="carson_crossing_output.json",
    )
    print("Saved Carson Crossing data to carson_crossing_output.json")
