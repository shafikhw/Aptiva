import requests

URL = "https://www.apartments.com/4901-s-drexel-blvd-chicago-il/xveplbn/"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

print("Fetching HTML...")

r = requests.get(URL, headers=headers)

if r.status_code != 200:
    print("Error:", r.status_code)
    exit()

with open("xveplbn_raw.html", "w", encoding="utf-8") as f:
    f.write(r.text)

print("Saved xveplbn_raw.html successfully!")
