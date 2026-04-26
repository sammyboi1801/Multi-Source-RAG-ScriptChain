"""
Supabase Multi-Source Scraper (v3)
Usage:
    python scraper.py               # scrape all three sources
    python scraper.py --only blogs  # scrape blogs only
    python scraper.py --only docs   # scrape docs only
    python scraper.py --only forums # scrape forums only
"""

import requests
import json
import time
import os
import re
import sys

OUTPUT_DIR = "./data"
os.makedirs(f"{OUTPUT_DIR}/docs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/blogs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/forums", exist_ok=True)

GITHUB_TOKEN = None  # Optional: "ghp_xxx" for higher rate limits

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved {len(data)} items → {path}")

def parse_mdx(raw):
    """Strip frontmatter and JSX from MDX, return (title, date, content)"""
    title, date = "", ""

    fm = re.search(r'^---\n([\s\S]*?)\n---', raw)
    if fm:
        fm_block = fm.group(1)
        t = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', fm_block, re.MULTILINE)
        d = re.search(r'^date:\s*["\']?(.+?)["\']?\s*$', fm_block, re.MULTILINE)
        title = t.group(1).strip('"\'') if t else ""
        date  = d.group(1).strip('"\'') if d else ""

    content = re.sub(r'^---[\s\S]*?---\n', '', raw).strip()
    content = re.sub(r'^import\s+.*?\n', '', content, flags=re.MULTILINE)
    # JSX components: <Foo .../> self-closing and <Foo>...</Foo> paired
    content = re.sub(r'<[A-Z][a-zA-Z]*[^>]*?\/>', '', content)
    content = re.sub(r'<[A-Z][a-zA-Z]*[^>]*>[\s\S]*?<\/[A-Z][a-zA-Z]*>', '', content)
    content = re.sub(r'^export\s+.*?\n', '', content, flags=re.MULTILINE)
    return title, date, clean_text(content)

# --- docs ---

DOC_FILES = [
    ("getting_started",         "apps/docs/content/guides/getting-started/features.mdx"),
    ("quickstart_nextjs",       "apps/docs/content/guides/getting-started/quickstarts/nextjs.mdx"),
    ("quickstart_react",        "apps/docs/content/guides/getting-started/quickstarts/reactjs.mdx"),
    ("quickstart_python",       "apps/docs/content/guides/getting-started/quickstarts/python.mdx"),
    ("quickstart_flutter",      "apps/docs/content/guides/getting-started/quickstarts/flutter.mdx"),
    ("auth_overview",           "apps/docs/content/guides/auth.mdx"),
    ("auth_email",              "apps/docs/content/guides/auth/auth-email.mdx"),
    ("auth_social_login",       "apps/docs/content/guides/auth/social-login.mdx"),
    ("auth_rls",                "apps/docs/content/guides/auth/row-level-security.mdx"),
    ("auth_users",              "apps/docs/content/guides/auth/managing-user-data.mdx"),
    ("auth_sessions",           "apps/docs/content/guides/auth/sessions.mdx"),
    ("auth_mfa",                "apps/docs/content/guides/auth/auth-mfa.mdx"),
    ("database_overview",       "apps/docs/content/guides/database/overview.mdx"),
    ("database_tables",         "apps/docs/content/guides/database/tables.mdx"),
    ("database_functions",      "apps/docs/content/guides/database/functions.mdx"),
    ("database_full_text",      "apps/docs/content/guides/database/full-text-search.mdx"),
    ("database_extensions",     "apps/docs/content/guides/database/extensions/overview.mdx"),
    ("database_rls",            "apps/docs/content/guides/database/postgres/row-level-security.mdx"),
    ("storage_overview",        "apps/docs/content/guides/storage.mdx"),
    ("storage_uploads",         "apps/docs/content/guides/storage/uploads.mdx"),
    ("storage_access_control",  "apps/docs/content/guides/storage/access-control.mdx"),
    ("storage_cdn",             "apps/docs/content/guides/storage/cdn/fundamentals.mdx"),
    ("realtime_overview",       "apps/docs/content/guides/realtime.mdx"),
    ("realtime_broadcast",      "apps/docs/content/guides/realtime/broadcast.mdx"),
    ("realtime_postgres",       "apps/docs/content/guides/realtime/postgres-changes.mdx"),
    ("realtime_presence",       "apps/docs/content/guides/realtime/presence.mdx"),
    ("edge_functions_overview", "apps/docs/content/guides/functions.mdx"),
    ("edge_functions_start",    "apps/docs/content/guides/functions/quickstart.mdx"),
    ("edge_functions_auth",     "apps/docs/content/guides/functions/auth.mdx"),
    ("cli_overview",            "apps/docs/content/guides/cli.mdx"),
    ("local_dev",               "apps/docs/content/guides/cli/local-development.mdx"),
    ("api_overview",            "apps/docs/content/guides/api.mdx"),
    ("api_rest",                "apps/docs/content/guides/api/rest/index.mdx"),
]

BASE_RAW = "https://raw.githubusercontent.com/supabase/supabase/master"

def scrape_docs():
    print("\n📚 Scraping Documentation (via GitHub raw)...")
    docs = []
    gh_headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        gh_headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    for slug, path in DOC_FILES:
        url = f"{BASE_RAW}/{path}"
        try:
            resp = requests.get(url, headers=gh_headers, timeout=10)
            if resp.status_code == 200:
                title, date, content = parse_mdx(resp.text)
                if len(content) > 150:
                    docs.append({
                        "id": slug,
                        "source_type": "documentation",
                        "url": f"https://supabase.com/docs/guides/{path.split('guides/')[-1].replace('.mdx','')}",
                        "title": title or slug.replace("_", " ").title(),
                        "content": content
                    })
                    print(f"  ✔ {slug} ({len(content)} chars)")
                else:
                    print(f"  ✗ {slug} - too short ({len(content)} chars)")
            else:
                print(f"  ✗ {slug} - HTTP {resp.status_code}")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ✗ {slug} - {e}")

    save_json(docs, f"{OUTPUT_DIR}/docs/supabase_docs.json")
    return docs

# --- blogs ---

# Format: (slug_id, exact_filename_without_extension)
# Verified against: github.com/supabase/supabase/tree/master/apps/www/_blog
BLOG_FILES = [
    ("supabase_local_dev",          "2023-08-08-supabase-local-dev"),
    ("realtime_rls",                "2021-12-01-realtime-row-level-security-in-postgresql"),
    ("storage_overview",            "2021-03-30-supabase-storage"),
    ("storage_beta",                "2021-07-27-storage-beta"),
    ("edge_functions_overview",     "2022-03-31-supabase-edge-functions"),
    ("realtime_multiplayer",        "2022-04-01-supabase-realtime-with-multiplayer-features"),
    ("auth_enterprise",             "2022-03-30-supabase-enterprise"),
    ("postgres_audit",              "2022-03-08-postgres-audit"),
    ("pg_graphql",                  "2021-12-03-pg-graphql"),
    ("graphql_available",           "2022-03-29-graphql-now-available"),
    ("postgrest_9",                 "2021-11-28-postgrest-9"),
    ("auth_sms_login",              "2021-07-28-supabase-auth-passwordless-sms-login"),
    ("postgres_roles_hooks",        "2021-07-01-roles-postgres-hooks"),
    ("supabase_reports",            "2021-07-29-supabase-reports-and-metrics"),
    ("whats_new_postgres14",        "2021-11-28-whats-new-in-postgres-14"),
    ("functions_updates",           "2021-07-30-supabase-functions-updates"),
    ("supabase_studio",             "2021-11-30-supabase-studio"),
    ("supabase_cli",                "2021-03-31-supabase-cli"),
    ("postgres_cron",               "2021-03-05-postgres-as-a-cron-server"),
    ("postgres_views",              "2020-11-18-postgresql-views"),
    ("postgres_backups",            "2020-07-17-postgresql-physical-logical-backups"),
    ("cracking_postgres",           "2021-02-27-cracking-postgres-interview"),
    ("supabase_series_a",           "2021-10-28-supabase-series-a"),
    ("beta_update_march22",         "2022-04-15-beta-update-march-2022"),
    ("supabase_pgbouncer",          "2021-04-02-supabase-pgbouncer"),
]

BLOG_BASE = "https://raw.githubusercontent.com/supabase/supabase/master/apps/www/_blog"

def try_blog_url(slug_id, filename):
    """Try fetching a blog post, with fallback filename variants"""
    candidates = [
        f"{BLOG_BASE}/{filename}.mdx",
        f"{BLOG_BASE}/{filename}.md",
    ]
    last_status = None
    for url in candidates:
        try:
            resp = requests.get(url, timeout=10)
            last_status = resp.status_code
            if resp.status_code == 200 and len(resp.text) > 200:
                return url, resp.text
        except Exception as e:
            print(f"  ⚠ Network error fetching {url}: {e}")
    if last_status:
        print(f"  ✗ {filename} - HTTP {last_status}")
    return None, None

def scrape_blogs():
    print("\n📝 Scraping Blog Posts (via GitHub raw)...")
    blogs = []

    for slug_id, filename in BLOG_FILES:
        url, raw = try_blog_url(slug_id, filename)
        if raw:
            title, date, content = parse_mdx(raw)
            if len(content) > 300:
                blogs.append({
                    "id": slug_id,
                    "source_type": "blog",
                    "url": f"https://supabase.com/blog/{filename[11:]}",  # strip YYYY-MM-DD- prefix
                    "github_raw_url": url,
                    "title": title or filename.replace("-", " ").title(),
                    "date": date,
                    "content": content
                })
                print(f"  ✔ {(title or filename)[:60]} ({len(content)} chars)")
            else:
                print(f"  ✗ {filename} - too short ({len(content)} chars)")
        else:
            print(f"      → verify at github.com/supabase/supabase/tree/master/apps/www/_blog")
        time.sleep(0.3)

    save_json(blogs, f"{OUTPUT_DIR}/blogs/supabase_blogs.json")
    return blogs

# --- forums ---

def fetch_github_discussions(max_discussions=50):
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    discussions = []
    seen_ids = set()

    search_queries = [
        "repo:supabase/supabase label:bug is:issue",
        "repo:supabase/supabase label:question is:issue",
        "repo:supabase/supabase auth error in:title is:issue",
        "repo:supabase/supabase storage upload in:title is:issue",
        "repo:supabase/supabase realtime not working in:title is:issue",
    ]

    for query in search_queries:
        if len(discussions) >= max_discussions:
            break
        try:
            resp = requests.get(
                "https://api.github.com/search/issues",
                headers=headers,
                params={"q": query, "per_page": 10, "sort": "comments"},
                timeout=10
            )
            if resp.status_code == 403:
                print("  ⚠ GitHub rate limit. Set GITHUB_TOKEN for more.")
                break
            if resp.status_code != 200:
                print(f"  ✗ GitHub API {resp.status_code}")
                continue

            for item in resp.json().get("items", []):
                if item["id"] in seen_ids:
                    continue
                seen_ids.add(item["id"])

                comments_data = []
                if item["comments"] > 0:
                    c_resp = requests.get(
                        item["comments_url"],
                        headers=headers,
                        params={"per_page": 5},
                        timeout=10
                    )
                    if c_resp.status_code == 200:
                        for c in c_resp.json():
                            comments_data.append({
                                "author": c["user"]["login"],
                                "body": clean_text(c["body"]),
                                "created_at": c["created_at"]
                            })
                    time.sleep(0.5)

                discussions.append({
                    "id": f"gh_issue_{item['number']}",
                    "source_type": "forum",
                    "url": item["html_url"],
                    "title": item["title"],
                    "question": clean_text(item["body"] or ""),
                    "author": item["user"]["login"],
                    "created_at": item["created_at"],
                    "state": item["state"],
                    "labels": [l["name"] for l in item.get("labels", [])],
                    "comments": comments_data,
                    "comment_count": item["comments"]
                })
                print(f"  ✔ #{item['number']}: {item['title'][:60]}")

            time.sleep(2)
        except Exception as e:
            print(f"  ✗ {e}")

    return discussions

def scrape_forums():
    print("\n💬 Scraping Forums (GitHub Issues)...")
    discussions = fetch_github_discussions(max_discussions=50)
    save_json(discussions, f"{OUTPUT_DIR}/forums/supabase_forums.json")
    return discussions

# --- main ---

def print_summary(docs, blogs, forums):
    print("\n" + "="*50)
    print("📊 SCRAPING COMPLETE - SUMMARY")
    print("="*50)
    print(f"  📚 Documentation pages : {len(docs)}")
    print(f"  📝 Blog posts          : {len(blogs)}")
    print(f"  💬 Forum threads       : {len(forums)}")
    total = (
        sum(len(d.get("content","")) for d in docs) +
        sum(len(b.get("content","")) for b in blogs) +
        sum(len(f.get("question","")) for f in forums)
    )
    print(f"  📦 Total content       : ~{total:,} characters")
    print("="*50)

if __name__ == "__main__":
    # Parse --only flag
    only = None
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = sys.argv[idx + 1].lower()

    print("🚀 Supabase Multi-Source Scraper v3")
    print("="*50)

    docs, blogs, forums = [], [], []

    if only == "docs":
        docs = scrape_docs()
    elif only == "blogs":
        blogs = scrape_blogs()
    elif only == "forums":
        forums = scrape_forums()
    else:
        docs   = scrape_docs()
        blogs  = scrape_blogs()
        forums = scrape_forums()

    print_summary(docs, blogs, forums)