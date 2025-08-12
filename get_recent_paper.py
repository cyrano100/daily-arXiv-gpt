import argparse
import datetime
import json
import os
import re
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree as ET

import pdfplumber
import requests
from dotenv import load_dotenv


def call_qwen(messages, model="qwen-max", temperature=0.0, max_tokens=1024):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请在 .env 中设置。")

    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def sanitize_filename(name: str) -> str:
    name = name.strip().strip('"').strip("'")
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name or "keywords"


def save_results(root_dir: str, keyword: str, data: list):
    """
    将结果保存到 root_dir/keyword.json
    """
    root = Path(root_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    fname = f"{sanitize_filename(keyword)}.json"
    fpath = root / fname
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(fpath)


def extract_affiliations_for_authors(pdf_url: str, authors_from_api: list):
    """
    参数:
      - pdf_url: 论文 PDF 链接
      - authors_from_api: 从 arXiv 得到的作者列表
    返回:
      - [{"name": <author>, "affiliations": [..]}, ...]
    """
    try:
        r = requests.get(pdf_url, timeout=60)
        r.raise_for_status()
        pdf_content = BytesIO(r.content)
        with pdfplumber.open(pdf_content) as pdf:
            first_page = pdf.pages[0]
            text = first_page.extract_text() or ""
    except Exception:
        return [{"name": a, "affiliations": []} for a in authors_from_api]

    text = re.sub(r"\s+", " ", text).strip()
    # text = text[:16000]

    system = (
        "You are a precise scholarly metadata linker. "
        "You will be given: (1) a fixed ordered list of authors from arXiv (authoritative), "
        "and (2) the first-page text of the paper PDF. "
        "Your job is to map each GIVEN author to their institutional affiliations found in the text. "
        "RULES:\n"
        "- DO NOT add, remove, rename, or reorder authors.\n"
        "- If you are unsure for an author, return an empty array for that author's affiliations.\n"
        "- Extract institution-level names (university, lab, company). Ignore emails and footnote symbols.\n"
        "- If superscripts/markers are present, use them to bind authors to affiliations.\n"
        "Return STRICT JSON only with this schema:\n"
        "{ \"authors\": [ {\"name\": \"...\", \"affiliations\": [\"...\"]} ] }\n"
        "The order must match exactly the order of the provided author list."
    )
    user = (
        "Author list (authoritative, ordered):\n"
        + json.dumps(authors_from_api, ensure_ascii=False)
        + "\n\nFirst-page text:\n"
        + text
        + "\n\nNow output JSON only."
    )

    try:
        content = call_qwen(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=1200,
        )
        cleaned = content.strip().strip("`")
        cleaned = re.sub(r"^json\n", "", cleaned, flags=re.IGNORECASE)
        data = json.loads(cleaned)
        out = []
        aff_by_name = { (a.get("name") or "").strip(): a.get("affiliations") or [] for a in data.get("authors", []) }
        for name in authors_from_api:
            aff = aff_by_name.get(name, [])
            aff = [s.strip() for s in aff if s and s.strip()]
            out.append({"name": name, "affiliations": aff})
        return out
    except Exception:
        return [{"name": a, "affiliations": []} for a in authors_from_api]


def parse_arxiv_feed(xml_text: str):
    """
    解析并得到 title, summary, url(abs), published_date(YYYY-MM-DD), pdf_url, authors(list[str])
    """
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)

    entries = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        id_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()

        published_raw = entry.findtext("atom:published", default="", namespaces=ns) or ""
        pub_date = ""
        if published_raw:
            try:
                pub_date = datetime.datetime.strptime(published_raw, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
            except Exception:
                pub_date = published_raw[:10]

        authors = []
        for a in entry.findall("atom:author", ns):
            nm = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            if nm:
                authors.append(nm)

        pdf_url = ""
        abs_url = ""
        for link in entry.findall("atom:link", ns):
            href = link.get("href", "")
            rel = link.get("rel", "")
            title_attr = link.get("title", "")
            if rel == "alternate" and "abs" in href:
                abs_url = href
            if title_attr.lower() == "pdf" or (rel == "related" and href.endswith(".pdf")):
                pdf_url = href
        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("abs", "pdf")

        entries.append(
            {
                "title": title,
                "summary": summary,
                "url": id_url,
                "pub_date": pub_date,
                "pdf_url": pdf_url,
                "authors": authors,
            }
        )
    return entries


def search_arxiv_papers(search_term, max_results=10):
    """
    根据关键词检索最新提交的论文
    """
    base_url = "http://export.arxiv.org/api/query?"
    search_query = (
        f"search_query=all:{search_term}&start=0&max_results={max_results}"
        f"&sortBy=submittedDate&sortOrder=descending"
    )
    resp = requests.get(base_url + search_query, timeout=60)
    if resp.status_code != 200:
        print("请求失败，请检查你的查询参数。")
        return []

    items = parse_arxiv_feed(resp.text)
    results = []
    for it in items:
        authors_aff = extract_affiliations_for_authors(it["pdf_url"], it["authors"])
        results.append(
            {
                "title": it["title"],
                "url": it["url"],
                "pub_date": it["pub_date"],
                "summary": it["summary"],
                "authors": authors_aff,
            }
        )
    return results


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="arXiv recent paper")
    parser.add_argument("-k", "--keyword", required=True, help="检索关键词")
    parser.add_argument("-s", "--save_dir", required=True, help="结果保存目录")
    parser.add_argument("-m", "--max", type=int, default=10, help="最多抓取结果数")
    args = parser.parse_args()

    papers = search_arxiv_papers(args.keyword, max_results=args.max)
    save_path = save_results(args.save_dir, args.keyword, papers)
    print(f"共保存 {len(papers)} 篇到：{save_path}")


if __name__ == "__main__":
    main()
