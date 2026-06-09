"""
Task 1 - Collect legal documents about drugs and controlled substances.

Run:
    python src/task1_collect_legal_docs.py

The script downloads at least 3 original PDF legal documents into:
    data/landing/legal/
"""

from pathlib import Path
import ssl
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

LEGAL_DOCUMENTS = [
    {
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "title": "Luat Phong, chong ma tuy 2021 (73/2021/QH14)",
        "url": "https://congan.sonla.gov.vn/wp-content/uploads/2022/05/1.-Luat-PCMT-2021.pdf",
    },
    {
        "filename": "nghi-dinh-105-2021.pdf",
        "title": "Nghi dinh 105/2021/ND-CP huong dan Luat Phong, chong ma tuy",
        "url": (
            "https://g7.cdnchinhphu.vn/api/download/stream?"
            "Url=wrAnHONIk_iV30imFi4FfAVl7oUvg5QpvcKV9XhqlQlmtiKkfWjxBaUmKVnIrT9Uho0RIbKvsTTrdNiTnWntacH-v_vk-8qD5rSld3MxPeOdZ-y3P9Y5T28KP95eu7OHZGjcuH3IoGpujRaha9uMHg~~"
            "&file_name=2021_1047+%2B+1048_105-2021-N%C4%90-CP.pdf"
        ),
    },
    {
        "filename": "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.pdf",
        "title": "Nghi dinh 57/2022/ND-CP quy dinh danh muc chat ma tuy va tien chat",
        "url": (
            "https://g7.cdnchinhphu.vn/api/download/stream?"
            "Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0Gvvr483flbRJTjcXEdO2_Pu0JyTUYpTyPvovUt0rel_BnYVLCGNsBDgvqb3aPsQBXd_uoyKha7iidNiWTFTqwuUPbqhAyHavQ~~"
            "&file_name=2022_709+%2B+710_57-2022-N%C4%90-CP.pdf"
        ),
    },
]


def setup_directory() -> None:
    """Create data/landing/legal/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory is ready: {DATA_DIR}")


def download_file(url: str, filename: str) -> Path:
    """Download a legal document and return its local path."""
    setup_directory()
    filepath = DATA_DIR / filename

    if filepath.exists() and filepath.stat().st_size > 1024:
        print(f"Already exists: {filepath.name}")
        return filepath

    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=60) as response:
            filepath.write_bytes(response.read())
    except (HTTPError, URLError) as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise RuntimeError(f"Could not download {url}") from exc

        context = ssl._create_unverified_context()
        with urlopen(request, timeout=60, context=context) as response:
            filepath.write_bytes(response.read())
    if filepath.stat().st_size <= 1024:
        raise ValueError(f"Downloaded file is too small: {filepath}")

    print(f"Downloaded: {filepath.name} ({filepath.stat().st_size:,} bytes)")
    return filepath


def collect_legal_documents() -> list[Path]:
    """Download all configured Task 1 legal documents."""
    downloaded_files = []
    for document in LEGAL_DOCUMENTS:
        print(f"Collecting: {document['title']}")
        downloaded_files.append(download_file(document["url"], document["filename"]))
    return downloaded_files


if __name__ == "__main__":
    collect_legal_documents()
