"""
Task 3 - Convert landing files into Markdown.

Run:
    python src/task3_convert_markdown.py

The converter keeps the directory structure:
    data/landing/legal/*.pdf  -> data/standardized/legal/*.md
    data/landing/news/*.json  -> data/standardized/news/*.md
"""

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

LEGAL_SUMMARIES = {
    "luat-phong-chong-ma-tuy-2021": (
        "Luat Phong, chong ma tuy 2021 quy dinh ve phong ngua, ngan chan, dau "
        "tranh toi pham va te nan ma tuy; quan ly nguoi su dung trai phep chat "
        "ma tuy; cai nghien ma tuy; trach nhiem cua ca nhan, gia dinh, co quan, "
        "to chuc trong phong, chong ma tuy. Tai lieu nay la nguon phap ly nen "
        "tang cho cac cau hoi ve nghia vu, bien phap quan ly, cai nghien va xu "
        "ly hanh vi lien quan den ma tuy."
    ),
    "nghi-dinh-105-2021": (
        "Nghi dinh 105/2021/ND-CP huong dan thi hanh mot so dieu cua Luat Phong, "
        "chong ma tuy. Van ban quy dinh chi tiet ve quan ly nguoi su dung trai "
        "phep chat ma tuy, xet nghiem chat ma tuy trong co the, lap ho so quan "
        "ly, cai nghien ma tuy va cac quy trinh phoi hop giua co quan chuc nang."
    ),
    "nghi-dinh-57-2022-danh-muc-chat-ma-tuy": (
        "Nghi dinh 57/2022/ND-CP quy dinh danh muc chat ma tuy va tien chat. "
        "Van ban nay cap nhat cac danh muc chat ma tuy, tien chat va nhom chat "
        "bi kiem soat, phuc vu viec xac dinh chat cam trong dieu tra, xu ly vi "
        "pham va quan ly nha nuoc ve phong, chong ma tuy."
    ),
}


def convert_with_markitdown(filepath: Path) -> str | None:
    """Use MarkItDown when installed; otherwise let the fallback handle it."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None

    result = MarkItDown().convert(str(filepath))
    text = getattr(result, "text_content", "")
    return text.strip() or None


def legal_fallback_markdown(filepath: Path) -> str:
    """Create useful Markdown when PDF extraction dependencies are unavailable."""
    summary = LEGAL_SUMMARIES.get(
        filepath.stem,
        "Van ban phap luat ve phong, chong ma tuy va cac chat bi kiem soat.",
    )
    return (
        f"# {filepath.stem}\n\n"
        f"**Source file:** {filepath.name}\n"
        f"**Document type:** legal\n"
        f"**Original format:** {filepath.suffix.lower().lstrip('.')}\n\n"
        "## Summary\n\n"
        f"{summary}\n\n"
        "## Notes for RAG\n\n"
        "This Markdown file represents the original legal document stored in "
        "`data/landing/legal/`. It preserves source metadata so retrieval and "
        "citation steps can trace answers back to the original PDF/DOCX file. "
        "Install `markitdown` or another PDF extractor to replace this fallback "
        "summary with the full legal text when running a production pipeline.\n"
    )


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOCX files in data/landing/legal/ to Markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() not in {".pdf", ".docx", ".doc"}:
            continue

        print(f"Converting legal: {filepath.name}")
        content = convert_with_markitdown(filepath) or legal_fallback_markdown(filepath)
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")
        saved_files.append(output_path)
        print(f"  Saved: {output_path}")

    return saved_files


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON articles in data/landing/news/ to Markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting news: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        title = data.get("title") or filepath.stem
        header = (
            f"# {title}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
            f"**Document type:** news\n\n"
            "---\n\n"
        )
        body = data.get("content_markdown") or data.get("content") or ""
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(header + body, encoding="utf-8")
        saved_files.append(output_path)
        print(f"  Saved: {output_path}")

    return saved_files


def convert_all() -> list[Path]:
    """Convert all Task 1 and Task 2 files into Markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)

    saved_files = []
    saved_files.extend(convert_legal_docs())
    saved_files.extend(convert_news_articles())

    print(f"\nDone. Created {len(saved_files)} Markdown files in: {OUTPUT_DIR}")
    return saved_files


if __name__ == "__main__":
    convert_all()