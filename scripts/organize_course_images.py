# 学习导读｜整理学习文档中的图片引用与文件位置，先计算计划，再校验路径与更新引用。
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
EXCLUDED_PARTS = {
    "demo code",
    ".pytest_cache",
    "sample_docs",
    "experiment_docs",
    ".uv-python",
    "__pycache__",
    "_orphaned_images",
}
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]\n]*\]\(([^)\n]+)\)")
HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
REMOTE_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
WINDOWS_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")


@dataclass(frozen=True)
class ImageReference:
    article: Path
    source: Path
    path_start: int
    path_end: int
    original_target: str


@dataclass(frozen=True)
class Move:
    source: Path
    target: Path


@dataclass
class Plan:
    articles: list[Path]
    references: list[ImageReference]
    active_moves: list[Move]
    orphan_moves: list[Move]


# 判断图片整理流程是否应跳过隐藏/生成目录或不支持文件。
def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


# 读取文本并记录 UTF-8 BOM 状态，以便写回时保持文件编码形式。
def read_text_preserving_bom(path: Path) -> tuple[str, bool]:
    data = path.read_bytes()
    has_bom = data.startswith(b"\xef\xbb\xbf")
    return data.decode("utf-8-sig"), has_bom


# 按原 BOM 状态写文本文件，避免批量更新引用时改变编码。
def write_text_preserving_bom(path: Path, text: str, has_bom: bool) -> None:
    data = text.encode("utf-8")
    if has_bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)


# 清理 Markdown 图片链接目标中的锚点、空白和路径前缀。
def clean_target(raw_target: str) -> str | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    if REMOTE_SCHEME_RE.match(target) and not WINDOWS_PATH_RE.match(target):
        return None
    target = target.split("#", 1)[0].split("?", 1)[0]
    return unquote(target)


# 将文章内相对图片链接解析为本地路径。
def resolve_target(article: Path, raw_target: str) -> Path | None:
    target = clean_target(raw_target)
    if target is None:
        return None
    path = Path(target.replace("/", os.sep))
    if not path.is_absolute():
        path = article.parent / path
    return Path(os.path.abspath(path))


# 扫描 Markdown 文章并提取图片引用及其原始文本范围。
def find_references(article: Path) -> list[ImageReference]:
    text, _ = read_text_preserving_bom(article)
    references: list[ImageReference] = []
    for pattern in (MARKDOWN_IMAGE_RE, HTML_IMAGE_RE):
        for match in pattern.finditer(text):
            raw_target = match.group(1)
            source = resolve_target(article, raw_target)
            if source is None:
                continue
            references.append(
                ImageReference(
                    article=article,
                    source=source.resolve(),
                    path_start=match.start(1),
                    path_end=match.end(1),
                    original_target=raw_target,
                )
            )
    return references


# 根据课程和文章路径推导对应的图片目录。
def article_image_directory(course_root: Path, article: Path) -> Path:
    relative = article.relative_to(course_root)
    top_level = relative.parts[0]
    if top_level.startswith("chapter_"):
        return course_root / top_level / "images" / article.stem
    if article.parent == course_root:
        return course_root / "images" / article.stem
    raise ValueError(f"Cannot determine image directory for {article}")


# 确认目标解析后仍在指定根目录内，阻止路径穿越。
def ensure_within(root: Path, path: Path) -> None:
    if os.path.commonpath((root.resolve(), path.resolve())) != str(root.resolve()):
        raise ValueError(f"Path escapes course root: {path}")


# 扫描课程文章和图片，生成待移动文件及引用修改的只读计划。
def build_plan(course_root: Path) -> Plan:
    archive_root = course_root / "_orphaned_images"
    articles = sorted(
        path
        for path in course_root.rglob("*.md")
        if not is_excluded(path.relative_to(course_root))
    )
    references = [reference for article in articles for reference in find_references(article)]

    missing = [reference for reference in references if not reference.source.is_file()]
    if missing:
        details = "\n".join(
            f"{item.article}: {item.original_target}" for item in missing
        )
        raise ValueError(f"Missing image references:\n{details}")

    owners: dict[str, set[Path]] = defaultdict(set)
    for reference in references:
        owners[os.path.normcase(str(reference.source))].add(reference.article)
    shared = {path: value for path, value in owners.items() if len(value) > 1}
    if shared:
        raise ValueError(f"Images referenced by multiple articles: {shared}")

    active_moves: list[Move] = []
    destination_sources: dict[str, Path] = {}
    for reference in references:
        target = article_image_directory(course_root, reference.article) / reference.source.name
        ensure_within(course_root, target)
        key = os.path.normcase(str(target))
        existing_source = destination_sources.get(key)
        if existing_source is not None and existing_source != reference.source:
            raise ValueError(
                f"Active image collision at {target}: {existing_source} and {reference.source}"
            )
        destination_sources[key] = reference.source
        if os.path.normcase(str(reference.source)) != key:
            active_moves.append(Move(reference.source, target))

    active_sources = {os.path.normcase(str(reference.source)) for reference in references}
    images = sorted(
        path.resolve()
        for path in course_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not is_excluded(path.relative_to(course_root))
    )
    orphan_moves = []
    for image in images:
        if os.path.normcase(str(image)) in active_sources:
            continue
        target = archive_root / image.relative_to(course_root)
        ensure_within(archive_root, target)
        orphan_moves.append(Move(image, target))

    all_targets: dict[str, Path] = {}
    for move in [*active_moves, *orphan_moves]:
        key = os.path.normcase(str(move.target))
        prior = all_targets.get(key)
        if prior is not None and prior != move.source:
            raise ValueError(f"Move collision at {move.target}: {prior} and {move.source}")
        all_targets[key] = move.source
        if move.target.exists() and move.target.resolve() != move.source.resolve():
            raise ValueError(f"Move target already exists: {move.target}")

    return Plan(
        articles=articles,
        references=references,
        active_moves=active_moves,
        orphan_moves=orphan_moves,
    )


# 按原图片路径建立计划查找表，供引用批量替换。
def target_by_source(plan: Plan) -> dict[str, Path]:
    targets = {
        os.path.normcase(str(move.source)): move.target for move in plan.active_moves
    }
    for reference in plan.references:
        key = os.path.normcase(str(reference.source))
        targets.setdefault(key, reference.source)
    return targets


# 计算图片移动后相对于文章的新链接地址。
def replacement_target(article: Path, target: Path, original: str) -> str:
    relative = os.path.relpath(target, article.parent).replace(os.sep, "/")
    relative = relative.replace(" ", "%20")
    if original.strip().startswith("<") and original.strip().endswith(">"):
        return f"<{relative}>"
    return relative


# 只替换 Markdown 图片链接区间并保持其余正文及 BOM 不变。
def updated_article_bytes(plan: Plan) -> dict[Path, bytes]:
    targets = target_by_source(plan)
    by_article: dict[Path, list[ImageReference]] = defaultdict(list)
    for reference in plan.references:
        by_article[reference.article].append(reference)

    updates: dict[Path, bytes] = {}
    for article, references in by_article.items():
        text, has_bom = read_text_preserving_bom(article)
        revised = text
        for reference in sorted(references, key=lambda item: item.path_start, reverse=True):
            target = targets[os.path.normcase(str(reference.source))]
            replacement = replacement_target(
                article,
                target,
                reference.original_target,
            )
            revised = revised[: reference.path_start] + replacement + revised[reference.path_end :]
        if revised != text:
            data = revised.encode("utf-8")
            if has_bom:
                data = b"\xef\xbb\xbf" + data
            updates[article] = data
    return updates


# 分块读取文件并计算内容 SHA-256 指纹。
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 为归档移动写入来源/校验和清单。
def write_archive_metadata(course_root: Path, orphan_moves: list[Move]) -> None:
    archive_root = course_root / "_orphaned_images"
    archive_root.mkdir(parents=True, exist_ok=True)
    readme = archive_root / "README.md"
    readme.write_text(
        "# 课程孤立图片归档\n\n"
        "此目录保存整理时未被任何课程 Markdown 引用的图片。文件保持原相对路径，"
        "便于追溯旧章节和旧素材来源。\n\n"
        "- 归档日期：2026-07-30\n"
        f"- 归档图片：{len(orphan_moves)} 张\n"
        "- 使用规则：不要从课程正文直接引用这里的文件；确认仍有价值时，先移回对应"
        "章节的 `images/<文章文件名>/`，再添加正文引用。\n"
        "- 详细清单：`manifest.csv`\n",
        encoding="utf-8",
    )

    with (archive_root / "manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["original_path", "archived_path", "size_bytes", "sha256"]
        )
        for move in sorted(orphan_moves, key=lambda item: str(item.source)):
            writer.writerow(
                [
                    move.source.relative_to(course_root).as_posix(),
                    move.target.relative_to(course_root).as_posix(),
                    move.target.stat().st_size,
                    sha256(move.target),
                ]
            )


# 清理执行归档后遗留的空图片目录。
def remove_empty_image_directories(course_root: Path) -> int:
    removed = 0
    directories = sorted(
        (path for path in course_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        relative = directory.relative_to(course_root)
        if "_orphaned_images" in relative.parts or "images" not in relative.parts:
            continue
        try:
            directory.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


# 执行已审核计划中的文件移动和文章更新，并处理失败回滚。
def apply_plan(course_root: Path, plan: Plan) -> None:
    article_updates = updated_article_bytes(plan)
    original_articles = {path: path.read_bytes() for path in article_updates}
    completed_moves: list[Move] = []
    try:
        for move in [*plan.orphan_moves, *plan.active_moves]:
            move.target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(move.source), str(move.target))
            completed_moves.append(move)
        for article, data in article_updates.items():
            article.write_bytes(data)
        write_archive_metadata(course_root, plan.orphan_moves)
        removed = remove_empty_image_directories(course_root)
        print(f"Removed empty image directories: {removed}")
    except Exception:
        for article, data in original_articles.items():
            article.write_bytes(data)
        for move in reversed(completed_moves):
            if move.target.exists() and not move.source.exists():
                move.source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(move.target), str(move.source))
        raise


# 检查归档文件数量、文章引用及内容校验是否符合计划。
def verify(course_root: Path, expected_archive_count: int | None = None) -> None:
    plan = build_plan(course_root)
    missing = [reference for reference in plan.references if not reference.source.is_file()]
    if missing:
        raise ValueError(f"References missing after organization: {len(missing)}")
    if plan.active_moves:
        raise ValueError(
            f"Referenced images outside their article directories: {len(plan.active_moves)}"
        )
    if plan.orphan_moves:
        raise ValueError(
            f"Unreferenced images remain in active course directories: {len(plan.orphan_moves)}"
        )

    archive_root = course_root / "_orphaned_images"
    archived_images = [
        path
        for path in archive_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if expected_archive_count is not None and len(archived_images) != expected_archive_count:
        raise ValueError(
            f"Expected {expected_archive_count} archived images, found {len(archived_images)}"
        )
    print(f"Verified article references: {len(plan.references)}")
    print(f"Verified active orphan images: {len(plan.orphan_moves)}")
    print(f"Verified archived images: {len(archived_images)}")


# 以可读表格输出计划统计和待处理路径。
def print_plan(plan: Plan) -> None:
    print(f"Course articles: {len(plan.articles)}")
    print(f"Image references: {len(plan.references)}")
    print(f"Referenced images to move: {len(plan.active_moves)}")
    print(f"Orphan images to archive: {len(plan.orphan_moves)}")
    print("Target layout: chapter_xx/images/<article filename>/<image filename>")


# 解析课程根目录和执行模式，生成整理计划并按需落盘。
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize course images by article and archive unreferenced images."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the planned moves and Markdown reference updates.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an already organized course image tree.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    course_root = repo_root / "docs" / "course"
    if args.verify:
        verify(course_root)
        return

    plan = build_plan(course_root)
    print_plan(plan)
    if not args.apply:
        print("Dry run only. Use --apply to perform the organization.")
        return

    archived_count = len(plan.orphan_moves)
    apply_plan(course_root, plan)
    verify(course_root, expected_archive_count=archived_count)


if __name__ == "__main__":
    main()
