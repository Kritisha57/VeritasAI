"""Header-aware Markdown parser that preserves code blocks and tables."""

import re
from pathlib import Path
from ingestion.chunker import HierarchicalChunker
from ingestion.models import ChildChunk, ParentChunk

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^(`{3,}|~{3,})")


class MarkdownParser:
    """Parses Markdown files into hierarchical Parent and Child chunks by heading tree."""

    def __init__(self, chunker: HierarchicalChunker | None = None) -> None:
        self.chunker = chunker or HierarchicalChunker()

    def parse(self, file_path: Path, doc_id: str) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """Splits Markdown along headings while keeping fenced code blocks intact."""
        raw_text = file_path.read_text(encoding="utf-8", errors="replace")
        sections = self._extract_sections(raw_text)

        all_parents: list[ParentChunk] = []
        all_children: list[ChildChunk] = []

        for idx, (breadcrumb, content) in enumerate(sections):
            parents, children = self.chunker.chunk_section(
                doc_id=doc_id,
                filename=file_path.name,
                section_path=breadcrumb,
                page_number=1,
                text=content,
                modality="text",
                start_index=idx * 100,
            )
            all_parents.extend(parents)
            all_children.extend(children)

        return all_parents, all_children

    @staticmethod
    def _extract_sections(markdown_text: str) -> list[tuple[str, str]]:
        """State machine that tracks H1-H6 breadcrumbs outside fenced code blocks."""
        headers: dict[int, str] = {}
        sections: list[tuple[str, str]] = []
        current_lines: list[str] = []
        current_path = "Introduction"
        in_fence = False

        for line in markdown_text.splitlines():
            if FENCE_RE.match(line.strip()):
                in_fence = not in_fence
                current_lines.append(line)
                continue

            if not in_fence:
                match = HEADER_RE.match(line)
                if match:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append((current_path, body))
                    current_lines = []

                    level = len(match.group(1))
                    title = match.group(2).strip()
                    headers[level] = title
                    for deeper in [k for k in headers if k > level]:
                        del headers[deeper]
                    current_path = " > ".join(headers[k] for k in sorted(headers))
                    continue

            current_lines.append(line)

        trailing = "\n".join(current_lines).strip()
        if trailing:
            sections.append((current_path, trailing))
        return sections
