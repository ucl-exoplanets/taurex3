"""Remote I/O helpers — gzipped JSON cache, URL detection, HTTP downloads."""

import gzip
import io
import json
import pathlib
import urllib.parse
from typing import Any
from typing import Optional
from typing import Union

import requests
from bs4 import BeautifulSoup


Pathish = Union[str, pathlib.Path]


# ---------------------------------------------------------------------------
# Gzipped JSON
# ---------------------------------------------------------------------------


def json_zip(data: Any, path_or_file: Union[Pathish, io.IOBase], **kwargs) -> None:
    """Write *data* as gzipped JSON."""
    if isinstance(path_or_file, (str, pathlib.Path)):
        with gzip.open(str(path_or_file), "wt", encoding="utf-8") as fh:
            json.dump(data, fh, **kwargs)
    else:
        with gzip.GzipFile(fileobj=path_or_file, mode="w") as gz:
            json.dump(data, io.TextIOWrapper(gz), **kwargs)


def json_unzip(path_or_file: Union[Pathish, io.IOBase]) -> Any:
    """Read gzipped JSON."""
    if isinstance(path_or_file, (str, pathlib.Path)):
        with gzip.open(str(path_or_file), "rt", encoding="utf-8") as fh:
            return json.load(fh)
    else:
        with gzip.GzipFile(fileobj=path_or_file, mode="r") as gz:
            return json.load(io.TextIOWrapper(gz))


# ---------------------------------------------------------------------------
# Remote URL detection
# ---------------------------------------------------------------------------

_REMOTE_SCHEMES = frozenset({"http", "https", "ftp", "sftp"})


def is_remote_url(url: str) -> bool:
    """Check whether *url* uses a remote scheme (http, https, ftp, sftp)."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    return parsed.scheme in _REMOTE_SCHEMES


# ---------------------------------------------------------------------------
# Directory listing parser (Apache-style)
# ---------------------------------------------------------------------------


def _apache_size(text: str) -> Optional[int]:
    """Parse an Apache-style size string (``125K``, ``3.4M``)."""
    text = text.strip()
    if not text or text == "-":
        return None
    try:
        units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
        suffix = text[-1].upper()
        if suffix in units:
            return int(float(text[:-1]) * units[suffix])
        return int(text)
    except (ValueError, IndexError):
        return None


def parse_directory_listing(url: str) -> list[dict]:
    """Parse an Apache auto-index page and return file metadata."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    entries: list[dict] = []
    for row in soup.select("table tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        link = cells[1].find("a")
        if link is None:
            continue
        href = urllib.parse.urljoin(url, link.get("href", ""))
        size_raw = cells[1].get_text() if len(cells) == 3 else cells[2].get_text()
        if cells[0].get_text().strip() == "Parent Directory":
            continue
        last_modified = cells[0].get_text(strip=True) if len(cells) >= 3 else ""
        entries.append(
            {
                "href": href,
                "size": _apache_size(size_raw),
                "last_modified": last_modified,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Single-file download
# ---------------------------------------------------------------------------


def download_to_directory(url: str, output_dir: pathlib.Path) -> pathlib.Path:
    """Download a single file from *url* into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = pathlib.Path(urllib.parse.urlparse(url).path).name
    dest = output_dir / filename
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                fh.write(chunk)
    return dest
