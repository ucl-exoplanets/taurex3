"""Handles citation information for Taurex."""
import typing as t
from functools import lru_cache
from urllib.error import HTTPError, URLError

from taurex.log.logger import setup_log

_log = setup_log(__name__)

try:
    from pybtex.database import Entry

    has_pybtex = True
except ImportError:
    _log.warning("Pybtex not installed. Citation functionality disabled")
    has_pybtex = False
    Entry = None


def cleanup_string(string: str) -> str:
    """Cleans up a string for bibtex."""
    return string.replace("{", "").replace("}", "").replace("\\", "")


def recurse_bibtex(
    obj: "Citable", entries: t.Optional[t.List[str]] = None
) -> t.List[str]:
    """Recursively search for bibtex entries in base classes."""
    entries = entries or []

    for b in obj.__class__.__bases__:
        if issubclass(b, Citable):
            entries.extend(b.BIBTEX_ENTRIES)
            recurse_bibtex(b, entries)
    return entries


def stringify_people(authors: t.List[str]) -> str:
    """Converts authors to a string."""
    return ", ".join([cleanup_string(str(p)) for p in authors])


def unique_citations_only(citations: t.List[str]) -> t.List[str]:
    """Removes duplicate citations."""
    current_citations = []
    for c in citations:
        if c not in current_citations:
            current_citations.append(c)
    return current_citations


def to_bibtex(citations: t.List[str]) -> str:
    """Converts citations to bibtex."""
    import uuid

    try:
        from pybtex.database import BibliographyData
    except ImportError:
        return str(citations)

    entries = {str(uuid.uuid4())[:8]: b for b in citations}
    bib_data = BibliographyData(entries=entries)

    return bib_data.to_string("bibtex")


def handle_publication(fields: t.Dict[str, str]) -> str:
    """Handles publication information."""
    journal = []
    if "journal" in fields:
        journal.append(cleanup_string(fields["journal"]))
    elif "booktitle" in fields:
        journal.append(cleanup_string(fields["booktitle"]))
    elif "archivePrefix" in fields:
        journal.append(cleanup_string(fields["archivePrefix"]))

    if "volume" in fields:
        journal.append(cleanup_string(fields["volume"]))
    elif "eprint" in fields:
        journal.append(cleanup_string(fields["eprint"]))
    if "pages" in fields:
        journal.append(cleanup_string(fields["pages"]))

    if "month" in fields:
        journal.append(cleanup_string(fields["month"]))

    if "year" in fields:
        journal.append(cleanup_string(fields["year"]))

    return ", ".join(journal)


def construct_nice_printable_string(entry: str, indent: int = 0) -> str:
    """Constructs a nice printable string for each entry."""
    mystring = ""
    indent = "".join(["\t"] * indent)
    form = f"{indent}%s\n"

    if isinstance(entry, str) or not has_pybtex:
        return f"Found non bibtex citation or pybtex not installed: {entry}\n"

    if "title" in entry.fields:
        mystring += form % cleanup_string(entry.fields["title"])

    people = entry.persons
    if "author" in people:
        mystring += form % stringify_people(people["author"])

    mystring += form % handle_publication(entry.fields)

    return mystring


class Citable:
    """Defines a class that contains citation information."""

    BIBTEX_ENTRIES = []
    """List of bibtex entries."""

    def citations(self) -> t.List[str]:
        """Returns a list of citations."""
        entries = self.BIBTEX_ENTRIES[:]
        entries = recurse_bibtex(self, entries)

        if not has_pybtex:
            return unique_citations_only(entries)
        all_citations = [Entry.from_string(b, "bibtex") for b in entries]

        return unique_citations_only(all_citations)

    def nice_citation(
        self,
        prefix: t.Optional[str] = "",
        start_idx: t.Optional[int] = 0,
        indent: t.Optional[int] = 0,
    ) -> str:
        """Returns a nice printable string of citations.

        Parameters
        ----------
        prefix: str, optional
            Prefix to add to each citation
        start_idx: int, optional
            Starting index of citation
        indent: int, optional
            Indentation of citation


        """
        entries = self.citations()

        if len(entries) == 0:
            return ""

        return "\n".join([construct_nice_printable_string(e) for e in entries])


@lru_cache(maxsize=100)
def doi_to_bibtex(doi: str) -> t.Optional[str]:
    """Converts a doi to bibtex.

    Parameters
    ----------
    doi: str
        DOI to convert

    """
    import urllib

    base_url = "http://dx.doi.org/"
    url = base_url + doi

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/x-bibtex")
    try:
        with urllib.request.urlopen(req) as f:  # noqa: S310
            return f.read().decode()
    except HTTPError as e:
        if e.code == 404:
            print("DOI not found.")
        else:
            print("Service unavailable.")

        return None
    except URLError:
        return None
