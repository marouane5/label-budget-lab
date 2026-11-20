"""Retrieve and validate the original UCI writer-disjoint digit split."""

from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


URL = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
# Hash the two actual inputs. UCI can repackage the ZIP without changing the data.
MEMBERS = {
    "optdigits.tra": (3823, "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd"),
    "optdigits.tes": (1797, "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8"),
}
DEFAULT_ARCHIVE = Path("data/optdigits.zip")


def read_digits(archive_path):
    """Return (pixels in [0, 1], digit labels) for train, then test."""
    result = []
    with ZipFile(archive_path) as archive:
        for name, (expected_rows, fingerprint) in MEMBERS.items():
            content = archive.read(name)
            if sha256(content).hexdigest() != fingerprint:
                raise ValueError(f"Unexpected contents of {name}. Check the UCI source.")
            rows = np.loadtxt(content.splitlines(), delimiter=",", dtype=np.int16)
            if rows.shape != (expected_rows, 65):
                raise ValueError(f"Unexpected dimensions for {name}: {rows.shape}")
            result.append((rows[:, :-1] / 16.0, rows[:, -1].astype(int)))
    return tuple(result)


def fetch_dataset(destination=DEFAULT_ARCHIVE):
    destination = Path(destination)
    if destination.exists():
        read_digits(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as pending:
        temporary = Path(pending.name)
    try:
        with urlopen(URL, timeout=60) as source:
            temporary.write_bytes(source.read())
        read_digits(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
