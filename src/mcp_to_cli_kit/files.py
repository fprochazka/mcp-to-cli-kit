"""Files the kit writes for its owner only: the config file, the tool cache and the stored tokens."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_private(path: Path, content: str) -> None:
    """Write a text file atomically, readable by its owner only.

    The content goes to a temporary file in the same directory, created with mode 0600, which is
    then renamed over the target. A reader never sees a half-written file, and a secret in it is
    never readable by others, not even for a moment. Missing directories are created with mode 0700.

    Raises:
        OSError: The directory or the file cannot be written.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
