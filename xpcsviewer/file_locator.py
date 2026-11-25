"""
File discovery and management for XPCS datasets.

Provides the FileLocator class for navigating file systems, discovering
XPCS data files, and managing file collections for analysis.

Classes:
    FileLocator: Main class for file discovery and management.

Functions:
    create_xpcs_dataset: Factory function to create XpcsFile objects.
"""

# Standard library imports
import datetime
import os
import threading
import traceback
from collections.abc import Callable

# Local imports
from .fileIO.hdf_reader import _file_identity
from .fileIO.qmap_utils import QMapManager
from .helper.listmodel import ListDataModel
from .utils.log_utils import log_timing
from .utils.logging_config import get_logger
from .xpcs_file import XpcsFile as XF

logger = get_logger(__name__)

_UNTRACKED = object()  # cache entry without a recorded file revision


def _get_file_size_mb(fname: str) -> float:
    """Get file size in MB, returns 0.0 if file doesn't exist."""
    try:
        return os.path.getsize(fname) / (1024 * 1024)
    except OSError:
        return 0.0


@log_timing(threshold_ms=500)
def create_xpcs_dataset(fname, **kwargs):
    """Create an XpcsFile object from a file path.

    Args:
        fname: Path to HDF5 XPCS data file.
        **kwargs: Additional arguments passed to XpcsFile.

    Returns:
        XpcsFile instance or None if loading fails.
    """
    file_size_mb = _get_file_size_mb(fname)
    logger.debug(
        f"Loading XPCS dataset: {os.path.basename(fname)} ({file_size_mb:.1f} MB)"
    )
    try:
        temp = XF(fname, **kwargs)
        return temp
    except KeyError as e:
        logger.warning(
            f"Failed to load file {fname}: Missing required HDF5 dataset - {e}"
        )
    except OSError as e:
        logger.warning(f"Failed to load file {fname}: File I/O error - {e}")
    except Exception as e:
        logger.warning(f"Failed to load file {fname}: {type(e).__name__} - {e}")
        logger.debug("Traceback: %s", traceback.format_exc())

    return None


class FileLocator:
    def __init__(self, path):
        self.path = path
        self.source = ListDataModel()
        self.source_search = ListDataModel()
        self.target = ListDataModel()
        self.qmap_manager = QMapManager()
        self.cache = {}
        # Source revision (``_file_identity``) each cache entry was loaded from;
        # an entry, including a failed-load sentinel, is only served while the
        # file at its path is still that revision.
        self._cache_identity: dict[str, tuple[int, ...] | None] = {}
        self._cache_lock = threading.Lock()
        self.timestamp = None
        # Full paths whose cached entry the last build() dropped because the
        # file on disk was replaced or modified since it was loaded.
        self.replaced_paths: set[str] = set()

    def set_path(self, path):
        with self._cache_lock:
            self.path = path

    def clear(self):
        self.source.clear()
        self.source_search.clear()

    def _is_current(self, full_fname):
        """Whether the cached entry still matches the file on disk.

        Must be called with ``_cache_lock`` held.
        """
        recorded = self._cache_identity.get(full_fname, _UNTRACKED)
        return recorded is _UNTRACKED or recorded == _file_identity(full_fname)

    def _drop_stale_cache(self):
        """Forget cache entries whose source file was replaced or modified.

        Returns the set of dropped full paths.
        """
        with self._cache_lock:
            stale = {p for p in self.cache if not self._is_current(p)}
            for full_fname in stale:
                del self.cache[full_fname]
                self._cache_identity.pop(full_fname, None)
        return stale

    def file_revision(self, fname):
        """On-disk revision (``_file_identity``) of *fname* (a name in the
        target list or an absolute path); this is what the next read serves."""
        return _file_identity(os.path.normpath(os.path.join(self.path, fname)))

    def _store(self, full_fname, xf_obj, identity):
        """Insert ``xf_obj`` loaded from revision ``identity`` (lock held)."""
        self.cache[full_fname] = xf_obj
        self._cache_identity[full_fname] = identity

    def get_xf_list(
        self,
        rows=None,
        filter_atype=None,
        filter_fitted=False,
        *,
        check_cancelled: Callable[[], None] | None = None,
    ):
        """
        get the cached xpcs_file list;
        :param rows: a list of index to select; if None is given, then use
        :param check_cancelled: raises InterruptedError if the lookup was cancelled;
        :return: list of xpcs_file objects;
        """
        # Keep the target rows tied to the directory they were selected from.
        with self._cache_lock:
            if check_cancelled is not None:
                check_cancelled()
            path = self.path
            target_snapshot = self.target.copy()

        if rows is None:
            selected = list(range(len(target_snapshot)))
        elif isinstance(rows, int):
            selected = [rows]
        else:
            selected = rows

        # If rows was explicitly passed as an empty list, return empty result
        if rows is not None and len(selected) == 0:
            return []

        # If no target files are selected but we have cached files, use the cache as fallback
        # This handles cases where files are loaded but target list is temporarily empty
        if not selected and self.cache:
            logger.info(
                "Using cached files as fallback for plotting (target list empty)"
            )
            with self._cache_lock:
                if check_cancelled is not None:
                    check_cancelled()
                ret = [
                    xf
                    for p, xf in self.cache.items()
                    if xf is not None and self._is_current(p)
                ]
            # Apply filters if specified
            if filter_fitted:
                ret = [xf for xf in ret if xf.fit_summary is not None]
            if filter_atype is not None:
                ret = [xf for xf in ret if filter_atype in getattr(xf, "atype", [])]
            return ret

        ret = []
        for n in selected:
            if n < 0 or n >= len(target_snapshot):
                continue
            full_fname = os.path.normpath(os.path.join(path, target_snapshot[n]))

            # Check the cache without holding the lock during I/O.
            # XpcsFile construction can be slow (HDF5 read) — holding the
            # cache lock throughout would block all other threads that need
            # to read or insert different files. (BUG-033)
            # Test membership, not None: a failed load is cached as None on
            # purpose so we don't re-read a broken file on every plot update.
            # An entry loaded from another revision of the file is stale.
            identity = _file_identity(full_fname)
            with self._cache_lock:
                is_cached = full_fname in self.cache and self._is_current(full_fname)
                xf_obj = self.cache.get(full_fname) if is_cached else None

            if not is_cached:
                # Construct the XpcsFile object outside the lock so HDF5 I/O
                # does not prevent concurrent cache lookups.
                if check_cancelled is not None:
                    check_cancelled()
                xf_obj = create_xpcs_dataset(full_fname, qmap_manager=self.qmap_manager)
                # Re-acquire the lock only to insert the new entry.
                # If another thread raced ahead and already inserted the same
                # file, prefer the existing entry so we never duplicate objects.
                with self._cache_lock:
                    if check_cancelled is not None:
                        check_cancelled()
                    if full_fname in self.cache and self._is_current(full_fname):
                        xf_obj = self.cache[full_fname]
                    else:
                        self._store(full_fname, xf_obj, identity)

            # Skip None objects (failed to load)
            if xf_obj is None:
                logger.warning(
                    f"Skipping invalid file {target_snapshot[n]} (failed to load)"
                )
                continue

            if xf_obj.fit_summary is None and filter_fitted:
                continue
            if filter_atype is None or filter_atype in xf_obj.atype:
                ret.append(xf_obj)

        return ret

    def get_hdf_info(self, fname, filter_str=None):
        """
        get the hdf information / hdf structure for fname
        :param fname: input filename
        :param fstr: list of filter string;
        :return: list of strings that contains the hdf information;
        """
        xf_obj = create_xpcs_dataset(
            os.path.normpath(os.path.join(self.path, fname)),
            qmap_manager=self.qmap_manager,
        )
        return xf_obj.get_hdf_info(filter_str)

    @log_timing(threshold_ms=1000)
    def add_target(self, alist, threshold=256, preload=True):
        if not alist:
            return
        loaded_count = 0
        total_size_mb = 0.0
        if preload and len(alist) <= threshold:
            for fn in alist:
                if fn in self.target:
                    continue
                full_fname = os.path.normpath(os.path.join(self.path, fn))
                file_size = _get_file_size_mb(full_fname)
                identity = _file_identity(full_fname)  # before the read
                xf_obj = create_xpcs_dataset(full_fname, qmap_manager=self.qmap_manager)
                if xf_obj is not None:
                    self.target.append(fn)
                    with self._cache_lock:
                        self._store(full_fname, xf_obj, identity)
                    loaded_count += 1
                    total_size_mb += file_size
            logger.info(
                f"Loaded {loaded_count}/{len(alist)} files "
                f"(total: {total_size_mb:.1f} MB)"
            )
        else:
            logger.info(
                f"Preload disabled or too many files ({len(alist)} > {threshold})"
            )
            self.target.extend(alist)
        self.timestamp = str(datetime.datetime.now())
        return

    def clear_target(self):
        self.target.clear()
        with self._cache_lock:
            self.cache.clear()
            self._cache_identity.clear()

    def remove_target(self, rlist):
        for x in rlist:
            if x in self.target:
                self.target.remove(x)
            with self._cache_lock:
                full_fname = os.path.normpath(os.path.join(self.path, x))
                self.cache.pop(full_fname, None)
                self._cache_identity.pop(full_fname, None)
        if not self.target:
            self.clear_target()
        self.timestamp = str(datetime.datetime.now())

    def reorder_target(self, row, direction="up"):
        size = len(self.target)
        assert 0 <= row < size, "check row value"
        if (direction == "up" and row == 0) or (
            direction == "down" and row == size - 1
        ):
            return -1

        item = self.target.pop(row)
        pos = row - 1 if direction == "up" else row + 1
        self.target.insert(pos, item)
        self.timestamp = str(datetime.datetime.now())
        # Return the new row index (int); caller converts to QModelIndex.
        # Boundary no-ops above return -1.
        return pos

    def search(self, val, filter_type="prefix"):
        assert filter_type in [
            "prefix",
            "substr",
        ], "filter_type must be prefix or substr"
        if filter_type == "prefix":
            selected = [x for x in self.source if x.startswith(val)]
        elif filter_type == "substr":
            filter_words = val.split()  # Split search query by whitespace
            selected = [x for x in self.source if all(t in x for t in filter_words)]
        self.source_search.replace(selected)

    def build(self, path=None, filter_list=(".hdf", ".h5"), sort_method="Filename"):
        if path is None:
            path = self.path

        if not path or not os.path.isdir(path):
            logger.warning("FileLocator.build called with invalid path: %r", path)
            return False

        self.set_path(path)
        # Reload: entries loaded from a since-replaced file must not outlive it.
        self.replaced_paths = self._drop_stale_cache()
        flist = [
            entry.name
            for entry in os.scandir(path)
            if entry.is_file()
            and entry.name.lower().endswith(filter_list)
            and not entry.name.startswith(".")
        ]
        if sort_method.startswith("Filename"):
            flist.sort()
        elif sort_method.startswith("Time"):
            # Schwartzian transform: pre-compute mtimes to avoid O(N log N) stat calls
            decorated = [(os.path.getmtime(os.path.join(path, f)), f) for f in flist]
            decorated.sort()
            flist = [f for _, f in decorated]
        elif sort_method.startswith("Index"):
            pass

        if sort_method.endswith("-reverse"):
            flist.reverse()
        self.source.replace(flist)
        return True


if __name__ == "__main__":
    # test1()
    fl = FileLocator(path="./data/files.txt")
