import lmdb
import orjson
from pathlib import Path
from typing import Any
from storage.lmdb_utils import get_lmdb_env


class ChatDB:
    """LMDB-backed key-value store for chat-related data.

    This class manages multiple named LMDB databases under a single environment
    and stores values as JSON bytes via `orjson`.
    """

    def __init__(
        self,
        path: str | Path,
        db_names: list[str],
        map_size: int = 1024 * 1024 * 1024
    ) -> None:
        """Initialize LMDB environment and named databases.

        Args:
            path: Directory path where LMDB files are stored.
            db_names: List of LMDB database names to create/open.
            map_size: Maximum LMDB map size in bytes.

        Raises:
            ValueError: If `db_names` is empty.
        """
        if not db_names:
            raise ValueError("db_names must not be empty")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = get_lmdb_env(
            path=self.path,
            max_dbs=len(db_names),
            map_size=map_size
        )
        self.dbs: dict[str, lmdb._Database] = {}
        for db_name in db_names:
            self.dbs[db_name] = self.env.open_db(
                key=db_name.encode("utf-8"),
                create=True
            )

    def _get_db(self, db_name: str) -> lmdb._Database:
        """Get LMDB handle for a named database.

        Args:
            db_name: Target database name.

        Returns:
            Open LMDB database handle.

        Raises:
            KeyError: If `db_name` was not configured at initialization.
        """
        if db_name not in self.dbs:
            raise KeyError(
                f"Unknown db_name '{db_name}'. "
                f"Available: {list(self.dbs.keys())}"
            )
        return self.dbs[db_name]

    def get(
        self,
        key: str,
        db_name: str
    ) -> Any:
        """Fetch and decode a value by key.

        Args:
            key: Lookup key.
            db_name: Target database name.

        Returns:
            Decoded Python object if key exists, otherwise `None`.

        Raises:
            KeyError: If `db_name` was not configured.
            orjson.JSONDecodeError: If stored bytes are not valid JSON.
        """
        db = self._get_db(db_name)
        with self.env.begin(db=db, write=False) as txn:
            value = txn.get(key.encode("utf-8"))
        if value is None:
            return None
        return orjson.loads(value)

    def put(
        self,
        key: str,
        value: Any,
        db_name: str,
        overwrite: bool = True
    ) -> bool:
        """Store a JSON-serializable value by key.

        Args:
            key: Key to write.
            value: JSON-serializable value.
            db_name: Target database name.
            overwrite: Whether to overwrite an existing key.

        Returns:
            `True` if the key/value pair was written, else `False` when
            `overwrite=False` and key already exists.

        Raises:
            KeyError: If `db_name` was not configured.
            TypeError: If `value` is not JSON-serializable by `orjson`.
        """
        db = self._get_db(db_name)
        payload = orjson.dumps(value)
        with self.env.begin(db=db, write=True) as txn:
            return txn.put(
                key.encode("utf-8"),
                payload,
                overwrite=overwrite
            )

    def keys(
        self,
        db_name: str
    ) -> list[str]:
        """List all keys in the specified database.

        Args:
            db_name: Target database name.

        Returns:
            List of keys decoded as UTF-8 strings.

        Raises:
            KeyError: If `db_name` was not configured.
        """
        db = self._get_db(db_name)
        with self.env.begin(db=db, write=False) as txn:
            with txn.cursor() as cursor:
                return [key.decode("utf-8") for key, _ in cursor]
