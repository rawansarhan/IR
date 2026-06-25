"""SQLite document store — original corpus persisted on disk."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "db"
DEFAULT_DB = DATA_DIR / "documents.db"


class DocumentStore:
    """Store and retrieve full document text by dataset + doc_id."""

    def __init__(self, db_path: str | Path = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def _table(self, dataset: str) -> str:
        safe = dataset.replace("-", "_")
        return f"docs_{safe}"

    def _ensure_table(self, dataset: str) -> None:
        table = self._table(dataset)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                doc_id TEXT PRIMARY KEY,
                text   TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def clear_dataset(self, dataset: str) -> None:
        table = self._table(dataset)
        self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        self._conn.commit()

    def store_documents(
        self,
        dataset: str,
        doc_ids: List[str],
        texts: List[str],
        batch_size: int = 1000,
    ) -> int:
        """Replace all documents for a dataset. Returns count stored."""
        self.clear_dataset(dataset)
        self._ensure_table(dataset)
        table = self._table(dataset)

        rows = list(zip(doc_ids, texts))
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._conn.executemany(
                f"INSERT INTO {table} (doc_id, text) VALUES (?, ?)",
                batch,
            )
        self._conn.commit()
        return len(rows)

    def get_document(self, dataset: str, doc_id: str) -> Optional[str]:
        self._ensure_table(dataset)
        table = self._table(dataset)
        cur = self._conn.execute(
            f"SELECT text FROM {table} WHERE doc_id = ?",
            (doc_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def count(self, dataset: str) -> int:
        self._ensure_table(dataset)
        table = self._table(dataset)
        cur = self._conn.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])

    def list_page(
        self,
        dataset: str,
        page: int = 1,
        page_size: int = 20,
    ) -> Tuple[int, List[dict]]:
        self._ensure_table(dataset)
        table = self._table(dataset)
        total = self.count(dataset)
        offset = (page - 1) * page_size
        cur = self._conn.execute(
            f"SELECT doc_id, text FROM {table} ORDER BY rowid LIMIT ? OFFSET ?",
            (page_size, offset),
        )
        docs = [
            {"doc_id": doc_id, "snippet": text[:150]}
            for doc_id, text in cur.fetchall()
        ]
        return total, docs

    def doc_ids(self, dataset: str) -> List[str]:
        self._ensure_table(dataset)
        table = self._table(dataset)
        cur = self._conn.execute(f"SELECT doc_id FROM {table} ORDER BY rowid")
        return [row[0] for row in cur.fetchall()]

    def is_loaded(self, dataset: str) -> bool:
        return self.count(dataset) > 0

    def close(self) -> None:
        self._conn.close()
