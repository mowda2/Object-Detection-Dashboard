import os, sqlite3, json, threading, logging
from typing import Dict, Iterable, Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LibraryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()  # Thread-safety lock
        self._init()

    def _init(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS files(
                hash TEXT PRIMARY KEY,
                filename TEXT,
                size_bytes INTEGER,
                stored_path TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS analyses(
                id TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                params_hash TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                model TEXT,
                include TEXT,
                conf REAL,
                mpp REAL,
                device TEXT,
                fps REAL,
                frames INTEGER,
                video_path TEXT,
                csv_path TEXT,
                json_path TEXT,
                poster_path TEXT,
                UNIQUE(file_hash, params_hash),
                FOREIGN KEY(file_hash) REFERENCES files(hash) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS class_stats(
                analysis_id TEXT,
                class_name TEXT,
                unique_count INTEGER,
                PRIMARY KEY(analysis_id, class_name),
                FOREIGN KEY(analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            );
            """)
            self._conn.commit()

    # ------------ files ------------
    def upsert_file(self, file_hash: str, filename: str, size_bytes: int, stored_path: str):
        with self._lock:
            logger.info(f"[LibraryDB] upsert_file called: hash={file_hash[:16]}..., filename={filename}")
            cur = self._conn.cursor()
            cur.execute("""
                INSERT INTO files(hash, filename, size_bytes, stored_path)
                VALUES(?,?,?,?)
                ON CONFLICT(hash) DO UPDATE SET
                    filename=excluded.filename,
                    size_bytes=excluded.size_bytes,
                    stored_path=excluded.stored_path
            """, (file_hash, filename, size_bytes, stored_path))
            self._conn.commit()
            logger.info(f"[LibraryDB] upsert_file committed")

    def get_file(self, file_hash: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM files WHERE hash=?", (file_hash,))
            return cur.fetchone()

    # ------------ analyses ------------
    def find_analysis(self, file_hash: str, params_hash: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM analyses WHERE file_hash=? AND params_hash=?",
                        (file_hash, params_hash))
            return cur.fetchone()

    def get_analysis(self, analysis_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,))
            return cur.fetchone()

    def insert_analysis(self, rec: Dict, class_counts: Dict[str, int]):
        with self._lock:
            logger.info(f"[LibraryDB] insert_analysis called with id={rec.get('id')}, file_hash={rec.get('file_hash')[:16]}...")
            
            try:
                cur = self._conn.cursor()
                cur.execute("""
                    INSERT OR IGNORE INTO analyses(
                        id,file_hash,params_hash,model,include,conf,mpp,device,
                        fps,frames,video_path,csv_path,json_path,poster_path
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    rec["id"], rec["file_hash"], rec["params_hash"],
                    rec.get("model"), rec.get("include"), rec.get("conf"),
                    rec.get("mpp"), rec.get("device"),
                    rec.get("fps"), rec.get("frames"),
                    rec.get("video_path"), rec.get("csv_path"),
                    rec.get("json_path"), rec.get("poster_path"),
                ))
                rows_affected = cur.rowcount
                logger.info(f"[LibraryDB] INSERT analyses affected {rows_affected} rows")
                
                # upsert class stats
                for k, v in (class_counts or {}).items():
                    cur.execute("""
                        INSERT INTO class_stats(analysis_id,class_name,unique_count)
                        VALUES(?,?,?)
                        ON CONFLICT(analysis_id,class_name) DO UPDATE SET unique_count=excluded.unique_count
                    """, (rec["id"], k, int(v)))
                self._conn.commit()
                logger.info(f"[LibraryDB] Committed successfully for id={rec.get('id')}")
            except Exception as e:
                logger.error(f"[LibraryDB] Error in insert_analysis: {e}")
                raise

    def delete_analysis(self, analysis_id: str):
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
            self._conn.commit()

    def list_analyses(self, q: str = "", limit: int = 100, offset: int = 0):
        with self._lock:
            q = (q or "").strip()
            cur = self._conn.cursor()
            if q:
                like = f"%{q}%"
                cur.execute("""
                    SELECT a.*, f.filename
                    FROM analyses a
                    JOIN files f ON f.hash=a.file_hash
                    WHERE f.filename LIKE ? OR a.model LIKE ? OR a.include LIKE ?
                    ORDER BY a.created_at DESC
                    LIMIT ? OFFSET ?
                """, (like, like, like, limit, offset))
            else:
                cur.execute("""
                    SELECT a.*, f.filename
                    FROM analyses a
                    JOIN files f ON f.hash=a.file_hash
                    ORDER BY a.created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            rows = cur.fetchall()

            # fetch class stats in one go
            ids = [r["id"] for r in rows]
            stats = {i: {} for i in ids}
            if ids:
                cur.execute(f"""
                    SELECT analysis_id,class_name,unique_count
                    FROM class_stats
                    WHERE analysis_id IN ({",".join("?"*len(ids))})
                """, ids)
                for r in cur.fetchall():
                    stats[r["analysis_id"]][r["class_name"]] = int(r["unique_count"])
            # assemble
            out = []
            for r in rows:
                d = dict(r)
                d["class_stats"] = stats.get(r["id"], {})
                out.append(d)
            return out
