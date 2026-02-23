# src/cache_manager.py
# Redis Cache Manager – kala_rasa_jtv NLP Service
#
# Key Design:
#   nlp:entity:{query_hash}        → hasil entity extraction  (TTL: 1 jam)
#   cbr:similarity:{query_hash}    → hasil similarity/matching (TTL: 30 menit)
#   cbr:popular:recipes            → rekomendasi populer       (TTL: 1 jam)
#   cbr:index:hash                 → hash versi case index     (TTL: 24 jam)
#   session:{session_id}           → conversation context      (TTL: 2 jam)

from __future__ import annotations

import json
import os
import hashlib
from datetime import timedelta
from typing import Any, Dict, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────

class CacheManager:
    """
    Abstraksi cache layer dengan fallback ke in-memory dict jika Redis tidak tersedia.

    TTL Constants (seconds):
    """

    TTL_ENTITY      = 3600       # 1 jam – hasil NER/entity extraction
    TTL_SIMILARITY  = 1800       # 30 menit – hasil CBR similarity
    TTL_POPULAR     = 3600       # 1 jam – resep populer
    TTL_INDEX_HASH  = 86400      # 24 jam – hash versi case index
    TTL_SESSION     = 7200       # 2 jam – conversation context

    def __init__(self):
        self._redis: Optional[Any] = None
        self._memory_cache: Dict[str, Dict] = {}  # fallback
        self._use_redis = False

        self._connect()

    def _connect(self):
        """Coba konek Redis, fallback ke in-memory jika gagal."""
        if not REDIS_AVAILABLE:
            print("  ⚠ redis-py not installed → using in-memory cache")
            return

        host     = os.getenv("REDIS_HOST", "127.0.0.1")
        port     = int(os.getenv("REDIS_PORT", 6379))
        db       = int(os.getenv("REDIS_DB", 0))
        password = os.getenv("REDIS_PASSWORD") or None
        prefix   = os.getenv("REDIS_PREFIX", "kala_rasa")

        try:
            self._redis = redis.Redis(
                host=host, port=port, db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            self._redis.ping()
            self._prefix = prefix
            self._use_redis = True
            print(f"  ✓ Redis connected ({host}:{port}/db{db})")
        except Exception as e:
            print(f"  ⚠ Redis connection failed ({e}) → using in-memory cache")
            self._redis = None

    # ── Generic get/set/delete ────────────────────────────────────────

    def get(self, key: str) -> Optional[Dict]:
        full_key = self._key(key)
        try:
            if self._use_redis:
                raw = self._redis.get(full_key)
                return json.loads(raw) if raw else None
            else:
                entry = self._memory_cache.get(full_key)
                return entry.get("value") if entry else None
        except Exception:
            return None

    def set(self, key: str, value: Dict, ttl: int = TTL_ENTITY) -> bool:
        full_key = self._key(key)
        try:
            if self._use_redis:
                self._redis.setex(full_key, ttl, json.dumps(value, ensure_ascii=False))
            else:
                self._memory_cache[full_key] = {"value": value}
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        full_key = self._key(key)
        try:
            if self._use_redis:
                self._redis.delete(full_key)
            else:
                self._memory_cache.pop(full_key, None)
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    # ── Specialized methods ────────────────────────────────────────────

    def get_entity_cache(self, query_hash: str) -> Optional[Dict]:
        return self.get(f"nlp:entity:{query_hash}")

    def set_entity_cache(self, query_hash: str, entities: Dict) -> bool:
        return self.set(f"nlp:entity:{query_hash}", entities, self.TTL_ENTITY)

    def get_similarity_cache(self, query_hash: str) -> Optional[Dict]:
        return self.get(f"cbr:similarity:{query_hash}")

    def set_similarity_cache(self, query_hash: str, result: Dict) -> bool:
        return self.set(f"cbr:similarity:{query_hash}", result, self.TTL_SIMILARITY)

    def get_popular_recipes(self) -> Optional[Dict]:
        return self.get("cbr:popular:recipes")

    def set_popular_recipes(self, recipes: Dict) -> bool:
        return self.set("cbr:popular:recipes", recipes, self.TTL_POPULAR)

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.get(f"session:{session_id}")

    def set_session(self, session_id: str, context: Dict) -> bool:
        return self.set(f"session:{session_id}", context, self.TTL_SESSION)

    def delete_session(self, session_id: str) -> bool:
        return self.delete(f"session:{session_id}")

    def get_index_hash(self) -> Optional[str]:
        result = self.get("cbr:index:hash")
        return result.get("hash") if result else None

    def set_index_hash(self, hash_val: str) -> bool:
        return self.set("cbr:index:hash", {"hash": hash_val}, self.TTL_INDEX_HASH)

    # ── Session management ────────────────────────────────────────────

    def refresh_session_ttl(self, session_id: str):
        """Perpanjang TTL session yang masih aktif."""
        if not self._use_redis:
            return
        try:
            self._redis.expire(self._key(f"session:{session_id}"), self.TTL_SESSION)
        except Exception:
            pass

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        stats = {
            "backend":      "redis" if self._use_redis else "in-memory",
            "connected":    self._use_redis,
        }
        if self._use_redis:
            try:
                info = self._redis.info("memory")
                stats["used_memory_human"] = info.get("used_memory_human")
                stats["connected_clients"] = self._redis.info("clients").get("connected_clients")
            except Exception:
                pass
        else:
            stats["entries"] = len(self._memory_cache)
        return stats

    # ── Key builder ────────────────────────────────────────────────────

    def _key(self, key: str) -> str:
        prefix = getattr(self, "_prefix", "kala_rasa")
        return f"{prefix}:{key}"

    # ── Invalidation helpers ───────────────────────────────────────────

    def invalidate_cbr_cache(self):
        """Hapus semua cache CBR saat index di-rebuild."""
        if self._use_redis:
            try:
                prefix = getattr(self, "_prefix", "kala_rasa")
                for key in self._redis.scan_iter(f"{prefix}:cbr:*"):
                    self._redis.delete(key)
            except Exception:
                pass
        else:
            keys_to_del = [k for k in self._memory_cache if ":cbr:" in k]
            for k in keys_to_del:
                del self._memory_cache[k]