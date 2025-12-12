import sqlite3
import uuid
import math
from pathlib import Path
from typing import Optional, List, Dict

DB_PATH = Path(__file__).parent / "agnived.db"


def init_db() -> None:
    """Initialize SQLite database with tables and indexes."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # Users table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            userid TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT DEFAULT (datetime('now')),
            is_active INTEGER DEFAULT 1
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_userid ON users(userid)")
    
    # Uploads table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            content_type TEXT,
            image BLOB NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            species TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_uploads_lat_lon ON uploads(latitude, longitude)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_uploads_user_id ON uploads(user_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_uploads_user_lat_lon ON uploads(user_id, latitude, longitude)")
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")


def get_connection() -> sqlite3.Connection:
    """Get a connection to the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def get_bounding_box(lat: float, lon: float, radius_km: float) -> Dict[str, float]:
    """Calculate bounding box for a radius (in km)."""
    lat_delta = radius_km / 111.32  # 1 degree latitude ≈ 111.32 km
    lon_delta = radius_km / (111.32 * math.cos(math.radians(lat)))
    
    return {
        "min_lat": lat - lat_delta,
        "max_lat": lat + lat_delta,
        "min_lon": lon - lon_delta,
        "max_lon": lon + lon_delta,
    }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km using Haversine formula."""
    R = 6371  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def get_uploads_in_radius(
    lat: float, 
    lon: float, 
    radius_km: float, 
    user_id: Optional[str] = None
) -> List[Dict]:
    """
    Get uploads within exact radius using two-step query (fast + precise).
    
    Step 1: Fast bounding box query (uses lat/lon index)
    Step 2: Precise Haversine distance filter (in-memory)
    """
    bbox = get_bounding_box(lat, lon, radius_km)
    
    conn = get_connection()
    cur = conn.cursor()
    
    if user_id:
        # Query with user filter (uses idx_uploads_user_lat_lon)
        cur.execute("""
            SELECT id, user_id, filename, content_type, latitude, longitude, species, created_at
            FROM uploads
            WHERE user_id = ?
              AND latitude BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
        """, (user_id, bbox["min_lat"], bbox["max_lat"], bbox["min_lon"], bbox["max_lon"]))
    else:
        # Query all users (uses idx_uploads_lat_lon)
        cur.execute("""
            SELECT id, user_id, filename, content_type, latitude, longitude, species, created_at
            FROM uploads
            WHERE latitude BETWEEN ? AND ?
              AND longitude BETWEEN ? AND ?
        """, (bbox["min_lat"], bbox["max_lat"], bbox["min_lon"], bbox["max_lon"]))
    
    results = cur.fetchall()
    conn.close()
    
    # Filter by exact Haversine distance
    filtered = []
    for row in results:
        distance = haversine_distance(lat, lon, row["latitude"], row["longitude"])
        if distance <= radius_km:
            filtered.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "filename": row["filename"],
                "content_type": row["content_type"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "species": row["species"],
                "created_at": row["created_at"],
                "distance_km": round(distance, 2)
            })
    
    # Sort by distance (nearest first)
    filtered.sort(key=lambda x: x["distance_km"])
    return filtered


def get_user_score(user_id: str) -> int:
    """Get user's upload count (score)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM uploads WHERE user_id = ?", (user_id,))
    score = cur.fetchone()[0]
    conn.close()
    return score


def get_user_uploads(user_id: str, limit: Optional[int] = None) -> List[Dict]:
    """Get all uploads by a specific user."""
    conn = get_connection()
    cur = conn.cursor()
    
    query = """
        SELECT id, filename, content_type, latitude, longitude, species, created_at
        FROM uploads
        WHERE user_id = ?
        ORDER BY created_at DESC
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query, (user_id,))
    results = cur.fetchall()
    conn.close()
    
    return [dict(row) for row in results]


if __name__ == "__main__":
    init_db()