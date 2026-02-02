import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sqlite3
from datetime import datetime
import re
import hashlib
import html

# =========================
# CONFIG
# =========================
DB_NAME = "aurora_ml.db"
ADMIN_PASSWORD = "aurora123"  # cambia si quieres
NUM_MESAS = 4

# Maestro SKU/EAN en la misma carpeta que app.py
MASTER_FILE = "maestro_sku_ean.xlsx"


# =========================
# TIMEZONE CHILE
# =========================
try:
    from zoneinfo import ZoneInfo  # py3.9+
    CL_TZ = ZoneInfo("America/Santiago")
    UTC_TZ = ZoneInfo("UTC")
except Exception:
    CL_TZ = None
    UTC_TZ = None


# PDF manifiestos
try:
    import pdfplumber
    HAS_PDF_LIB = True
except ImportError:
    HAS_PDF_LIB = False


# =========================
# UTILIDADES
# =========================
def now_iso():
    # Guardamos UTC naive ISO (sin tz). Luego to_chile_display lo convierte a hora Chile.
    return datetime.utcnow().isoformat(timespec="seconds")



# =========================
# TEXT HELPERS
# =========================
UBC_RE = re.compile(r"\[\s*UBC\s*:\s*([^\]]+)\]", re.IGNORECASE)

def split_title_ubc(title: str):
    """Return (title_without_ubc, ubc_str_or_empty)."""
    t = str(title or "").strip()
    ubc = ""
    m = UBC_RE.search(t)
    if m:
        ubc = m.group(1).strip()
        # remove the whole [UBC: ...] chunk
        t = UBC_RE.sub("", t).strip()
        # collapse double spaces
        t = re.sub(r"\s{2,}", " ", t)
    return t, ubc

def to_chile_display(iso_str: str) -> str:
    """Convierte ISO guardado (asumido UTC server) a hora Chile para mostrar."""
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso_str))
        if CL_TZ is None or UTC_TZ is None:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        dt_utc = dt.replace(tzinfo=UTC_TZ)
        dt_cl = dt_utc.astimezone(CL_TZ)
        return dt_cl.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(iso_str)


def normalize_sku(value) -> str:
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    if re.fullmatch(r"\d+(\.\d+)?[eE][+-]?\d+", s):
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    return s


def only_digits(s: str) -> str:
    return re.sub(r"\D", "", str(s or ""))


def split_barcodes(cell_value) -> list[str]:
    if cell_value is None:
        return []
    s = str(cell_value).strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"[\s,;]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        d = only_digits(p)
        if d:
            out.append(d)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def force_tel_keyboard(label: str):
    """Fuerza teclado numérico tipo 'teléfono' para el input con aria-label=label."""
    safe = label.replace("\\", "\\\\").replace('"', '\\"')
    components.html(
        f"""
        <script>
        (function() {{
          const label = "{safe}";
          let tries = 0;
          function apply() {{
            const inputs = window.parent.document.querySelectorAll('input[aria-label="' + label + '"]');
            if (!inputs || inputs.length === 0) {{
              tries++;
              if (tries < 30) setTimeout(apply, 200);
              return;
            }}
            inputs.forEach((el) => {{
              try {{
                el.setAttribute('type', 'tel');
                el.setAttribute('inputmode', 'numeric');
                el.setAttribute('pattern', '[0-9]*');
                el.setAttribute('autocomplete', 'off');
              }} catch (e) {{}}
            }});
          }}
          apply();
          setTimeout(apply, 500);
          setTimeout(apply, 1200);
        }})();
        </script>
        """,
        height=0,
    )


def autofocus_input(label: str):
    """Pone foco inmediato en un input por aria-label."""
    safe = label.replace("\\", "\\\\").replace('"', '\\"')
    components.html(
        f"""
        <script>
        (function() {{
          const label = "{safe}";
          let tries = 0;
          function focusIt() {{
            const el = window.parent.document.querySelector('input[aria-label="' + label + '"]');
            if (!el) {{
              tries++;
              if (tries < 40) setTimeout(focusIt, 120);
              return;
            }}
            try {{
              el.focus();
              el.select();
            }} catch (e) {{}}
          }}
          focusIt();
          setTimeout(focusIt, 300);
          setTimeout(focusIt, 900);
        }})();
        </script>
        """,
        height=0,
    )


# =========================
# DB INIT
# =========================
def init_db():
    conn = get_conn()
    c = conn.cursor()

    # --- FLEX/COLECTA ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ml_order_id TEXT UNIQUE,
        buyer TEXT,
        created_at TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        sku_ml TEXT,
        title_ml TEXT,
        title_tec TEXT,
        qty INTEGER
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS pickers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_ots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ot_code TEXT UNIQUE,
        picker_id INTEGER,
        status TEXT,
        created_at TEXT,
        closed_at TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ot_id INTEGER,
        sku_ml TEXT,
        title_ml TEXT,
        title_tec TEXT,
        qty_total INTEGER,
        qty_picked INTEGER DEFAULT 0,
        status TEXT DEFAULT 'PENDING',
        decided_at TEXT,
        confirm_mode TEXT,
        defer_rank INTEGER DEFAULT 0,
        defer_at TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_incidences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ot_id INTEGER,
        sku_ml TEXT,
        qty_total INTEGER,
        qty_picked INTEGER,
        qty_missing INTEGER,
        reason TEXT,
        created_at TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS ot_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ot_id INTEGER,
        order_id INTEGER
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS sorting_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ot_id INTEGER,
        order_id INTEGER,
        status TEXT,
        marked_at TEXT,
        mesa INTEGER,
        printed_at TEXT
    );
    """)

    # Maestro EAN/SKU (común)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sku_barcodes (
        barcode TEXT PRIMARY KEY,
        sku_ml TEXT
    );
    """)

    # --- FULL: Acopio ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS full_batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_name TEXT,
        status TEXT DEFAULT 'OPEN',
        created_at TEXT,
        closed_at TEXT
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS full_batch_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER,
        sku_ml TEXT,
        title TEXT,
        areas TEXT,
        nros TEXT,
        etiquetar TEXT,
        es_pack TEXT,
        instruccion TEXT,
        vence TEXT,
        qty_required INTEGER DEFAULT 0,
        qty_checked INTEGER DEFAULT 0,
        status TEXT DEFAULT 'PENDING',
        updated_at TEXT,
        UNIQUE(batch_id, sku_ml)
    );
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS full_incidences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER,
        sku_ml TEXT,
        qty_required INTEGER,
        qty_checked INTEGER,
        diff INTEGER,
        reason TEXT,
        created_at TEXT
    );
    """)

    # --- SORTING (Camarero) ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS sorting_manifests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at TEXT,
        status TEXT  -- ACTIVE / DONE
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sorting_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        manifest_id INTEGER,
        page_no INTEGER,
        mesa INTEGER,
        status TEXT, -- PENDING / IN_PROGRESS / DONE
        created_at TEXT,
        closed_at TEXT,
        UNIQUE(manifest_id, page_no)
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sorting_run_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER,
        seq INTEGER,
        ml_order_id TEXT,
        pack_id TEXT,
        sku TEXT,
        title_ml TEXT,
        title_tec TEXT,
        qty INTEGER,
        buyer TEXT,
        address TEXT,
        shipment_id TEXT,
        status TEXT, -- PENDING / DONE / INCIDENCE
        done_at TEXT,
        incidence_note TEXT
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sorting_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        manifest_id INTEGER,
        pack_id TEXT,
        shipment_id TEXT,
        buyer TEXT,
        address TEXT,
        raw TEXT,
        UNIQUE(manifest_id, pack_id)
    );
    """)

    # --- MIGRACIONES SUAVES (para BD antiguas) ---
    def _cols(table: str) -> set:
        try:
            c.execute(f"PRAGMA table_info({table});")
            return {r[1] for r in c.fetchall()}
        except Exception:
            return set()

    def _ensure_col(table: str, col: str, ddl: str):
        cols = _cols(table)
        if col in cols:
            return
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl};")
        except Exception:
            # Si falla (por locks o tablas raras), no botar la app.
            pass

        # picking_tasks (nuevas columnas para reordenar por "Surtido en venta")
    _ensure_col("picking_tasks", "defer_rank", "INTEGER DEFAULT 0")
    _ensure_col("picking_tasks", "defer_at", "TEXT")

# sorting_manifests
    _ensure_col("sorting_manifests", "name", "TEXT")
    _ensure_col("sorting_manifests", "created_at", "TEXT")
    _ensure_col("sorting_manifests", "status", "TEXT")

    # sorting_runs
    _ensure_col("sorting_runs", "manifest_id", "INTEGER")
    _ensure_col("sorting_runs", "page_no", "INTEGER")
    _ensure_col("sorting_runs", "mesa", "INTEGER")
    _ensure_col("sorting_runs", "status", "TEXT")
    _ensure_col("sorting_runs", "created_at", "TEXT")
    _ensure_col("sorting_runs", "closed_at", "TEXT")

    # sorting_run_items
    _ensure_col("sorting_run_items", "run_id", "INTEGER")
    _ensure_col("sorting_run_items", "seq", "INTEGER")
    _ensure_col("sorting_run_items", "ml_order_id", "TEXT")
    _ensure_col("sorting_run_items", "pack_id", "TEXT")
    _ensure_col("sorting_run_items", "sku", "TEXT")
    _ensure_col("sorting_run_items", "title_ml", "TEXT")
    _ensure_col("sorting_run_items", "title_tec", "TEXT")
    _ensure_col("sorting_run_items", "qty", "INTEGER")
    _ensure_col("sorting_run_items", "buyer", "TEXT")
    _ensure_col("sorting_run_items", "address", "TEXT")
    _ensure_col("sorting_run_items", "shipment_id", "TEXT")
    _ensure_col("sorting_run_items", "status", "TEXT")
    _ensure_col("sorting_run_items", "done_at", "TEXT")
    _ensure_col("sorting_run_items", "incidence_note", "TEXT")

    # sorting_labels
    _ensure_col("sorting_labels", "manifest_id", "INTEGER")
    _ensure_col("sorting_labels", "pack_id", "TEXT")
    _ensure_col("sorting_labels", "shipment_id", "TEXT")
    _ensure_col("sorting_labels", "buyer", "TEXT")
    _ensure_col("sorting_labels", "address", "TEXT")
    _ensure_col("sorting_labels", "raw", "TEXT")

    # Asegurar índices/constraints para UPSERT (BD antiguas)
    try:
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sorting_labels_manifest_pack ON sorting_labels(manifest_id, pack_id);")
    except Exception:
        pass
    try:
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sorting_runs_manifest_page ON sorting_runs(manifest_id, page_no);")
    except Exception:
        pass

    conn.commit()
    conn.close()


# =========================
# MAESTRO SKU/EAN (AUTO)
# =========================
def load_master_from_path(path: str) -> tuple[dict, dict, list]:
    inv_map_sku = {}
    barcode_to_sku = {}
    conflicts = []

    if not path or not os.path.exists(path):
        return inv_map_sku, barcode_to_sku, conflicts

    df = pd.read_excel(path, dtype=str)
    cols = df.columns.tolist()
    lower = [str(c).strip().lower() for c in cols]

    sku_col = None
    if "sku" in lower:
        sku_col = cols[lower.index("sku")]

    tech_col = None
    for cand in ["artículo", "articulo", "descripcion", "descripción", "nombre", "producto", "detalle"]:
        if cand in lower:
            tech_col = cols[lower.index(cand)]
            break

    barcode_col = None
    for cand in ["codigo de barras", "código de barras", "barcode", "ean", "eans"]:
        if cand in lower:
            barcode_col = cols[lower.index(cand)]
            break

    # Fallback por si el archivo no trae headers claros
    if sku_col is None or tech_col is None:
        df0 = pd.read_excel(path, header=None, dtype=str)
        if df0.shape[1] >= 2:
            a, b = df0.columns[0], df0.columns[1]
            sample = df0.head(200)

            def score(series):
                s = 0
                for v in series:
                    if re.fullmatch(r"\d{4,}", normalize_sku(v)):
                        s += 1
                return s

            sa, sb = score(sample[a]), score(sample[b])
            if sb >= sa:
                sku_col, tech_col = b, a
            else:
                sku_col, tech_col = a, b
            df = df0
            barcode_col = None  # sin header no asumimos dónde está EAN

    for _, r in df.iterrows():
        sku = normalize_sku(r.get(sku_col, ""))
        if not sku:
            continue

        tech = str(r.get(tech_col, "")).strip() if tech_col is not None else ""
        if tech and tech.lower() != "nan":
            inv_map_sku[sku] = tech

        if barcode_col is not None:
            codes = split_barcodes(r.get(barcode_col, ""))
            for code in codes:
                if code in barcode_to_sku and barcode_to_sku[code] != sku:
                    conflicts.append((code, barcode_to_sku[code], sku))
                    continue
                barcode_to_sku[code] = sku

    return inv_map_sku, barcode_to_sku, conflicts


# Cache extra: lookup directo del título "tal cual" en el maestro (sin limpiar)
_MASTER_DF_CACHE = {"path": None, "mtime": None, "df": None}

def _load_master_df_cached(path: str):
    """Carga el Excel del maestro una sola vez (por mtime) para poder buscar el texto crudo."""
    if not path or not os.path.exists(path):
        return None
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    if (_MASTER_DF_CACHE.get("path") == path and _MASTER_DF_CACHE.get("mtime") == mtime
            and _MASTER_DF_CACHE.get("df") is not None):
        return _MASTER_DF_CACHE["df"]

    try:
        dfm = pd.read_excel(path, dtype=str)
    except Exception:
        return None

    _MASTER_DF_CACHE.update({"path": path, "mtime": mtime, "df": dfm})
    return dfm

def master_raw_title_lookup(path: str, sku: str) -> str:
    """Devuelve el texto EXACTO del maestro para ese SKU (tal cual viene en la celda)."""
    dfm = _load_master_df_cached(path)
    if dfm is None or dfm.empty:
        return ""
    cols = list(dfm.columns)
    lower = [str(c).strip().lower() for c in cols]

    # columna SKU
    sku_col = None
    if "sku" in lower:
        sku_col = cols[lower.index("sku")]
    if sku_col is None:
        return ""

    # preferir columnas típicas de descripción/título
    pref = [
        "descripción", "descripcion", "artículo", "articulo",
        "detalle", "producto", "nombre", "descripción pack", "nombre pack"
    ]
    title_col = None
    for cand in pref:
        if cand in lower:
            title_col = cols[lower.index(cand)]
            break
    # si no hay, tomar la primera no-SKU
    if title_col is None:
        for c in cols:
            if c != sku_col:
                title_col = c
                break
    if title_col is None:
        return ""

    target = normalize_sku(sku)
    if not target:
        return ""

    try:
        ser = dfm[sku_col].astype(str).map(normalize_sku)
        hits = dfm.loc[ser == target]
    except Exception:
        return ""

    if hits.empty:
        return ""

    val = hits.iloc[0][title_col]
    if val is None:
        return ""
    sval = str(val)
    if sval.lower() == "nan":
        return ""
    return sval


def upsert_barcodes_to_db(barcode_to_sku: dict):
    if not barcode_to_sku:
        return
    conn = get_conn()
    c = conn.cursor()
    for bc, sku in barcode_to_sku.items():
        c.execute("INSERT OR REPLACE INTO sku_barcodes (barcode, sku_ml) VALUES (?, ?)", (bc, sku))
    conn.commit()
    conn.close()


def resolve_scan_to_sku(scan: str, barcode_to_sku: dict) -> str:
    raw = str(scan).strip()
    digits = only_digits(raw)
    if digits and digits in barcode_to_sku:
        return barcode_to_sku[digits]
    return normalize_sku(raw)


def extract_location_suffix(text: str) -> str:
    """Extracts location/UBC suffix like '[UBC: 1234]' from a title."""
    t = str(text or "").strip()
    if not t:
        return ""
    # Common pattern in Aurora: '[UBC: 2260]' or '[ubc: 2260]'
    m = re.search(r"(\[\s*UBC\s*:\s*[^\]]+\])\s*$", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Sometimes without brackets: 'UBC: 2260' at end
    m = re.search(r"(UBC\s*:\s*\d+)\s*$", t, flags=re.IGNORECASE)
    if m:
        return f"[{m.group(1).strip()}]"
    return ""

def strip_location_suffix(text: str) -> str:
    """Remove trailing location suffix like '[UBC: 2260]' if present."""
    t = str(text or "").strip()
    if not t:
        return ""
    # remove bracketed suffix
    t2 = re.sub(r"\s*(\[\s*UBC\s*:\s*[^\]]+\])\s*$", "", t, flags=re.IGNORECASE).strip()
    # remove unbracketed suffix
    t2 = re.sub(r"\s*(UBC\s*:\s*\d+)\s*$", "", t2, flags=re.IGNORECASE).strip()
    return t2



def with_location(title_display: str, title_tec: str) -> str:
    """Ensures the product title shown includes the location suffix when available."""
    base = str(title_display or "").strip()
    tec = str(title_tec or "").strip()

    # If base already contains a suffix, keep it
    if extract_location_suffix(base):
        return base

    # If technical title contains suffix, append it
    suf = extract_location_suffix(tec)
    if suf:
        return f"{base} {suf}".strip()

    return base


@st.cache_data(show_spinner=False)
def get_master_cached(master_path: str) -> tuple[dict, dict, list]:
    return load_master_from_path(master_path)


def master_bootstrap(master_path: str):
    inv_map_sku, barcode_to_sku, conflicts = get_master_cached(master_path)
    upsert_barcodes_to_db(barcode_to_sku)
    return inv_map_sku, barcode_to_sku, conflicts


# =========================
# PARSER PDF MANIFIESTO
# =========================
def parse_manifest_pdf(uploaded_file) -> pd.DataFrame:
    if not HAS_PDF_LIB:
        raise RuntimeError("Falta pdfplumber. Agrega 'pdfplumber' a requirements.txt")

    records = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            text = text.replace("\r", "\n")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            for i, line in enumerate(lines):
                if "Cantidad" not in line:
                    continue

                m_qty = re.search(r"Cantidad\s*[:#]?\s*([0-9]+)", line)
                if not m_qty:
                    continue
                qty = int(m_qty.group(1))

                sku = None
                order = None
                buyer = ""
                start = max(0, i - 12)

                m_sku_cur = re.search(r"SKU\s*[:#]?\s*([0-9A-Za-z.\-]+)", line)
                if m_sku_cur:
                    sku = normalize_sku(m_sku_cur.group(1))

                m_ord_cur = re.search(r"Venta\s*[:#]?\s*([0-9]+)", line)
                if m_ord_cur:
                    order = m_ord_cur.group(1).strip()

                if sku is None or order is None:
                    for j in range(i - 1, start - 1, -1):
                        l = lines[j]
                        if sku is None and "SKU" in l:
                            m_sku = re.search(r"SKU\s*[:#]?\s*([0-9A-Za-z.\-]+)", l)
                            if m_sku:
                                sku = normalize_sku(m_sku.group(1))
                        if order is None and "Venta" in l:
                            m_ord = re.search(r"Venta\s*[:#]?\s*([0-9]+)", l)
                            if m_ord:
                                order = m_ord.group(1).strip()

                if not (order and sku):
                    continue

                venta_idx = None
                for k in range(start, i):
                    if "Venta" in lines[k]:
                        venta_idx = k
                        break
                if venta_idx is not None:
                    for k in range(venta_idx + 1, i):
                        cand = lines[k].strip()
                        low = cand.lower()
                        if any(tok in low for tok in [
                            "venta", "sku", "pack id", "cantidad",
                            "código carrier", "firma carrier", "fecha y hora de retiro"
                        ]):
                            continue
                        if re.fullmatch(r"[0-9 .:/-]+", cand):
                            continue
                        buyer = cand
                        break

                records.append({
                    "ml_order_id": order,
                    "buyer": buyer,
                    "sku_ml": sku,
                    "title_ml": "",
                    "qty": qty
                })

    return pd.DataFrame(records, columns=["ml_order_id", "buyer", "sku_ml", "title_ml", "qty"])


# =========================
# IMPORTAR VENTAS (FLEX)
# =========================
def import_sales_excel(file) -> pd.DataFrame:
    """Importa reporte de ventas ML.

    Importante: en los reportes de ML, los envíos con varios productos vienen con una fila
    de cabecera 'Paquete de X productos' (sin SKU / sin unidades) y luego X filas con los ítems.
    Para que el KPI 'Ventas' refleje lo que tú ves por colores (paquetes/envíos), agrupamos esos
    ítems bajo el ID de la fila cabecera.
    """
    df = pd.read_excel(file, header=[4, 5])
    df.columns = [" | ".join([str(x) for x in col if str(x) != "nan"]) for col in df.columns]

    COLUMN_ORDER_ID = "Ventas | # de venta"
    COLUMN_STATUS = "Ventas | Estado"
    COLUMN_QTY = "Ventas | Unidades"
    COLUMN_SKU = "Publicaciones | SKU"
    COLUMN_TITLE = "Publicaciones | Título de la publicación"
    COLUMN_BUYER = "Compradores | Comprador"

    required = [COLUMN_ORDER_ID, COLUMN_STATUS, COLUMN_QTY, COLUMN_SKU, COLUMN_TITLE, COLUMN_BUYER]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    # Normalizamos a texto para trabajar seguro
    work = df[required].copy()
    work.columns = ["ml_order_id", "status", "qty", "sku_ml", "title_ml", "buyer"]

    # Helpers
    def _clean_str(x) -> str:
        if pd.isna(x):
            return ""
        return str(x).strip()

    records = []
    current_pkg_id = None
    current_pkg_buyer = ""
    remaining_items = 0

    pkg_re = re.compile(r"^Paquete\s+de\s+(\d+)\s+productos?$", re.IGNORECASE)

    for _, r in work.iterrows():
        status = _clean_str(r.get("status"))
        ml_id = _clean_str(r.get("ml_order_id"))
        buyer = _clean_str(r.get("buyer"))
        sku = _clean_str(r.get("sku_ml"))
        title = _clean_str(r.get("title_ml"))
        qty = pd.to_numeric(r.get("qty"), errors="coerce")

        # Detecta fila cabecera del paquete (no trae SKU/qty)
        m = pkg_re.match(status)
        if m:
            try:
                remaining_items = int(m.group(1))
            except Exception:
                remaining_items = 0
            current_pkg_id = ml_id if ml_id else None
            current_pkg_buyer = buyer
            continue

        # Filas sin SKU/qty -> se ignoran
        if not sku or pd.isna(qty):
            continue

        qty_int = int(qty) if not pd.isna(qty) else 0
        if qty_int <= 0:
            continue

        sku_norm = normalize_sku(sku)

        # Si estamos dentro de un paquete, agrupamos bajo el ID del paquete
        if current_pkg_id and remaining_items > 0:
            records.append(
                {
                    "ml_order_id": current_pkg_id,
                    "buyer": current_pkg_buyer or buyer,
                    "sku_ml": sku_norm,
                    "title_ml": title,
                    "qty": qty_int,
                }
            )
            remaining_items -= 1
            if remaining_items <= 0:
                current_pkg_id = None
                current_pkg_buyer = ""
            continue

        # Venta normal (1 producto)
        records.append(
            {
                "ml_order_id": ml_id,
                "buyer": buyer,
                "sku_ml": sku_norm,
                "title_ml": title,
                "qty": qty_int,
            }
        )

    out = pd.DataFrame(records, columns=["ml_order_id", "buyer", "sku_ml", "title_ml", "qty"])
    return out
def save_orders_and_build_ots(sales_df: pd.DataFrame, inv_map_sku: dict, num_pickers: int):
    conn = get_conn()
    c = conn.cursor()

    # Reset corrida (no borra histórico; eso lo hace admin reset total)
    c.execute("DELETE FROM picking_tasks;")
    c.execute("DELETE FROM picking_incidences;")
    c.execute("DELETE FROM ot_orders;")
    c.execute("DELETE FROM sorting_status;")
    c.execute("DELETE FROM picking_ots;")
    c.execute("DELETE FROM pickers;")

    order_id_by_ml = {}
    for ml_order_id, g in sales_df.groupby("ml_order_id"):
        ml_order_id = str(ml_order_id).strip()
        buyer = str(g["buyer"].iloc[0]) if "buyer" in g.columns else ""
        created = now_iso()

        c.execute("SELECT id FROM orders WHERE ml_order_id = ?", (ml_order_id,))
        row = c.fetchone()
        if row:
            order_id = row[0]
            c.execute("UPDATE orders SET buyer=?, created_at=? WHERE id=?", (buyer, created, order_id))
            c.execute("DELETE FROM order_items WHERE order_id=?", (order_id,))
        else:
            c.execute("INSERT INTO orders (ml_order_id, buyer, created_at) VALUES (?,?,?)", (ml_order_id, buyer, created))
            order_id = c.lastrowid

        order_id_by_ml[ml_order_id] = order_id

        for _, r in g.iterrows():
            sku = normalize_sku(r["sku_ml"])
            qty = int(r["qty"])
            title_ml = str(r.get("title_ml", "") or "").strip()
            title_tec = inv_map_sku.get(sku, "")
            title_eff = title_tec if title_tec else title_ml

            c.execute(
                "INSERT INTO order_items (order_id, sku_ml, title_ml, title_tec, qty) VALUES (?,?,?,?,?)",
                (order_id, sku, title_eff, title_tec, qty)
            )

    picker_ids = []
    for i in range(int(num_pickers)):
        name = f"P{i+1}"
        c.execute("INSERT INTO pickers (name) VALUES (?)", (name,))
        picker_ids.append(c.lastrowid)

    ot_ids = []
    for pid in picker_ids:
        c.execute(
            "INSERT INTO picking_ots (ot_code, picker_id, status, created_at, closed_at) VALUES (?,?,?,?,?)",
            ("", pid, "OPEN", now_iso(), None)
        )
        ot_id = c.lastrowid
        ot_code = f"OT{ot_id:06d}"
        c.execute("UPDATE picking_ots SET ot_code=? WHERE id=?", (ot_code, ot_id))
        ot_ids.append(ot_id)

    unique_orders = sales_df[["ml_order_id"]].drop_duplicates().reset_index(drop=True)
    assignments = {}
    for idx, row in unique_orders.iterrows():
        ot_id = ot_ids[idx % len(ot_ids)]
        assignments[str(row["ml_order_id"]).strip()] = ot_id

    for idx, (ml_order_id, ot_id) in enumerate(assignments.items()):
        order_id = order_id_by_ml[ml_order_id]
        mesa = (idx % NUM_MESAS) + 1
        c.execute("INSERT INTO ot_orders (ot_id, order_id) VALUES (?,?)", (ot_id, order_id))
        c.execute("""
            INSERT INTO sorting_status (ot_id, order_id, status, marked_at, mesa, printed_at)
            VALUES (?,?,?,?,?,?)
        """, (ot_id, order_id, "PENDING", None, mesa, None))

    for ot_id in ot_ids:
        c.execute("""
            SELECT oi.sku_ml,
                   COALESCE(NULLIF(oi.title_tec,''), oi.title_ml) AS title,
                   MAX(COALESCE(oi.title_tec,'')) AS title_tec_any,
                   SUM(oi.qty) as total
            FROM ot_orders oo
            JOIN order_items oi ON oi.order_id = oo.order_id
            WHERE oo.ot_id = ?
            GROUP BY oi.sku_ml, title
            ORDER BY CAST(oi.sku_ml AS INTEGER), oi.sku_ml
        """, (ot_id,))
        rows = c.fetchall()
        for sku, title, title_tec_any, total in rows:
            c.execute("""
                INSERT INTO picking_tasks (ot_id, sku_ml, title_ml, title_tec, qty_total, qty_picked, status, decided_at, confirm_mode)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (ot_id, sku, title, title_tec_any, int(total), 0, "PENDING", None, None))

    conn.commit()
    conn.close()


# =========================
# UI: LOBBY APP (MODO)
# =========================
def page_app_lobby():
    st.markdown("## Ferretería Aurora – WMS")
    st.caption("Selecciona el flujo de trabajo")

    st.markdown(
        """
        <style>
        .lobbybtn button {
            width: 100% !important;
            padding: 22px 14px !important;
            font-size: 22px !important;
            font-weight: 900 !important;
            border-radius: 18px !important;
        }
        .lobbywrap { max-width: 1100px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="lobbywrap">', unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown('<div class="lobbybtn">', unsafe_allow_html=True)
        if st.button("📦 Picking pedidos Flex y Colecta", key="mode_flex_pick"):
            st.session_state.app_mode = "FLEX_PICK"
            st.session_state.pop("selected_picker", None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Picking por OT, incidencias, admin, etc.")

    with colB:
        st.markdown('<div class="lobbybtn">', unsafe_allow_html=True)
        if st.button("🧾 Sorting pedidos Flex y Colecta", key="mode_sorting"):
            st.session_state.app_mode = "SORTING"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Camarero por mesa/página (1 página = 1 mesa).")

    with colC:
        st.markdown('<div class="lobbybtn">', unsafe_allow_html=True)
        if st.button("🏷️ Preparación productos Full", key="mode_full"):
            st.session_state.app_mode = "FULL"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Control de acopio Full (escaneo + chequeo vs Excel).")

    st.markdown("</div>", unsafe_allow_html=True)
def page_import(inv_map_sku: dict):
    st.header("Importar ventas")
    origen = st.radio("Origen", ["Excel Mercado Libre", "Manifiesto PDF (etiquetas)"], horizontal=True)
    num_pickers = st.number_input("Cantidad de pickeadores", min_value=1, max_value=20, value=5, step=1)

    if origen == "Excel Mercado Libre":
        file = st.file_uploader("Ventas ML (xlsx)", type=["xlsx"], key="ml_excel")
        if not file:
            st.info("Sube el Excel de ventas.")
            return
        sales_df = import_sales_excel(file)
    else:
        pdf_file = st.file_uploader("Manifiesto PDF", type=["pdf"], key="ml_pdf")
        if not pdf_file:
            st.info("Sube el PDF.")
            return
        sales_df = parse_manifest_pdf(pdf_file)

    st.subheader("Vista previa")
    st.dataframe(sales_df.head(30))

    if st.button("Cargar y generar OTs"):
        save_orders_and_build_ots(sales_df, inv_map_sku, int(num_pickers))
        st.success("OTs creadas. Anda a Picking y selecciona P1, P2, ...")


# =========================
# UI: PICKING (FLEX)
# =========================
def picking_lobby():
    st.markdown("### Picking")
    st.caption("Selecciona tu pickeador")

    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name FROM pickers ORDER BY name")
    rows = c.fetchall()
    conn.close()

    if not rows:
        st.info("Aún no hay pickeadores. Primero importa ventas y genera OTs.")
        return False

    pickers = [r[0] for r in rows]

    st.markdown(
        """
        <style>
        .bigbtn button {
            width: 100% !important;
            padding: 18px 10px !important;
            font-size: 22px !important;
            font-weight: 900 !important;
            border-radius: 16px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(3)
    chosen = None
    for i, p in enumerate(pickers):
        with cols[i % 3]:
            st.markdown('<div class="bigbtn">', unsafe_allow_html=True)
            if st.button(p, key=f"pick_{p}"):
                chosen = p
            st.markdown('</div>', unsafe_allow_html=True)

    if chosen:
        st.session_state.selected_picker = chosen
        st.rerun()

    return "selected_picker" in st.session_state


def page_picking():
    if "selected_picker" not in st.session_state:
        ok = picking_lobby()
        if not ok:
            return

    picker_name = st.session_state.get("selected_picker", "")
    if not picker_name:
        st.session_state.pop("selected_picker", None)
        st.rerun()

    topA, topB = st.columns([2, 1])
    with topA:
        st.markdown(f"### Picking (PDA) — {picker_name}")
    with topB:
        if st.button("Cambiar pickeador"):
            st.session_state.pop("selected_picker", None)
            st.rerun()

    st.markdown(
        """
        <style>
        div.block-container { padding-top: 0.6rem; padding-bottom: 1rem; }
        .hero { padding: 10px 12px; border-radius: 12px; background: rgba(0,0,0,0.04); margin: 6px 0 8px 0; }
        .hero .sku { font-size: 26px; font-weight: 900; margin: 0; }
        .hero .prod { font-size: 22px; font-weight: 800; margin: 6px 0 0 0; line-height: 1.15; }
        .hero .qty { font-size: 26px; font-weight: 900; margin: 8px 0 0 0; }
.hero .loc { font-size: 18px; font-weight: 900; margin: 6px 0 0 0; opacity: 0.9; }
        .smallcap { font-size: 12px; opacity: 0.75; margin: 0 0 4px 0; }
        .scanok { display:inline-block; padding: 6px 10px; border-radius: 10px; font-weight: 900; }
        .ok { background: rgba(0, 200, 0, 0.15); }
        .bad { background: rgba(255, 0, 0, 0.12); }
        </style>
        """,
        unsafe_allow_html=True
    )

    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT barcode, sku_ml FROM sku_barcodes")
    barcode_to_sku = {r[0]: r[1] for r in c.fetchall()}

    c.execute("""
        SELECT po.id, po.ot_code, po.status
        FROM picking_ots po
        JOIN pickers pk ON pk.id = po.picker_id
        WHERE pk.name = ?
        ORDER BY po.ot_code
    """, (picker_name,))
    ots = c.fetchall()
    if not ots:
        st.error(f"No existe OT para {picker_name}. Importa ventas y genera OTs.")
        conn.close()
        return

    ot_row = None
    for r in ots:
        if r[2] != "PICKED":
            ot_row = r
            break
    if ot_row is None:
        ot_row = ots[0]

    ot_id, ot_code, ot_status = ot_row

    if ot_status == "PICKED":
        st.success("OT cerrada (PICKED).")
        conn.close()
        return

    c.execute("""
        SELECT id, sku_ml, title_ml, title_tec,
               qty_total, qty_picked, status
        FROM picking_tasks
        WHERE ot_id=?
        ORDER BY COALESCE(defer_rank,0) ASC, CAST(sku_ml AS INTEGER), sku_ml
    """, (ot_id,))
    tasks = c.fetchall()

    total_tasks = len(tasks)
    done_small = sum(1 for t in tasks if t[6] in ("DONE", "INCIDENCE"))
    st.caption(f"Resueltos: {done_small}/{total_tasks}")

    current = next((t for t in tasks if t[6] == "PENDING"), None)
    if current is None:
        st.success("No quedan SKUs pendientes.")
        if st.button("Cerrar OT"):
            c.execute("UPDATE picking_ots SET status='PICKED', closed_at=? WHERE id=?", (now_iso(), ot_id))
            conn.commit()
            st.success("OT cerrada.")
        conn.close()
        return

    task_id, sku_expected, title_ml, title_tec, qty_total, qty_picked, status = current

    # Título: prioridad absoluta al texto crudo del maestro (tal cual). Si no existe, cae a title_tec/title_ml.
    raw_master = master_raw_title_lookup(MASTER_FILE, sku_expected)
    producto_show = raw_master if raw_master else (title_tec if title_tec not in (None, "") else (title_ml or ""))
    if "pick_state" not in st.session_state:
        st.session_state.pick_state = {}
    state = st.session_state.pick_state
    if str(task_id) not in state:
        state[str(task_id)] = {
            "confirmed": False,
            "confirm_mode": None,
            "scan_value": "",
            "qty_input": "",
            "needs_decision": False,
            "missing": 0,
            "show_manual_confirm": False,
            "scan_status": "idle",
            "scan_msg": "",
            "last_sku_expected": None
        }
    s = state[str(task_id)]

    if s.get("last_sku_expected") != sku_expected:
        s["last_sku_expected"] = sku_expected
        s["confirmed"] = False
        s["confirm_mode"] = None
        s["needs_decision"] = False
        s["missing"] = 0
        s["show_manual_confirm"] = False
        s["scan_status"] = "idle"
        s["scan_msg"] = ""
        s["qty_input"] = ""
        s["scan_value"] = ""

    # Tarjeta principal: mostrar el título tal cual (incluye UBC/ubicación aunque venga al inicio/medio/final)
    st.caption(f"OT: {ot_code}")
    st.markdown(f"### SKU: {sku_expected}")

    st.markdown(
        f'<div class="hero"><div class="prod" style="white-space: normal; overflow-wrap: anywhere; word-break: break-word;">{html.escape(str(producto_show))}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"### Solicitado: {qty_total}")

    if s["scan_status"] == "ok":
        st.markdown(
            f'<span class="scanok ok">✅ OK</span> {s["scan_msg"]}',
            unsafe_allow_html=True,
        )
    elif s["scan_status"] == "bad":
        st.markdown(
            f'<span class="scanok bad">❌ ERROR</span> {s["scan_msg"]}',
            unsafe_allow_html=True,
        )
        st.markdown(f'<span class="scanok bad">❌ ERROR</span> {s["scan_msg"]}', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        scan_label = "Escaneo"
        scan = st.text_input(scan_label, value=s["scan_value"], key=f"scan_{task_id}")
        force_tel_keyboard(scan_label)
        autofocus_input(scan_label)

    with col2:
        if st.button("Validar"):
            sku_detected = resolve_scan_to_sku(scan, barcode_to_sku)
            if not sku_detected:
                s["scan_status"] = "bad"
                s["scan_msg"] = "No se pudo leer el código."
                s["confirmed"] = False
                s["confirm_mode"] = None
            elif sku_detected != sku_expected:
                s["scan_status"] = "bad"
                s["scan_msg"] = f"Leído: {sku_detected}"
                s["confirmed"] = False
                s["confirm_mode"] = None
            else:
                s["scan_status"] = "ok"
                s["scan_msg"] = "Producto correcto."
                s["confirmed"] = True
                s["confirm_mode"] = "SCAN"
                s["scan_value"] = scan
            st.rerun()

    with col3:
        if st.button("Sin EAN"):
            s["show_manual_confirm"] = True
            st.rerun()

    with col4:
        if st.button("Surtido en venta"):
            # Siempre manda este SKU al final de la fila (rotación circular).
            # Implementación: defer_rank = (máximo defer_rank en esta OT) + 1
            try:
                c.execute("SELECT COALESCE(MAX(defer_rank), 0) FROM picking_tasks WHERE ot_id=?", (ot_id,))
                max_rank = c.fetchone()[0] or 0
                new_rank = int(max_rank) + 1
                c.execute(
                    "UPDATE picking_tasks SET defer_rank=?, defer_at=? WHERE id=?",
                    (new_rank, now_iso(), task_id)
                )
                conn.commit()
            except Exception:
                pass
            # Limpiar estado UI de este task y seguir con el siguiente
            state.pop(str(task_id), None)
            st.rerun()

    if s.get("show_manual_confirm", False) and not s["confirmed"]:
        st.info("Confirmación manual")
        st.write(f"✅ {producto_show}")
        if st.button("Confirmar", key=f"confirm_manual_{task_id}"):
            s["confirmed"] = True
            s["confirm_mode"] = "MANUAL_NO_EAN"
            s["show_manual_confirm"] = False
            s["scan_status"] = "ok"
            s["scan_msg"] = "Confirmado manual."
            st.rerun()

    qty_label = "Cantidad"
    qty_in = st.text_input(
        qty_label,
        value=s["qty_input"],
        disabled=not s["confirmed"],
        key=f"qty_{task_id}"
    )
    force_tel_keyboard(qty_label)

    if st.button("Confirmar cantidad", disabled=not s["confirmed"]):
        try:
            q = int(str(qty_in).strip())
        except Exception:
            st.error("Ingresa un número válido.")
            q = None

        if q is not None:
            s["qty_input"] = str(q)

            if q > int(qty_total):
                st.error(f"La cantidad ({q}) supera solicitado ({qty_total}).")
                s["needs_decision"] = False

            elif q == int(qty_total):
                c.execute("""
                    UPDATE picking_tasks
                    SET qty_picked=?, status='DONE', decided_at=?, confirm_mode=?
                    WHERE id=?
                """, (q, now_iso(), s["confirm_mode"], task_id))
                conn.commit()
                state.pop(str(task_id), None)
                st.success("OK. Siguiente…")
                st.rerun()

            else:
                missing = int(qty_total) - q
                s["needs_decision"] = True
                s["missing"] = missing
                st.warning(f"Faltan {missing}. Debes decidir (incidencias o reintentar).")

    if s["needs_decision"]:
        st.error(f"DECISIÓN: faltan {s['missing']} unidades.")
        colA, colB = st.columns(2)

        with colA:
            if st.button("A incidencias y seguir"):
                q = int(s["qty_input"])
                missing = int(qty_total) - q

                c.execute("""
                    INSERT INTO picking_incidences (ot_id, sku_ml, qty_total, qty_picked, qty_missing, reason, created_at)
                    VALUES (?,?,?,?,?,?,?)
                """, (ot_id, sku_expected, int(qty_total), q, missing, "FALTANTE", now_iso()))

                c.execute("""
                    UPDATE picking_tasks
                    SET qty_picked=?, status='INCIDENCE', decided_at=?, confirm_mode=?
                    WHERE id=?
                """, (q, now_iso(), s["confirm_mode"], task_id))

                conn.commit()
                state.pop(str(task_id), None)
                st.success("Enviado a incidencias. Siguiente…")
                st.rerun()

        with colB:
            if st.button("Reintentar"):
                s["needs_decision"] = False
                st.info("Ajusta cantidad y confirma nuevamente.")

    conn.close()


# =========================
# FULL: Importar Excel -> Batch
# =========================
def _pick_col(cols_lower: list[str], cols_orig: list[str], candidates: list[str]):
    for cand in candidates:
        if cand in cols_lower:
            return cols_orig[cols_lower.index(cand)]
    return None


def _safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s

def _cell_to_str(x) -> str:
    """Convierte celdas que pueden venir como Series (por columnas duplicadas) a string limpio."""
    try:
        # Si por error hay columnas duplicadas, pandas puede entregar Series en vez de escalar
        if isinstance(x, pd.Series):
            for v in x.tolist():
                s = _safe_str(v)
                if s:
                    return s
            return ""
    except Exception:
        pass
    return _safe_str(x)


def read_full_excel(file) -> pd.DataFrame:
    """
    Lee todas las hojas y devuelve un DF normalizado:
    sku_ml, title, qty_required, area, nro, etiquetar, es_pack, instruccion, vence, sheet
    """
    xls = pd.ExcelFile(file)
    all_rows = []
    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh, dtype=str)
        if df is None or df.empty:
            continue

        cols_orig = df.columns.tolist()
        cols_lower = [str(c).strip().lower() for c in cols_orig]

        sku_col = _pick_col(cols_lower, cols_orig, ["sku", "sku_ml", "codigo", "código", "cod", "ubc", "cod sku"])
        qty_col = _pick_col(cols_lower, cols_orig, ["cantidad", "qty", "unidades", "cant", "cant.", "cantidad total"])
        title_col = _pick_col(cols_lower, cols_orig, ["articulo", "artículo", "descripcion", "descripción", "producto", "detalle", "artículo / producto"])

        area_col = _pick_col(cols_lower, cols_orig, ["area", "área", "zona", "ubicacion", "ubicación"])
        nro_col = _pick_col(cols_lower, cols_orig, ["nro", "n°", "numero", "número", "num", "#", "n"])

        etiquetar_col = _pick_col(cols_lower, cols_orig, ["etiquetar", "etiqueta"])
        pack_col = _pick_col(cols_lower, cols_orig, ["es pack", "pack", "es_pack", "espack"])
        instr_col = _pick_col(cols_lower, cols_orig, ["instruccion", "instrucción", "obs", "observacion", "observación", "nota", "notas"])
        vence_col = _pick_col(cols_lower, cols_orig, ["vence", "vencimiento", "fecha vence", "fecha_vencimiento"])

        # Fallback mínimo: si no hay columnas clave, intentar por posición
        if sku_col is None or qty_col is None:
            if df.shape[1] >= 3:
                # intento: col0 area, col1 nro, col2 sku, col3 desc, col4 qty
                sku_col = sku_col or cols_orig[min(2, len(cols_orig) - 1)]
                qty_col = qty_col or cols_orig[min(4, len(cols_orig) - 1)]
                title_col = title_col or cols_orig[min(3, len(cols_orig) - 1)]
                area_col = area_col or cols_orig[0]
                nro_col = nro_col or cols_orig[min(1, len(cols_orig) - 1)]

        for _, r in df.iterrows():
            sku = normalize_sku(r.get(sku_col, "")) if sku_col else ""
            if not sku:
                continue

            qty_raw = r.get(qty_col, "") if qty_col else ""
            try:
                qty = int(float(str(qty_raw).strip())) if str(qty_raw).strip() else 0
            except Exception:
                qty = 0
            if qty <= 0:
                continue

            title = _safe_str(r.get(title_col, "")) if title_col else ""
            area = _safe_str(r.get(area_col, "")) if area_col else ""
            nro = _safe_str(r.get(nro_col, "")) if nro_col else ""
            etiquetar = _safe_str(r.get(etiquetar_col, "")) if etiquetar_col else ""
            es_pack = _safe_str(r.get(pack_col, "")) if pack_col else ""
            instruccion = _safe_str(r.get(instr_col, "")) if instr_col else ""
            vence = _safe_str(r.get(vence_col, "")) if vence_col else ""

            all_rows.append({
                "sheet": sh,
                "sku_ml": sku,
                "title": title,
                "qty_required": qty,
                "area": area,
                "nro": nro,
                "etiquetar": etiquetar,
                "es_pack": es_pack,
                "instruccion": instruccion,
                "vence": vence,
            })

    return pd.DataFrame(all_rows)


def compute_full_status(qty_required: int, qty_checked: int, has_incidence: bool = False) -> str:
    if qty_checked <= 0:
        return "PENDING"
    if qty_checked == qty_required and not has_incidence:
        return "OK"
    if qty_checked == qty_required and has_incidence:
        return "OK_WITH_ISSUES"
    if qty_checked < qty_required and has_incidence:
        return "INCIDENCE"
    if qty_checked < qty_required:
        return "PARTIAL"
    if qty_checked > qty_required:
        return "OVER"
    return "PENDING"


def get_open_full_batches():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, batch_name, status, created_at FROM full_batches WHERE status='OPEN' ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows


def upsert_full_batch_from_df(df: pd.DataFrame, batch_name: str):
    """
    Crea un batch y carga items agregados por SKU.
    """
    if df is None or df.empty:
        raise ValueError("El Excel no tiene filas válidas (SKU/Cantidad).")

    # Agregar por SKU
    agg = {}
    for _, r in df.iterrows():
        sku = normalize_sku(r.get("sku_ml", ""))
        if not sku:
            continue

        qty = int(r.get("qty_required", 0) or 0)
        if qty <= 0:
            continue

        if sku not in agg:
            agg[sku] = {
                "sku_ml": sku,
                "title": _cell_to_str(r.get("title", "")),
                "qty_required": 0,
                "areas": set(),
                "nros": set(),
                "etiquetar": "",
                "es_pack": "",
                "instruccion": "",
                "vence": "",
            }

        a = agg[sku]
        a["qty_required"] += qty

        area = _safe_str(r.get("area", ""))
        nro = _safe_str(r.get("nro", ""))
        if area:
            a["areas"].add(area)
        if nro:
            a["nros"].add(nro)

        # En campos opcionales, guardamos el primero no vacío (si hay)
        for k in ["etiquetar", "es_pack", "instruccion", "vence"]:
            v = _safe_str(r.get(k, ""))
            if v and not a.get(k):
                a[k] = v

        # si no hay título, intentar completar después con maestro (en UI)
        if not a["title"]:
            a["title"] = _cell_to_str(r.get("title", ""))

    conn = get_conn()
    c = conn.cursor()

    created = now_iso()
    c.execute(
        "INSERT INTO full_batches (batch_name, status, created_at, closed_at) VALUES (?,?,?,?)",
        (batch_name, "OPEN", created, None)
    )
    batch_id = c.lastrowid

    for sku, a in agg.items():
        areas_txt = " / ".join(sorted(a["areas"])) if a["areas"] else ""
        nros_txt = " / ".join(sorted(a["nros"])) if a["nros"] else ""
        c.execute("""
            INSERT INTO full_batch_items
            (batch_id, sku_ml, title, areas, nros, etiquetar, es_pack, instruccion, vence, qty_required, qty_checked, status, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            batch_id, sku, a["title"], areas_txt, nros_txt,
            a.get("etiquetar", ""), a.get("es_pack", ""), a.get("instruccion", ""), a.get("vence", ""),
            int(a["qty_required"]), 0, "PENDING", now_iso()
        ))

    conn.commit()
    conn.close()
    return batch_id


def get_full_batch_summary(batch_id: int):
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT batch_name, status, created_at, closed_at FROM full_batches WHERE id=?", (batch_id,))
    b = c.fetchone()

    c.execute("""
        SELECT
            COUNT(*) as n_skus,
            SUM(qty_required) as req_units,
            SUM(qty_checked) as chk_units,
            SUM(CASE WHEN status='OK' THEN 1 ELSE 0 END) as ok_skus,
            SUM(CASE WHEN status IN ('PARTIAL','INCIDENCE','OVER','OK_WITH_ISSUES') THEN 1 ELSE 0 END) as touched_skus,
            SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) as pending_skus
        FROM full_batch_items
        WHERE batch_id=?
    """, (batch_id,))
    s = c.fetchone()

    conn.close()
    return b, s


# =========================
# UI: FULL - CARGA EXCEL
# =========================
def page_full_upload(inv_map_sku: dict):
    st.header("Full – Cargar Excel")

    # Confirmación (mensaje flash)
    if st.session_state.get("full_flash"):
        st.success(st.session_state.get("full_flash"))
        st.session_state["full_flash"] = ""

    # Solo 1 corrida a la vez: si hay lote abierto, no permitir cargar otro
    open_batches = get_open_full_batches()
    if open_batches:
        active_id, active_name, active_status, active_created = open_batches[0]
        st.warning(
            f"Ya hay un lote Full en curso (#{active_id}). "
            "Para cargar uno nuevo, ve a **Full – Admin** y usa **Reiniciar corrida (BORRA TODO)**."
        )
        return

    # Nombre de lote automático (no se muestra)
    batch_name = f"FULL_{datetime.now().strftime('%Y-%m-%d_%H%M')}"

    file = st.file_uploader("Excel de preparación Full (xlsx)", type=["xlsx"], key="full_excel")
    if not file:
        st.info("Sube el Excel que usan para enviar hojas a auxiliares.")
        return

    try:
        df = read_full_excel(file)
    except Exception as e:
        st.error(f"No pude leer el Excel: {e}")
        return

    if df.empty:
        st.warning("El archivo se leyó, pero no encontré filas válidas (SKU/Cantidad).")
        return

    # Completar título desde maestro si está vacío
    df2 = df.copy()
    df2["title_eff"] = df2.apply(lambda r: r["title"] if str(r["title"]).strip() else inv_map_sku.get(r["sku_ml"], ""), axis=1)

    st.subheader("Vista previa (primeras 50 filas)")
    st.dataframe(df2.head(50))

    st.caption("Se agregará por SKU (sumando cantidades de todas las hojas).")

    if st.button("✅ Crear lote y cargar"):
        try:
            # Guardar SOLO un 'title' (evita duplicar columnas y que se muestre como Series)
            df_save = df2.copy()
            if "title_eff" in df_save.columns:
                if "title" in df_save.columns:
                    df_save = df_save.drop(columns=["title"])
                df_save = df_save.rename(columns={"title_eff": "title"})

            batch_id = upsert_full_batch_from_df(df_save, str(batch_name).strip())

            # Mostrar confirmación aunque hagamos rerun
            st.session_state["full_flash"] = f"✅ Lote Full cargado correctamente (#{batch_id})."
            st.session_state.full_selected_batch = batch_id
            st.rerun()
        except Exception as e:
            st.error(str(e))




def page_full_supervisor(inv_map_sku: dict):
    st.header("Full – Supervisor de acopio")

    # Resolver lote activo: debe existir un lote OPEN (solo trabajamos con 1 a la vez)
    open_batches = get_open_full_batches()
    if not open_batches:
        st.info("No hay un lote Full en curso. Ve a **Full – Cargar Excel** para crear la corrida.")
        return

    batch_id, _batch_name, _status, _created_at = open_batches[0]

    # Map barcode->sku desde DB (maestro ya lo cargó)
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT barcode, sku_ml FROM sku_barcodes")
    barcode_to_sku = {r[0]: r[1] for r in c.fetchall()}
    conn.close()

    st.markdown(
        """
        <style>
        .hero2 { padding: 10px 12px; border-radius: 12px; background: rgba(0,0,0,0.04); margin: 8px 0; }
        .hero2 .sku { font-size: 26px; font-weight: 900; margin: 0; }
        .hero2 .prod { font-size: 22px; font-weight: 800; margin: 6px 0 0 0; line-height: 1.15; }
        .hero2 .qty { font-size: 20px; font-weight: 900; margin: 8px 0 0 0; }
        .hero2 .meta { font-size: 14px; font-weight: 700; margin: 6px 0 0 0; opacity: 0.85; line-height: 1.2; }
        .tag { display:inline-block; padding: 6px 10px; border-radius: 10px; font-weight: 900; }
        .ok { background: rgba(0, 200, 0, 0.15); }
        .bad { background: rgba(255, 0, 0, 0.12); }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Estado UI supervisor (por lote)
    if "full_sup_state" not in st.session_state:
        st.session_state.full_sup_state = {}
    state = st.session_state.full_sup_state
    if str(batch_id) not in state:
        state[str(batch_id)] = {
            "sku_current": "",
            "msg": "",
            "msg_kind": "idle",
            "confirm_partial": False,
            "pending_qty": None,
            "scan_nonce": 0,
            "qty_nonce": 0
        }
    sst = state[str(batch_id)]

    scan_key = f"full_scan_{batch_id}_{sst.get('scan_nonce',0)}"
    qty_key  = f"full_qty_{batch_id}_{sst.get('qty_nonce',0)}"

    # Mensaje flash (se muestra una vez)
    flash_key = f"full_flash_{batch_id}"
    if flash_key in st.session_state:
        kind, msg = st.session_state.get(flash_key, ("info", ""))
        if msg:
            if kind == "warning":
                st.warning(msg)
            elif kind == "success":
                st.success(msg)
            else:
                st.info(msg)
        st.session_state.pop(flash_key, None)

    scan_label = "Escaneo"
    scan = st.text_input(scan_label, key=scan_key)
    force_tel_keyboard(scan_label)
    autofocus_input(scan_label)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🔎 Buscar / Validar", key=f"full_find_{batch_id}"):
            sku = resolve_scan_to_sku(scan, barcode_to_sku)
            sst["sku_current"] = sku
            sst["confirm_partial"] = False
            sst["pending_qty"] = None
            try:
                st.session_state[qty_key] = ""
            except Exception:
                pass

            if not sku:
                sst["msg_kind"] = "bad"
                sst["msg"] = "No se pudo leer el código."
                st.rerun()

            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                SELECT 1
                FROM full_batch_items
                WHERE batch_id=? AND sku_ml=?
            """, (batch_id, sku))
            ok = c.fetchone()
            conn.close()

            if not ok:
                sst["msg_kind"] = "bad"
                sst["msg"] = f"{sku} no pertenece a este lote."
                sst["sku_current"] = ""
            else:
                sst["msg_kind"] = "ok"
                sst["msg"] = "SKU encontrado."
            st.rerun()

    with colB:
        if st.button("🧹 Limpiar", key=f"full_clear_{batch_id}"):
            sst["sku_current"] = ""
            sst["msg_kind"] = "idle"
            sst["msg"] = ""
            sst["confirm_partial"] = False
            sst["pending_qty"] = None
            sst["scan_nonce"] = int(sst.get("scan_nonce",0)) + 1
            sst["qty_nonce"]  = int(sst.get("qty_nonce",0)) + 1
            st.rerun()

    if sst.get("msg_kind") == "ok":
        st.markdown(f'<span class="tag ok">✅ OK</span> {sst.get("msg","")}', unsafe_allow_html=True)
    elif sst.get("msg_kind") == "bad":
        st.markdown(f'<span class="tag bad">❌ ERROR</span> {sst.get("msg","")}', unsafe_allow_html=True)

    sku_cur = normalize_sku(sst.get("sku_current", ""))
    if not sku_cur:
        st.info("Escanea un producto para ver datos.")
        return

    # Traer datos del SKU desde el lote
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT sku_ml, COALESCE(NULLIF(title,''),''), qty_required, COALESCE(qty_checked,0), COALESCE(etiquetar,''), COALESCE(es_pack,''), COALESCE(instruccion,''), COALESCE(vence,'')
        FROM full_batch_items
        WHERE batch_id=? AND sku_ml=?
    """, (batch_id, sku_cur))
    row = c.fetchone()
    conn.close()

    if not row:
        st.warning("El SKU no está en el lote (vuelve a validar).")
        return

    sku_db, title_db, qty_req, qty_chk, etiquetar_db, es_pack_db, instruccion_db, vence_db = row
    title_clean = str(title_db or "").strip()
    # Seguridad: si por algún motivo title viene como Series/objeto raro
    if hasattr(title_db, "iloc"):
        try:
            title_clean = str(title_db.iloc[0] or "").strip()
        except Exception:
            title_clean = str(title_db).strip()
    if not title_clean:
        title_clean = inv_map_sku.get(sku_db, "")

    pending = int(qty_req) - int(qty_chk)
    if pending < 0:
        pending = 0

    # Campos extra del Excel Full
    etiquetar_txt = str(etiquetar_db or "").strip() or "-"
    es_pack_txt = str(es_pack_db or "").strip() or "-"
    instruccion_txt = str(instruccion_db or "").strip() or "-"
    vence_txt = str(vence_db or "").strip() or "-"

    st.markdown(
        f"""
        <div class="hero2">
            <div class="sku">SKU: {sku_db}</div>
            <div class="prod">{title_clean}</div>
            <div class="qty">Solicitado: {int(qty_req)} • Acopiado: {int(qty_chk)} • Pendiente: {pending}</div>
            <div class="meta">ETIQUETAR: {etiquetar_txt} • ES PACK: {es_pack_txt}<br/>INSTRUCCIÓN: {instruccion_txt} • VENCE: {vence_txt}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    qty_label = "Cantidad a acopiar"
    qty_in = st.text_input(qty_label, key=qty_key)
    force_tel_keyboard(qty_label)

    def do_acopio(q: int):
        conn2 = get_conn()
        c2 = conn2.cursor()
        c2.execute("""
            UPDATE full_batch_items
            SET qty_checked = COALESCE(qty_checked,0) + ?,
                status = CASE WHEN (COALESCE(qty_checked,0) + ?) >= COALESCE(qty_required,0) THEN 'OK' ELSE 'PENDING' END,
                updated_at = ?
            WHERE batch_id=? AND sku_ml=?
        """, (q, q, now_iso(), batch_id, sku_db))
        conn2.commit()
        conn2.close()

        # Limpiar campos para siguiente escaneo
        sst["sku_current"] = ""
        sst["msg_kind"] = "idle"
        sst["msg"] = ""
        sst["confirm_partial"] = False
        sst["pending_qty"] = None
        sst["scan_nonce"] = int(sst.get("scan_nonce",0)) + 1
        sst["qty_nonce"]  = int(sst.get("qty_nonce",0)) + 1

        st.session_state[flash_key] = ("success", f"✅ Acopio registrado: {q} unidad(es).")
        st.rerun()

    # Si está pendiente confirmación parcial, mostrar confirmación ANTES de acopiar
    if sst.get("confirm_partial") and sst.get("pending_qty") is not None:
        q_pending = int(sst["pending_qty"])
        st.warning(f"Vas a acopiar **{q_pending}** unidad(es), pero el pendiente actual es **{pending}**. ¿Confirmas acopio parcial?")
        colP1, colP2 = st.columns(2)
        with colP1:
            if st.button("✅ Sí, confirmar acopio parcial", key=f"full_confirm_partial_yes_{batch_id}"):
                # Revalidar pendiente para evitar carrera
                if q_pending <= 0:
                    st.error("Cantidad inválida.")
                    return
                if q_pending > pending:
                    st.error(f"No puedes acopiar {q_pending}. Pendiente actual: {pending}.")
                    sst["confirm_partial"] = False
                    sst["pending_qty"] = None
                    return
                do_acopio(q_pending)
        with colP2:
            if st.button("Cancelar", key=f"full_confirm_partial_no_{batch_id}"):
                sst["confirm_partial"] = False
                sst["pending_qty"] = None
                st.session_state[flash_key] = ("info", "Acopio parcial cancelado. Ajusta cantidad y confirma nuevamente.")
                st.rerun()

        # Importante: no mostrar el botón normal mientras espera confirmación
        return

    colC, colD = st.columns([1, 1])
    with colC:
        if st.button("✅ Confirmar acopio", key=f"full_confirm_{batch_id}"):
            try:
                q = int(str(qty_in).strip())
            except Exception:
                st.error("Ingresa un número válido.")
                return

            if q <= 0:
                st.error("La cantidad debe ser mayor a 0.")
                return

            # No permitimos sobrantes: no puede superar el pendiente
            if q > pending:
                st.error(f"No puedes acopiar {q}. Pendiente actual: {pending}.")
                return

            # Si es menor al pendiente, pedir confirmación ANTES de acopiar
            if q < pending:
                sst["confirm_partial"] = True
                sst["pending_qty"] = q
                st.rerun()

            # Si es exacto, acopia directo
            do_acopio(q)

    with colD:
        if st.button("🧹 Limpiar campos", key=f"full_clear2_{batch_id}"):
            sst["sku_current"] = ""
            sst["msg_kind"] = "idle"
            sst["msg"] = ""
            sst["confirm_partial"] = False
            sst["pending_qty"] = None
            sst["scan_nonce"] = int(sst.get("scan_nonce",0)) + 1
            sst["qty_nonce"]  = int(sst.get("qty_nonce",0)) + 1
            st.rerun()


def page_full_admin():
    st.header("Full – Administrador (progreso)")

    batches = get_open_full_batches()
    if not batches:
        st.info("No hay lotes Full cargados aún.")
        return

    options = [f"#{bid} — {name} ({status})" for bid, name, status, _ in batches]
    default_idx = 0
    if "full_selected_batch" in st.session_state:
        for i, (bid, *_rest) in enumerate(batches):
            if bid == st.session_state.full_selected_batch:
                default_idx = i
                break

    sel = st.selectbox("Lote", options, index=default_idx)
    batch_id = batches[options.index(sel)][0]
    st.session_state.full_selected_batch = batch_id

    b, s = get_full_batch_summary(batch_id)
    if not b:
        st.error("No se encontró el lote.")
        return

    batch_name, bstatus, created_at, closed_at = b
    n_skus, req_units, chk_units, ok_skus, touched_skus, pending_skus = s
    n_skus = int(n_skus or 0)
    req_units = int(req_units or 0)
    chk_units = int(chk_units or 0)
    ok_skus = int(ok_skus or 0)
    pending_skus = int(pending_skus or 0)

    prog = (chk_units / req_units) if req_units else 0.0

    st.caption(f"Lote: {batch_name} • Creado: {to_chile_display(created_at)} • Estado: {bstatus}")
    st.progress(min(max(prog, 0.0), 1.0))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Progreso unidades", f"{prog*100:.1f}%")
    c2.metric("Unidades acopiadas", f"{chk_units}/{req_units}")
    c3.metric("SKUs OK", f"{ok_skus}/{n_skus}")
    c4.metric("SKUs pendientes", pending_skus)

    conn = get_conn()
    c = conn.cursor()

    st.subheader("Detalle por SKU")
    c.execute("""
        SELECT sku_ml, COALESCE(NULLIF(title,''),''), qty_required, qty_checked,
               (qty_required - qty_checked) as pendiente,
               status, updated_at, areas, nros
        FROM full_batch_items
        WHERE batch_id=?
        ORDER BY status, CAST(sku_ml AS INTEGER), sku_ml
    """, (batch_id,))
    rows = c.fetchall()
    df = pd.DataFrame(rows, columns=["SKU", "Artículo", "Solicitado", "Acopiado", "Pendiente", "Estado", "Actualizado", "Áreas", "Nros"])
    df["Actualizado"] = df["Actualizado"].apply(to_chile_display)
    st.dataframe(df, use_container_width=True)

    st.subheader("Incidencias")
    c.execute("""
        SELECT sku_ml, qty_required, qty_checked, diff, reason, created_at
        FROM full_incidences
        WHERE batch_id=?
        ORDER BY created_at DESC
    """, (batch_id,))
    inc = c.fetchall()
    if inc:
        df_inc = pd.DataFrame(inc, columns=["SKU", "Req", "Chk", "Diff", "Motivo", "Hora"])
        df_inc["Hora"] = df_inc["Hora"].apply(to_chile_display)
        st.dataframe(df_inc, use_container_width=True)
    else:
        st.info("Sin incidencias registradas para este lote.")

    st.divider()

    st.subheader("Acciones")

    # Reiniciar corrida FULL (borrar todo lo cargado para Full)
    if "full_confirm_reset" not in st.session_state:
        st.session_state.full_confirm_reset = False

    if not st.session_state.full_confirm_reset:
        if st.button("🔄 Reiniciar corrida (BORRA TODO Full)"):
            st.session_state.full_confirm_reset = True
            st.warning("⚠️ Esto borrará TODOS los datos de Full (lote, items y registros de acopio). Confirma abajo.")
            st.rerun()
    else:
        st.error("CONFIRMACIÓN: se borrará TODO lo relacionado a Full.")
        colA, colB = st.columns(2)
        with colA:
            if st.button("✅ Sí, borrar todo y reiniciar Full"):
                conn2 = get_conn()
                c2 = conn2.cursor()
                c2.execute("DELETE FROM full_incidences;")
                c2.execute("DELETE FROM full_batch_items;")
                c2.execute("DELETE FROM full_batches;")
                conn2.commit()
                conn2.close()

                st.session_state.full_confirm_reset = False
                st.session_state.pop("full_selected_batch", None)

                # limpiar estados UI del supervisor
                if "full_supervisor_state" in st.session_state:
                    st.session_state.pop("full_supervisor_state", None)

                st.success("Full reiniciado (todo borrado).")
                st.rerun()
        with colB:
            if st.button("Cancelar"):
                st.session_state.full_confirm_reset = False
                st.info("Reinicio cancelado.")
                st.rerun()

    conn.close()


# =========================
# UI: ADMIN (FLEX)
# =========================
def page_admin():
    st.header("Administrador")
    pwd = st.text_input("Contraseña", type="password")
    if pwd != ADMIN_PASSWORD:
        st.info("Ingresa contraseña para administrar.")
        return

    conn = get_conn()
    c = conn.cursor()

    st.subheader("Resumen")
    c.execute("SELECT COUNT(*) FROM orders")
    n_orders = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM order_items")
    n_items = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM picking_ots")
    n_ots = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM picking_incidences")
    n_inc = c.fetchone()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventas", n_orders)
    col2.metric("Líneas", n_items)
    col3.metric("OTs", n_ots)
    col4.metric("Incidencias", n_inc)

    st.subheader("Estado OTs")
    c.execute("""
        SELECT po.ot_code, pk.name, po.status, po.created_at, po.closed_at,
               SUM(CASE WHEN pt.status='PENDING' THEN 1 ELSE 0 END) as pendientes,
               SUM(CASE WHEN pt.status IN ('DONE','INCIDENCE') THEN 1 ELSE 0 END) as resueltas,
               SUM(CASE WHEN pt.confirm_mode='MANUAL_NO_EAN' THEN 1 ELSE 0 END) as manual_no_ean
        FROM picking_ots po
        JOIN pickers pk ON pk.id = po.picker_id
        LEFT JOIN picking_tasks pt ON pt.ot_id = po.id
        GROUP BY po.ot_code, pk.name, po.status, po.created_at, po.closed_at
        ORDER BY po.ot_code
    """)
    df = pd.DataFrame(c.fetchall(), columns=[
        "OT", "Picker", "Estado", "Creada", "Cerrada",
        "Pendientes", "Resueltas", "Sin EAN"
    ])
    df["Creada"] = df["Creada"].apply(to_chile_display)
    df["Cerrada"] = df["Cerrada"].apply(to_chile_display)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Incidencias")
    c.execute("""
        SELECT po.ot_code, pk.name, pi.sku_ml, pi.qty_total, pi.qty_picked, pi.qty_missing, pi.reason, pi.created_at
        FROM picking_incidences pi
        JOIN picking_ots po ON po.id = pi.ot_id
        JOIN pickers pk ON pk.id = po.picker_id
        ORDER BY pi.created_at DESC
    """)
    inc_rows = c.fetchall()
    if inc_rows:
        df_inc = pd.DataFrame(inc_rows, columns=["OT", "Picker", "SKU", "Solicitado", "Pickeado", "Faltante", "Motivo", "Hora"])
        df_inc["Hora"] = df_inc["Hora"].apply(to_chile_display)
        st.dataframe(df_inc)
    else:
        st.info("Sin incidencias en la corrida actual.")

    st.divider()
    st.subheader("Acciones")

    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if not st.session_state.confirm_reset:
        if st.button("Reiniciar corrida (BORRA TODO)"):
            st.session_state.confirm_reset = True
            st.warning("⚠️ Esto borrará TODA la información (OTs, tareas, incidencias y ventas). Confirma abajo.")
            st.rerun()
    else:
        st.error("CONFIRMACIÓN: se borrarán TODOS los datos del sistema.")
        colA, colB = st.columns(2)
        with colA:
            if st.button("✅ Sí, borrar todo y reiniciar"):
                c.execute("DELETE FROM picking_tasks;")
                c.execute("DELETE FROM picking_incidences;")
                c.execute("DELETE FROM sorting_status;")
                c.execute("DELETE FROM ot_orders;")
                c.execute("DELETE FROM picking_ots;")
                c.execute("DELETE FROM pickers;")
                c.execute("DELETE FROM order_items;")
                c.execute("DELETE FROM orders;")
                conn.commit()
                st.session_state.confirm_reset = False
                st.success("Sistema reiniciado (todo borrado).")
                st.session_state.pop("selected_picker", None)
                st.rerun()
        with colB:
            if st.button("Cancelar"):
                st.session_state.confirm_reset = False
                st.info("Reinicio cancelado.")
                st.rerun()

    conn.close()

# =========================
# SORTING (CAMARERO)
# =========================

def now_iso():
    return datetime.now(CL_TZ).isoformat() if CL_TZ else datetime.now().isoformat()

def get_active_sorting_manifest():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, name, created_at, status FROM sorting_manifests WHERE status='ACTIVE' ORDER BY id DESC LIMIT 1;")
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "name": row[1], "created_at": row[2], "status": row[3]}

def create_sorting_manifest(name: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO sorting_manifests (name, created_at, status) VALUES (?,?, 'ACTIVE');", (name, now_iso()))
    mid = c.lastrowid
    conn.commit()
    conn.close()
    return mid


def ensure_active_manifest(manifest_name: str) -> int:
    """Garantiza que exista un manifiesto ACTIVE en DB para este flujo.
    Si session_state trae un id viejo (por reinicio/refresh), lo reemplaza.
    """
    active = get_active_sorting_manifest()
    if active:
        st.session_state.sorting_manifest_id = active["id"]
        st.session_state.sorting_manifest_name = active["name"]
        return active["id"]

    # No hay ACTIVE en DB: si en sesión hay un id, validamos que exista y esté ACTIVE
    mid = st.session_state.get("sorting_manifest_id")
    if mid:
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT id, status, name FROM sorting_manifests WHERE id=?;", (int(mid),))
        row = c.fetchone()
        conn.close()
        if row and row[1] == "ACTIVE":
            st.session_state.sorting_manifest_name = row[2]
            return int(mid)

    # Crear uno nuevo
    mid = create_sorting_manifest(manifest_name)
    st.session_state.sorting_manifest_id = mid
    st.session_state.sorting_manifest_name = manifest_name
    # Limpieza de estado de carga previo
    st.session_state.pop("sorting_last_zpl_hash", None)
    return mid


def mark_manifest_done(manifest_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE sorting_manifests SET status='DONE' WHERE id=?;", (manifest_id,))
    conn.commit()
    conn.close()

from typing import Optional


def reset_sorting_all(manifest_id: Optional[int] = None):
    """Borra TODO lo relacionado a Sorting.
    Si manifest_id es None, borra todas las tablas sorting_* (recuperación ante estados corruptos).
    """
    conn = get_conn()
    c = conn.cursor()
    try:
        if manifest_id is None:
            c.execute("DELETE FROM sorting_run_items;")
            c.execute("DELETE FROM sorting_runs;")
            c.execute("DELETE FROM sorting_labels;")
            c.execute("DELETE FROM sorting_manifests;")
        else:
            # Primero items (dependen de runs)
            c.execute("""DELETE FROM sorting_run_items
                         WHERE run_id IN (SELECT id FROM sorting_runs WHERE manifest_id=?);""", (manifest_id,))
            c.execute("DELETE FROM sorting_runs WHERE manifest_id=?;", (manifest_id,))
            c.execute("DELETE FROM sorting_labels WHERE manifest_id=?;", (manifest_id,))
            c.execute("DELETE FROM sorting_manifests WHERE id=?;", (manifest_id,))
        conn.commit()
    finally:
        conn.close()


def decode_fh(text: str) -> str:
    # ZPL ^FH uses _HH hex escapes
    def repl(m):
        try:
            return bytes([int(m.group(1), 16)]).decode("latin-1")
        except Exception:
            return m.group(0)
    return re.sub(r"_(..)", repl, text)

def clean_address(text: str) -> str:
    if not text:
        return ""
    t = decode_fh(text)
    # remove JSON objects from QR payloads if present
    t = re.sub(r"\{.*?\}", "", t)
    t = t.replace("->", " ")
    t = re.sub(r"\s+", " ", t).strip()
    # cut off technical tails often present
    t = re.sub(r"\s*\(\s*Liberador.*$", "", t, flags=re.IGNORECASE).strip()
    return t

def parse_zpl_labels(raw: str):
    # Returns dict pack_id -> {shipment_id,buyer,address,raw}
    # and dict shipment_id -> same (for FLEX QR)
    pack_map = {}
    ship_map = {}

    # collect ^FD...^FS fields and decode ^FH content
    fd = re.findall(r"\^FD(.*?)\^FS", raw, flags=re.DOTALL)
    fd = [decode_fh(x.replace("\n"," ").replace("\r"," ").strip()) for x in fd if x]
    joined = " ".join(fd)

    # Split by ^XA/^XZ blocks
    blocks = re.split(r"\^XA", raw)
    for b in blocks:
        if "^XZ" not in b:
            continue
        # shipment id from barcode
        ship = None
        m = re.search(r"\^FD>:\s*(\d{6,20})", b)
        if m:
            ship = m.group(1)
        # shipment id from QR JSON
        if not ship:
            m = re.search(r'"id"\s*:\s*"(\d{6,20})"', b)
            if m:
                ship = m.group(1)

        # pack id (may be split across fields)
        pack = None
        # try "Pack ID:" with digits/spaces following
        dec_b = decode_fh(b.replace("\n"," ").replace("\r"," "))
        m = re.search(r"Pack ID:\s*([0-9 ]{6,30})", dec_b)
        if m:
            pack = re.sub(r"\s+", "", m.group(1))
        # fallback: if we see a 17-18 digit starting with 20000
        if not pack:
            m = re.search(r"\b(20000\d{7,20})\b", dec_b)
            if m:
                pack = m.group(1)

        # buyer and address heuristics
        buyer = None
        addr = None
        # buyer often appears after ' - ' near end
        m = re.search(r"\b([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)?)\s*\(", dec_b)
        if m:
            buyer = m.group(1).strip()
        # domicile/address text
        m = re.search(r"Domicilio:\s*([^\^]+?)(?:Ciudad de destino:|\^FS|$)", dec_b, flags=re.IGNORECASE)
        if m:
            addr = clean_address(m.group(1))
        else:
            # try line that contains comuna / ciudad
            m = re.search(r"(?:\bComuna\b|\bCiudad\b|\bRM\b).{10,200}", dec_b)
            if m:
                addr = clean_address(m.group(0))

        rec = {"pack_id": pack, "shipment_id": ship, "buyer": buyer, "address": addr, "raw": b}
        if pack:
            pack_map[pack] = rec
        if ship:
            ship_map[ship] = rec

    return pack_map, ship_map

def parse_control_pdf_by_page(pdf_file):
    if not HAS_PDF_LIB:
        st.error("Falta pdfplumber en el entorno.")
        return None
    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [x.strip() for x in text.split("\n") if x.strip()]
            items = []
            seq = 0
            current_pack = None
            current_order = None
            current_buyer = None
            for ln in lines:
                # detect pack/order
                m = re.search(r"Pack ID:\s*(\d+)", ln)
                if m:
                    current_pack = m.group(1)
                    continue
                m = re.search(r"Venta:\s*(\d+)", ln)
                if m:
                    current_order = m.group(1)
                    continue
                m = re.search(r"Nombre\s+Cantidad:\s*(.+)$", ln)
                if m:
                    current_buyer = m.group(1).strip()
                    continue
                # SKU + qty line sometimes combined
                msku = re.search(r"SKU:\s*([0-9A-Za-z]+)", ln)
                mqty = re.search(r"Cantidad:\s*(\d+)", ln)
                if msku and mqty:
                    sku = msku.group(1)
                    qty = int(mqty.group(1))
                    seq += 1
                    items.append({
                        "seq": seq,
                        "ml_order_id": current_order,
                        "pack_id": current_pack,
                        "sku": sku,
                        "qty": qty,
                        "title_ml": "",
                        "buyer": current_buyer
                    })
                    continue
                # title line heuristics: if line looks like product title and last item has empty title
                if items and not items[-1]["title_ml"]:
                    # avoid attribute lines like "Acabado: Mate"
                    if ":" in ln and not re.search(r"\bMELI\b", ln, re.IGNORECASE):
                        # treat as attribute not title
                        pass
                    else:
                        items[-1]["title_ml"] = ln[:200]
            pages.append({"page_no": pno, "items": items})
    return pages

def upsert_labels_to_db(manifest_id: int, pack_map: dict, raw: str):
    conn = get_conn()
    c = conn.cursor()
    for pack_id, rec in pack_map.items():
        c.execute(
            """INSERT INTO sorting_labels (manifest_id, pack_id, shipment_id, buyer, address, raw)
                 VALUES (?,?,?,?,?,?)
                 ON CONFLICT(manifest_id, pack_id) DO UPDATE SET
                    shipment_id=excluded.shipment_id,
                    buyer=excluded.buyer,
                    address=excluded.address,
                    raw=excluded.raw;""",
            (manifest_id, pack_id, rec.get("shipment_id"), rec.get("buyer"), rec.get("address"), raw)
        )
    conn.commit()
    conn.close()


def apply_labels_to_existing_items(manifest_id: int):
    """Propaga shipment_id/buyer/address desde sorting_labels hacia sorting_run_items ya creados.
    Útil cuando se cargan etiquetas DESPUÉS de crear corridas.
    """
    conn = get_conn()
    c = conn.cursor()
    # Solo actualizar campos vacíos o nulos para no pisar datos ya confirmados/manuales
    c.execute(
        """UPDATE sorting_run_items
               SET shipment_id = COALESCE(shipment_id, (SELECT l.shipment_id FROM sorting_labels l
                                                      WHERE l.manifest_id=? AND l.pack_id=sorting_run_items.pack_id)),
                   buyer       = CASE WHEN (buyer IS NULL OR buyer='') THEN (SELECT l.buyer FROM sorting_labels l
                                                      WHERE l.manifest_id=? AND l.pack_id=sorting_run_items.pack_id) ELSE buyer END,
                   address     = CASE WHEN (address IS NULL OR address='') THEN (SELECT l.address FROM sorting_labels l
                                                      WHERE l.manifest_id=? AND l.pack_id=sorting_run_items.pack_id) ELSE address END
             WHERE run_id IN (SELECT id FROM sorting_runs WHERE manifest_id=?);""",
        (manifest_id, manifest_id, manifest_id)
    )
    conn.commit()
    conn.close()


def create_runs_and_items(manifest_id: int, assignments: dict, pages: list, inv_map_sku: dict, barcode_to_sku: dict):
    # assignments: page_no -> mesa
    conn = get_conn()
    c = conn.cursor()
    # load labels for this manifest
    c.execute("SELECT pack_id, shipment_id, buyer, address FROM sorting_labels WHERE manifest_id=?;", (manifest_id,))
    label_rows = c.fetchall()
    labels = {r[0]: {"shipment_id": r[1], "buyer": r[2], "address": r[3]} for r in label_rows}

    for page in pages:
        pno = page["page_no"]
        mesa = assignments.get(pno)
        if not mesa:
            continue
        c.execute(
            """INSERT INTO sorting_runs (manifest_id, page_no, mesa, status, created_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(manifest_id, page_no) DO UPDATE SET
                   mesa=excluded.mesa,
                   status='PENDING',
                   closed_at=NULL;""",
            (manifest_id, pno, int(mesa), "PENDING", now_iso())
        )
        c.execute("SELECT id FROM sorting_runs WHERE manifest_id=? AND page_no=?;", (manifest_id, pno))
        run_id = c.fetchone()[0]
        # clear previous items if re-created
        c.execute("DELETE FROM sorting_run_items WHERE run_id=?;", (run_id,))
        for it in page["items"]:
            sku = str(it.get("sku") or "").strip()
            title_ml = (it.get("title_ml") or "").strip()
            # translate using maestro
            title_tec = inv_map_sku.get(sku, "") if inv_map_sku else ""
            buyer = it.get("buyer") or ""
            pack_id = it.get("pack_id") or ""
            ship = labels.get(pack_id, {}).get("shipment_id") if pack_id else None
            addr = labels.get(pack_id, {}).get("address") if pack_id else None
            buyer2 = labels.get(pack_id, {}).get("buyer") if pack_id else None
            if buyer2 and not buyer:
                buyer = buyer2
            c.execute(
                """INSERT INTO sorting_run_items
                    (run_id, seq, ml_order_id, pack_id, sku, title_ml, title_tec, qty, buyer, address, shipment_id, status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?, 'PENDING');""",
                (run_id, it["seq"], it.get("ml_order_id"), pack_id, sku, title_ml, title_tec, int(it.get("qty") or 1),
                 buyer, addr, ship)
            )
    conn.commit()
    conn.close()

def page_sorting_upload(inv_map_sku: dict, barcode_to_sku: dict):
    st.header("🧾 Sorting pedidos Flex y Colecta")
    st.caption("Carga 1 manifiesto (Control PDF) y asigna TODAS las páginas a mesas. 1 página = 1 mesa.")

    active = get_active_sorting_manifest()

    # Siempre permitir cargar etiquetas y/o continuar asignación para el manifiesto ACTIVO.
    if active:
        st.info(f"Manifiesto ACTIVO: **{active['name']}** (creado: {to_chile_display(active['created_at'])})")
        st.warning("Este manifiesto sigue en proceso. Puedes cargar etiquetas y/o terminar la asignación de páginas. "
                   "No se permite crear un manifiesto nuevo hasta finalizar éste.")
        mid = active["id"]
        st.session_state.sorting_manifest_id = mid
        # Para continuar (re)asignación, pedimos el PDF del manifiesto activo (para volver a parsear páginas si es necesario)
        pdf = st.file_uploader("Control / Manifiesto ACTIVO (PDF)", type=["pdf"], key="sorting_pdf_active")
        zpl = st.file_uploader("Etiquetas de envío (TXT/ZPL) (puedes cargarlo ahora)", type=["txt"], key="sorting_zpl_active")

        # Permite cargar etiquetas en cualquier momento (sin bucles por rerun)
        if zpl is not None:
            raw_bytes = zpl.getvalue()
            zpl_hash = hashlib.md5(raw_bytes).hexdigest()
            if st.session_state.get("sorting_last_zpl_hash") != zpl_hash:
                raw = raw_bytes.decode("utf-8", errors="ignore")
                pack_map, ship_map = parse_zpl_labels(raw)
                upsert_labels_to_db(mid, pack_map, raw)
                apply_labels_to_existing_items(mid)
                st.info(f"Etiquetas detectadas: {len(pack_map)} con Pack ID / {len(ship_map)} con envío (QR/barra).")
                apply_labels_to_existing_items(mid)
                st.info(f"Etiquetas detectadas: {len(pack_map)} con Pack ID / {len(ship_map)} con envío (QR/barra).")
                st.session_state["sorting_last_zpl_hash"] = zpl_hash
                st.success("Etiquetas cargadas/actualizadas para el manifiesto activo.")
            else:
                st.info("Etiquetas ya procesadas (sin cambios).")

        # Si ya existen corridas creadas, solo mostramos resumen y enviamos a Camarero
        conn = get_conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(1) FROM sorting_runs WHERE manifest_id=?;", (mid,))
        run_count = int(c.fetchone()[0] or 0)
        conn.close()

        if run_count > 0:
            st.success(f"Corridas ya creadas para este manifiesto: **{run_count}**. Ve a 'Camarero' para completarlas.")
            st.stop()

        # Si aún no hay corridas, necesitamos páginas del PDF para asignarlas.
        # Si ya las tenemos en sesión (por una carga previa), NO obligamos a re-subir el PDF.
        if not pdf:
            pages = st.session_state.get("sorting_parsed_pages")
            if not pages:
                st.info("Sube el PDF del manifiesto activo para asignar sus páginas a mesas.")
                st.stop()
        else:
            pages = parse_control_pdf_by_page(pdf)
        if not pages:
            st.error("No se pudo leer el PDF.")
            st.stop()

        manifest_name = getattr(pdf, "name", active["name"]) or active["name"]
        st.session_state.sorting_parsed_pages = pages
        st.session_state.sorting_manifest_name = manifest_name

    else:
        pdf = st.file_uploader("Control / Manifiesto (PDF)", type=["pdf"], key="sorting_pdf")
        zpl = st.file_uploader("Etiquetas de envío (TXT/ZPL) (opcional pero recomendado)", type=["txt"], key="sorting_zpl")

        if not pdf:
            st.info("Sube el PDF para continuar.")
            return

        pages = parse_control_pdf_by_page(pdf)
        if not pages:
            st.error("No se pudo leer el PDF.")
            return

        manifest_name = getattr(pdf, "name", "manifiesto.pdf")
        if "sorting_parsed_pages" not in st.session_state or st.session_state.get("sorting_manifest_name") != manifest_name:
            st.session_state.sorting_parsed_pages = pages
            st.session_state.sorting_manifest_name = manifest_name

        # create manifest row now (robusto: no depender solo de session_state)
        mid = ensure_active_manifest(manifest_name)
        # store labels if uploaded (sin bucles)
        if zpl is not None:
            raw_bytes = zpl.getvalue()
            zpl_hash = hashlib.md5(raw_bytes).hexdigest()
            if st.session_state.get("sorting_last_zpl_hash") != zpl_hash:
                raw = raw_bytes.decode("utf-8", errors="ignore")
                pack_map, ship_map = parse_zpl_labels(raw)
                upsert_labels_to_db(mid, pack_map, raw)
                apply_labels_to_existing_items(mid)
                st.info(f"Etiquetas detectadas: {len(pack_map)} con Pack ID / {len(ship_map)} con envío (QR/barra).")
                apply_labels_to_existing_items(mid)
                st.info(f"Etiquetas detectadas: {len(pack_map)} con Pack ID / {len(ship_map)} con envío (QR/barra).")
                st.session_state["sorting_last_zpl_hash"] = zpl_hash
    mid = st.session_state.sorting_manifest_id
    pages = st.session_state.sorting_parsed_pages

    st.subheader("Asignación de páginas a mesas (obligatorio)")
    st.write(f"Páginas detectadas: **{len(pages)}**. Debes asignar TODAS para continuar.")
    # assignment UI
    assignments = st.session_state.get("sorting_assignments", {})
    used_pages = set(assignments.keys())
    cols = st.columns(3)
    with cols[0]:
        mesa_sel = st.selectbox("Mesa", list(range(1, NUM_MESAS+1)), key="mesa_sel")
    with cols[1]:
        unassigned = [p["page_no"] for p in pages if p["page_no"] not in used_pages]
        page_sel = st.selectbox("Página sin asignar", unassigned if unassigned else [p["page_no"] for p in pages], key="page_sel")
    with cols[2]:
        if st.button("➕ Asignar"):
            assignments[int(page_sel)] = int(mesa_sel)
            st.session_state.sorting_assignments = assignments
            st.rerun()

    # show table
    df_assign = pd.DataFrame([{"Página": p["page_no"], "Mesa": assignments.get(p["page_no"], "")} for p in pages])
    st.dataframe(df_assign, use_container_width=True, hide_index=True)

    missing = [p["page_no"] for p in pages if p["page_no"] not in assignments]
    if missing:
        st.error(f"Faltan páginas por asignar: {missing}")
        st.stop()

    if st.button("✅ Crear corridas"):
        try:
            create_runs_and_items(mid, assignments, pages, inv_map_sku, barcode_to_sku)
            # Verificación rápida
            conn = get_conn()
            c = conn.cursor()
            c.execute("SELECT COUNT(1) FROM sorting_runs WHERE manifest_id=?;", (mid,))
            rc = int(c.fetchone()[0] or 0)
            conn.close()
            if rc <= 0:
                st.error("No se crearon corridas (0). Revisa que el PDF tenga páginas parseadas y que estén asignadas a mesas.")
            else:
                st.success(f"Corridas creadas: **{rc}**. Ve a 'Camarero'.")
                st.rerun()
        except Exception as e:
            st.exception(e)

def get_next_run_for_mesa(mesa: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """SELECT r.id, r.page_no, r.status, m.name
             FROM sorting_runs r
             JOIN sorting_manifests m ON m.id=r.manifest_id
             WHERE r.mesa=? AND r.status!='DONE'
             ORDER BY r.page_no ASC, r.id ASC
             LIMIT 1;""",
        (int(mesa),)
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"run_id": row[0], "page_no": row[1], "status": row[2], "manifest_name": row[3]}

def get_next_group(run_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """SELECT ml_order_id, pack_id, MIN(seq) as mseq
             FROM sorting_run_items
             WHERE run_id=? AND status!='DONE'
             GROUP BY ml_order_id, pack_id
             ORDER BY mseq ASC
             LIMIT 1;""",
        (run_id,)
    )
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    order_id, pack_id, _ = row
    c.execute(
        """SELECT id, sku, title_ml, title_tec, qty, buyer, address, shipment_id, status
             FROM sorting_run_items
             WHERE run_id=? AND ml_order_id=? AND pack_id=?
             ORDER BY seq ASC;""",
        (run_id, order_id, pack_id)
    )
    items = []
    for r in c.fetchall():
        items.append({
            "id": r[0],
            "sku": r[1],
            "title_ml": r[2] or "",
            "title_tec": r[3] or "",
            "qty": r[4] or 1,
            "buyer": r[5] or "",
            "address": r[6] or "",
            "shipment_id": r[7] or "",
            "status": r[8] or "PENDING",
        })
    conn.close()
    return {"ml_order_id": order_id, "pack_id": pack_id, "items": items}

def mark_item_done(item_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE sorting_run_items SET status='DONE', done_at=? WHERE id=?;", (now_iso(), int(item_id)))
    conn.commit()
    conn.close()

def mark_item_incidence(item_id: int, note: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE sorting_run_items SET status='INCIDENCE', incidence_note=?, done_at=? WHERE id=?;", (note, now_iso(), int(item_id)))
    conn.commit()
    conn.close()

def maybe_close_run(run_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(1) FROM sorting_run_items WHERE run_id=? AND status!='DONE';", (run_id,))
    remaining = c.fetchone()[0]
    if remaining == 0:
        c.execute("UPDATE sorting_runs SET status='DONE', closed_at=? WHERE id=?;", (now_iso(), run_id))
        conn.commit()
    conn.close()

def maybe_close_manifest_if_done():
    active = get_active_sorting_manifest()
    if not active:
        return
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(1) FROM sorting_runs WHERE manifest_id=? AND status!='DONE';", (active["id"],))
    rem = c.fetchone()[0]
    conn.close()
    if rem == 0:
        mark_manifest_done(active["id"])
        # clear session state
        for k in ["sorting_manifest_id","sorting_parsed_pages","sorting_manifest_name","sorting_assignments"]:
            st.session_state.pop(k, None)


# =========================
# SORTING - HELPERS EXTRA
# =========================
def extract_shipment_id_from_scan(raw: str) -> str:
    """Recibe lo que devuelve el lector (QR JSON Flex, número de barra Colecta, o texto mixto)
    y devuelve el shipment_id (solo dígitos)."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    # Caso QR JSON (Flex / otros)
    if s.lstrip().startswith("{"):
        try:
            import json as _json
            obj = _json.loads(s)
            _id = obj.get("id")
            if _id:
                m = re.search(r"\d{8,15}", str(_id))
                return m.group(0) if m else ""
        except Exception:
            pass
    # Caso mixto o solo números: tomamos el primer grupo largo
    m = re.search(r"\d{10,15}", s)
    if m:
        return m.group(0)
    # Fallback para algunos lectores que separan con comas y dejan números de 9-10
    m2 = re.search(r"\d{8,15}", s)
    return m2.group(0) if m2 else ""


def page_sorting_camarero(inv_map_sku: dict, barcode_to_sku: dict):
    st.header("👷 Camarero")

    active = get_active_sorting_manifest()
    if not active:
        st.warning("No hay manifiesto activo. Ve a Sorting → Carga y crea corridas.")
        return

    mesa = st.selectbox("Selecciona tu mesa", list(range(1, NUM_MESAS+1)), key="camarero_mesa")

    # Estado UI: venta abierta por etiqueta (shipment_id)
    if "camarero_active_shipment" not in st.session_state:
        st.session_state.camarero_active_shipment = ""
    if "camarero_active_ml_order" not in st.session_state:
        st.session_state.camarero_active_ml_order = ""
    if "camarero_active_run_id" not in st.session_state:
        st.session_state.camarero_active_run_id = None

    def _reset_to_scan_label():
        st.session_state.camarero_active_shipment = ""
        st.session_state.camarero_active_ml_order = ""
        st.session_state.camarero_active_run_id = None
        st.session_state["camarero_label_scan"] = ""
        st.session_state["camarero_sku_scan"] = ""

    # ---------- PANTALLA A: escanear etiqueta ----------
    if not st.session_state.camarero_active_shipment:
        st.subheader("Escanea la etiqueta (QR Flex o barra Colecta)")
        raw_label = st.text_input("Escanea etiqueta aquí", key="camarero_label_scan", placeholder="Ej: QR Flex (JSON) o número 4638...")

        colA, colB = st.columns([1, 2])
        with colA:
            open_btn = st.button("Abrir venta", use_container_width=True)
        with colB:
            st.caption("Tip: escanea **lo más grande** de la etiqueta (Flex: QR | Colecta: barra).")

        if open_btn or (raw_label and raw_label.endswith("\n")):
            shipment_id = extract_shipment_id_from_scan(raw_label)
            if not shipment_id:
                st.error("No pude leer el ID de envío desde el escaneo. Prueba escanear el código de barras grande o el QR.")
                return

            conn = get_conn()
            c = conn.cursor()

            # Buscamos una venta PENDIENTE en esta mesa para este shipment_id dentro del manifiesto activo.
            c.execute(
                """SELECT i.run_id, i.ml_order_id, i.pack_id
                   FROM sorting_run_items i
                   JOIN sorting_runs r ON r.id = i.run_id
                   WHERE r.manifest_id=? AND r.mesa=? AND i.shipment_id=? AND i.status='PENDING'
                   ORDER BY r.page_no ASC, i.seq ASC
                   LIMIT 1;""",
                (active["id"], int(mesa), str(shipment_id))
            )
            row = c.fetchone()

            # Si no hay pendientes, vemos si existe pero ya está cerrada (DONE/INCIDENCE)
            if not row:
                c.execute(
                    """SELECT i.run_id, i.ml_order_id, i.pack_id,
                              SUM(CASE WHEN i.status='PENDING' THEN 1 ELSE 0 END) as pend
                       FROM sorting_run_items i
                       JOIN sorting_runs r ON r.id = i.run_id
                       WHERE r.manifest_id=? AND r.mesa=? AND i.shipment_id=?
                       GROUP BY i.run_id, i.ml_order_id, i.pack_id
                       ORDER BY r.page_no ASC, MIN(i.seq) ASC
                       LIMIT 1;""",
                    (active["id"], int(mesa), str(shipment_id))
                )
                row2 = c.fetchone()
                conn.close()
                if not row2:
                    st.error(f"No encontré esta etiqueta (envío {shipment_id}) en las corridas de la mesa {mesa}.")
                    st.info("Revisa que asignaste la página correcta a esta mesa o que el manifiesto corresponde al día.")
                    return
                else:
                    st.warning(f"El envío {shipment_id} ya no tiene productos pendientes en esta mesa (ya cerrado o con incidencias).")
                    _reset_to_scan_label()
                    st.rerun()

            conn.close()
            run_id, ml_order_id, pack_id = row[0], row[1], row[2]
            st.session_state.camarero_active_shipment = str(shipment_id)
            st.session_state.camarero_active_ml_order = str(ml_order_id or "")
            st.session_state.camarero_active_run_id = int(run_id) if run_id is not None else None
            st.rerun()

        # También mostramos si hay corridas pendientes en mesa (para guiar al operador)
        run = get_next_run_for_mesa(mesa)
        if not run:
            st.success("No hay corridas pendientes para esta mesa.")
            maybe_close_manifest_if_done()
        else:
            st.info(f"Hay corridas pendientes en esta mesa. Escanea una etiqueta para abrir la venta.")
        return

    # ---------- PANTALLA B: venta abierta ----------
    shipment_id = st.session_state.camarero_active_shipment

    # Traemos items de esa venta (por shipment + mesa + manifiesto activo)
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """SELECT i.id, i.run_id, r.page_no, r.mesa, i.ml_order_id, i.pack_id, i.sku, i.title_ml, i.title_tec, i.qty,
                  i.buyer, i.address, i.shipment_id, i.status
           FROM sorting_run_items i
           JOIN sorting_runs r ON r.id=i.run_id
           WHERE r.manifest_id=? AND r.mesa=? AND i.shipment_id=?
           ORDER BY r.page_no ASC, i.seq ASC;""",
        (active["id"], int(mesa), str(shipment_id))
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        st.error("No encontré items para esta etiqueta en esta mesa (quizá cambiaste de mesa).")
        _reset_to_scan_label()
        return

    # Header compacto
    # Tomamos venta desde el primer item
    first = rows[0]
    page_no = first[2]
    ml_order_id = first[4] or "-"
    pack_id = first[5] or "-"
    buyer = first[10] or "-"
    address = first[11] or "-"

    total_items = len(rows)
    pending_items = [r for r in rows if r[13] == "PENDING"]
    done_count = total_items - len(pending_items)

    st.markdown(f"**Mesa {mesa} · Página {page_no} · Venta {ml_order_id} · Envío {shipment_id} · {done_count}/{total_items}**")

    # Botón para volver (si se equivocó de etiqueta)
    colh1, colh2 = st.columns([1, 1])
    with colh1:
        if st.button("⬅ Cambiar etiqueta", use_container_width=True):
            _reset_to_scan_label()
            st.rerun()
    with colh2:
        st.caption("Cuando termines todo, usa **Cerrar venta**.")

    # Producto actual (primer pendiente)
    current = pending_items[0] if pending_items else None

    st.divider()
    st.subheader("Productos de esta venta/etiqueta")

    # Lista compacta de productos
    for r in rows:
        _id, _run_id, _page_no, _mesa, _ml, _pack, sku, t_ml, t_tec, qty, *_rest, _ship, status = r
        dot = "🟠" if status == "PENDING" else ("🟢" if status == "DONE" else "🔴")
        title_show = t_tec or t_ml or ""
        st.write(f"{dot} **{title_show}**  · SKU: {sku} · Qty: {qty} · **{status}**")

    st.divider()

    if not current:
        st.success("✅ Todos los productos de esta etiqueta están procesados.")
        if st.button("✅ Cerrar venta", type="primary", use_container_width=True):
            # No cambia estados (ya están DONE/INCIDENCE). Solo vuelve a escanear etiqueta.
            _reset_to_scan_label()
            st.rerun()
        return

    # Producto actual (en grande)
    item_id = current[0]
    sku_expected = str(current[6] or "").strip()
    title_ml = current[7] or ""
    title_tec = current[8] or ""
    qty_need = int(current[9] or 1)

    title_show = title_tec or title_ml or sku_expected

    st.markdown(f"### {title_show}")
    st.markdown(f"**SKU:** {sku_expected}  ·  **Requiere:** {qty_need} unidad(es)")

    # Input de SKU/EAN
    scan = st.text_input("Escanea SKU/EAN aquí", key="camarero_sku_scan", placeholder="Escanea el producto…")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        validar = st.button("Validar", use_container_width=True)
    with col2:
        manual = st.button("Confirmar manual", use_container_width=True)
    with col3:
        incidencia = st.button("Incidencia (faltante)", use_container_width=True)

    def _mark_done(_item_id: int):
        conn = get_conn()
        c = conn.cursor()
        c.execute("UPDATE sorting_run_items SET status='DONE', done_at=? WHERE id=?;", (now_iso(), int(_item_id)))
        conn.commit()
        conn.close()

    def _mark_incid(_item_id: int, note: str):
        conn = get_conn()
        c = conn.cursor()
        c.execute("UPDATE sorting_run_items SET status='INCIDENCE', done_at=?, incidence_note=? WHERE id=?;",
                  (now_iso(), note or "faltante", int(_item_id)))
        conn.commit()
        conn.close()

    if validar:
        scanned = (scan or "").strip()
        if not scanned:
            st.warning("Escanea un código primero.")
        else:
            # Normalizamos: si el escaneo es EAN, lo pasamos a SKU si existe mapeo
            scanned_norm = scanned
            if scanned_norm in barcode_to_sku:
                scanned_norm = str(barcode_to_sku.get(scanned_norm) or scanned_norm).strip()

            if scanned_norm == sku_expected:
                _mark_done(item_id)
                st.session_state["camarero_sku_scan"] = ""
                st.rerun()
            else:
                st.error(f"Código no coincide. Esperado SKU {sku_expected}.")
                st.session_state["camarero_sku_scan"] = ""

    if manual:
        _mark_done(item_id)
        st.session_state["camarero_sku_scan"] = ""
        st.rerun()

    if incidencia:
        _mark_incid(item_id, "faltante")
        st.session_state["camarero_sku_scan"] = ""
        st.rerun()
def page_sorting_admin():
    st.header("📊 Admin")

    # Reiniciar corrida (BORRA TODO) - SOLO Sorting
    st.markdown("### 🧹 Reiniciar corrida")
    st.caption("Borra el manifiesto activo y todas sus corridas/items/etiquetas. Úsalo solo si necesitas partir de cero.")
    active = get_active_sorting_manifest()

    # Confirmación en 2 pasos para evitar clic accidental
    if st.checkbox("Entiendo que esto borrará TODO el Sorting actual", key="sorting_reset_ack"):
        if st.button("🗑️ Reiniciar corrida (BORRA TODO)", type="primary"):
            try:
                reset_sorting_all(active["id"] if active else None)
            except Exception as e:
                st.error(f"No pude reiniciar: {e}")
                return

            # Limpieza de estado en sesión (Sorting + Camarero)
            for k in [
                "sorting_assignments",
                "sorting_last_zpl_hash",
                "sorting_manifest_name",
                "sorting_parsed_pages",
                "camarero_label_scan",
                "camarero_sku_scan",
            ]:
                st.session_state.pop(k, None)

            st.success("Sorting reiniciado. Puedes cargar un manifiesto nuevo.")
            st.rerun()

    if not active:
        st.info("No hay manifiesto activo.")
        return
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """SELECT page_no, mesa, status,
                   (SELECT SUM(CASE WHEN status='DONE' THEN 1 ELSE 0 END) FROM sorting_run_items i WHERE i.run_id=r.id) as done,
                   (SELECT COUNT(1) FROM sorting_run_items i WHERE i.run_id=r.id) as total
             FROM sorting_runs r
             WHERE manifest_id=?
             ORDER BY page_no ASC;""",
        (active["id"],)
    )
    rows = c.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["Página","Mesa","Estado","OK","Total"])
    st.dataframe(df, use_container_width=True, hide_index=True)

# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title="Aurora ML – WMS", layout="wide")
    init_db()

    # Auto-carga maestro desde repo (sirve para ambos modos)
    inv_map_sku, barcode_to_sku, conflicts = master_bootstrap(MASTER_FILE)

    # Si no hay modo seleccionado, mostramos lobby y salimos
    if "app_mode" not in st.session_state:
        page_app_lobby()
        return

    # Sidebar común
    st.sidebar.title("Ferretería Aurora – WMS")

    # Botón para volver al lobby
    if st.sidebar.button("⬅️ Cambiar modo"):
        st.session_state.pop("app_mode", None)
        st.session_state.pop("selected_picker", None)
        st.session_state.pop("full_selected_batch", None)
        st.rerun()

    # Estado maestro (lo dejamos en sidebar, bajo el título)
    if os.path.exists(MASTER_FILE):
        st.sidebar.success(f"Maestro OK: {len(inv_map_sku)} SKUs / {len(barcode_to_sku)} EAN")
        if conflicts:
            st.sidebar.warning(f"Conflictos EAN: {len(conflicts)} (se usa el primero)")
    else:
        st.sidebar.warning(f"No se encontró {MASTER_FILE}. (La app funciona, pero sin maestro)")

    mode = st.session_state.get("app_mode", "FLEX_PICK")

    # ==========
    # MODO FLEX / COLECTA (lo actual)
    # ==========
    if mode == "FLEX_PICK":
        pages = [
            "1) Picking",
            "2) Importar ventas",
            "3) Administrador",
        ]
        page = st.sidebar.radio("Menú", pages, index=0)

        if page.startswith("1"):
            page_picking()
        elif page.startswith("2"):
            page_import(inv_map_sku)
        else:
            page_admin()


    # ==========
    # MODO SORTING (Camarero)
    # ==========
    elif mode == "SORTING":
        pages = [
            "1) Cargar manifiesto y asignar mesas",
            "2) Camarero",
            "3) Admin",
        ]
        page = st.sidebar.radio("Menú", pages, index=0)

        if page.startswith("1"):
            page_sorting_upload(inv_map_sku, barcode_to_sku)
        elif page.startswith("2"):
            page_sorting_camarero(inv_map_sku, barcode_to_sku)
        else:
            page_sorting_admin()

    # ==========
    # MODO FULL (nuevo módulo completo)
    # ==========
    else:
        pages = [
            "1) Cargar Excel Full",
            "2) Supervisor de acopio",
            "3) Admin Full (progreso)",
        ]
        page = st.sidebar.radio("Menú", pages, index=0)

        if page.startswith("1"):
            page_full_upload(inv_map_sku)
        elif page.startswith("2"):
            page_full_supervisor(inv_map_sku)
        else:
            page_full_admin()


if __name__ == "__main__":
    main()
