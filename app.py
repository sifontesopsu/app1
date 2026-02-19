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
# SFX (sonidos retro)
# =========================
# Reproduce un sonido corto (tipo "moneda" arcade) cuando una acción es OK,
# y sonidos distintos para DUPLICADO / EXTRA / ERROR. No bloquea el beep del PDA.
def _sfx_trigger(kind: str):
    try:
        st.session_state["_sfx_nonce"] = int(st.session_state.get("_sfx_nonce", 0)) + 1
        st.session_state["_sfx_kind"] = str(kind or "").upper()
    except Exception:
        pass

def _sfx_render():
    kind = str(st.session_state.get("_sfx_kind", "") or "").upper()
    nonce = int(st.session_state.get("_sfx_nonce", 0) or 0)
    if not kind or nonce <= 0:
        return

    # WebAudio (sin archivos externos)
    js = f"""
    <script>
    (function(){{
      const kind = {kind!r};
      try {{
        const AC = window.AudioContext || window.webkitAudioContext;
        const ctx = new AC();
        const now = ctx.currentTime;

        function tone(freq, t0, dur, type, gain){{
          const o = ctx.createOscillator();
          const g = ctx.createGain();
          o.type = type || 'square';
          o.frequency.setValueAtTime(freq, t0);
          g.gain.setValueAtTime(0.0001, t0);
          g.gain.exponentialRampToValueAtTime(gain || 0.12, t0 + 0.01);
          g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
          o.connect(g); g.connect(ctx.destination);
          o.start(t0); o.stop(t0 + dur + 0.02);
        }}

        // "Moneda" tipo arcade (arpegio corto)
        function coin(){{
          tone(988, now+0.00, 0.07, 'square', 0.14);
          tone(1319, now+0.08, 0.07, 'square', 0.13);
          tone(1760, now+0.16, 0.06, 'square', 0.12);
        }}

        // Duplicado: "rebote" (dos tonos descendentes)
        function dup(){{
          tone(660, now+0.00, 0.08, 'sawtooth', 0.10);
          tone(440, now+0.10, 0.10, 'sawtooth', 0.10);
        }}

        // Extra: "power-up" suave
        function extra(){{
          tone(523, now+0.00, 0.06, 'triangle', 0.10);
          tone(659, now+0.07, 0.06, 'triangle', 0.10);
          tone(784, now+0.14, 0.08, 'triangle', 0.10);
        }}

        // Error: buzzer corto
        function err(){{
          tone(220, now+0.00, 0.18, 'square', 0.12);
        }}

        if (kind === 'OK') coin();
        else if (kind === 'DUP' || kind === 'DUPLICADO') dup();
        else if (kind === 'EXTRA') extra();
        else err();
        setTimeout(()=>{{ try{{ctx.close();}}catch(e){{}} }}, 600);
      }} catch(e) {{}}
    }})();
    </script>
    """
    components.html(js, height=0, key=f"_sfx_{nonce}")


# =========================
# CONFIG
# =========================
DB_NAME = "aurora_ml.db"
ADMIN_PASSWORD = "aurora123"  # cambia si quieres
NUM_MESAS = 4


# =========================
# TABLAS POR MÓDULO (para respaldo parcial)
# =========================
PICKING_TABLES = [
    "orders",
    "order_items",
    "pickers",
    "picking_ots",
    "picking_tasks",
    "picking_incidences",
    "cortes_tasks",
    "ot_orders",
    "sorting_status",
]
FULL_TABLES = [
    "full_batches","full_batch_items","full_incidences"
]
SORTING_TABLES = [
    # Sorting v1
    "sorting_manifests","sorting_runs","sorting_run_items","sorting_labels",
    # Sorting v2 (control + etiquetas + corridas)
    "s2_manifests","s2_files","s2_page_assign","s2_sales","s2_items","s2_labels","s2_pack_ship"
]

# Maestro SKU/EAN en la misma carpeta que app.py
MASTER_FILE = "maestro_sku_ean.xlsx"



# Maestro de SKUs para CORTES (rollos / corte manual)
CORTES_FILE = "CORTES.xlsx"
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
    """ISO timestamp in Chile time (America/Santiago) with UTC offset."""
    if CL_TZ is not None:
        return datetime.now(CL_TZ).isoformat(timespec="seconds")
    return datetime.now().isoformat(timespec="seconds")



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
    """Muestra timestamps en hora Chile.

    - Si el ISO trae zona/offset, se convierte a America/Santiago.
    - Si es naive (sin zona), se muestra tal cual (asumido ya en hora Chile).
    """
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso_str))
        if CL_TZ is None:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        if dt.tzinfo is not None:
            dt = dt.astimezone(CL_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
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


# =========================
# BACKUP/RESTORE POR MÓDULO (SQLite parcial)
# =========================
def _db_table_exists(conn, table: str) -> bool:
    try:
        row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (table,)).fetchone()
        return bool(row)
    except Exception:
        return False

def _export_tables_to_db_bytes(tables: list[str]) -> bytes:
    """Exporta SOLO las tablas indicadas a un .db (bytes). No toca el DB actual."""
    import tempfile
    conn_src = get_conn()
    csrc = conn_src.cursor()
    # Crear DB temporal
    fd, tmp_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn_out = sqlite3.connect(tmp_path, check_same_thread=False)
    cout = conn_out.cursor()
    try:
        for tname in tables:
            if not _db_table_exists(conn_src, tname):
                continue
            row = csrc.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (tname,)).fetchone()
            create_sql = row[0] if row and row[0] else None
            if not create_sql:
                continue
            cout.execute(create_sql)
            rows = csrc.execute(f"SELECT * FROM {tname};").fetchall()
            if rows:
                ncols = len(rows[0])
                ph = ",".join(["?"] * ncols)
                cout.executemany(f"INSERT INTO {tname} VALUES ({ph});", rows)
        conn_out.commit()
        conn_out.close()
        conn_src.close()
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        try:
            conn_out.close()
        except Exception:
            pass
        try:
            conn_src.close()
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def _restore_tables_from_db_bytes(db_bytes: bytes, tables: list[str]) -> tuple[bool, str|None]:
    """Restaura SOLO las tablas indicadas desde un .db (bytes). Mantiene el resto intacto."""
    import tempfile
    # Guardar uploaded db a temp
    fd, up_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    with open(up_path, "wb") as f:
        f.write(db_bytes)

    conn_src = sqlite3.connect(up_path, check_same_thread=False)
    csrc = conn_src.cursor()
    conn_dst = get_conn()
    cdst = conn_dst.cursor()

    try:
        # Validación mínima: que exista al menos 1 de las tablas esperadas
        any_ok = False
        for tname in tables:
            if _db_table_exists(conn_src, tname):
                any_ok = True
                break
        if not any_ok:
            return False, "El respaldo no contiene las tablas esperadas para este módulo."

        # Transacción de reemplazo parcial
        cdst.execute("BEGIN;")
        for tname in tables:
            if not _db_table_exists(conn_src, tname):
                continue

            # Leer schema desde respaldo
            row = csrc.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (tname,)).fetchone()
            create_sql = row[0] if row and row[0] else None
            if not create_sql:
                continue

            # Reemplazar tabla
            cdst.execute(f"DROP TABLE IF EXISTS {tname};")
            cdst.execute(create_sql)

            # Copiar filas
            rows = csrc.execute(f"SELECT * FROM {tname};").fetchall()
            if rows:
                ncols = len(rows[0])
                ph = ",".join(["?"] * ncols)
                cdst.executemany(f"INSERT INTO {tname} VALUES ({ph});", rows)

        conn_dst.commit()
        return True, None
    except Exception as e:
        try:
            conn_dst.rollback()
        except Exception:
            pass
        return False, str(e)
    finally:
        try:
            conn_src.close()
        except Exception:
            pass
        try:
            conn_dst.close()
        except Exception:
            pass
        try:
            os.remove(up_path)
        except Exception:
            pass

def _render_module_backup_ui(scope_key: str, scope_label: str, tables: list[str]):
    """UI para respaldar/restaurar SOLO un módulo (tablas específicas)."""
    with st.expander(f"💾 Respaldo / Restauración — {scope_label}", expanded=False):
        st.caption(
            "Este respaldo es SOLO de este módulo (tablas específicas). "
            "No toca datos de otros módulos. "
            "Nota: el mapa común de códigos (sku_barcodes) no se incluye aquí."
        )
        # Password gate sólo para acciones críticas
        pwd2 = st.text_input("Contraseña admin", type="password", key=f"pwd_{scope_key}")
        if pwd2 != ADMIN_PASSWORD:
            st.info("Ingresa la contraseña para habilitar respaldo/restauración.")
            return

        # Backup
        try:
            data = _export_tables_to_db_bytes(tables)
            st.download_button(
                f"⬇️ Descargar respaldo ({scope_key}.db)",
                data=data,
                file_name=f"aurora_{scope_key}.db",
                mime="application/octet-stream",
                use_container_width=True,
                key=f"dl_{scope_key}",
            )
        except Exception as e:
            st.warning(f"No se pudo preparar el respaldo: {e}")

        st.divider()

        up = st.file_uploader(
            f"⬆️ Restaurar respaldo de {scope_label} (.db)",
            type=["db"],
            key=f"up_{scope_key}",
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            confirm = st.text_input("Escribe RESTAURAR para confirmar", value="", key=f"cf_{scope_key}")
        with col2:
            do = st.button(
                "♻️ Restaurar",
                type="primary",
                disabled=not (up and confirm.strip().upper() == "RESTAURAR"),
                key=f"do_{scope_key}",
            )
        if do and up is not None:
            ok, err = _restore_tables_from_db_bytes(up.getvalue(), tables)
            if ok:
                st.success("✅ Restaurado. Recargando…")
                st.rerun()
            else:
                st.error(f"No se pudo restaurar: {err}")





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
        note TEXT,
        created_at TEXT
    );
    """)

    # --- CORTES (rollos / corte manual) ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS cortes_tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ot_id INTEGER,
        sku_ml TEXT,
        title_ml TEXT,
        title_tec TEXT,
        qty_total INTEGER,
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

    # --- CONTADOR DE PAQUETES (Flex/Colecta) ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS pkg_counter_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,               -- FLEX / COLECTA
    status TEXT DEFAULT 'OPEN',
    created_at TEXT,
    closed_at TEXT
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS pkg_counter_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    label_key TEXT,
    raw TEXT,
    scanned_at TEXT,
    UNIQUE(run_id, label_key)
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
    _ensure_col("picking_incidences", "note", "TEXT")

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
    raw = str(scan or "").strip()
    digits = only_digits(raw)
    sku = ""
    if digits and digits in barcode_to_sku:
        sku = barcode_to_sku[digits]
    else:
        sku = normalize_sku(raw)

    # SFX: solo si realmente se intentó escanear algo
    if raw:
        _sfx_trigger("OK" if sku else "ERROR")
    return sku


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
# CORTES (lista de SKUs)
# =========================
def load_cortes_set(path: str = CORTES_FILE) -> set:
    """Carga listado de SKUs que requieren corte manual desde Excel (defensivo)."""
    # Cache en session_state para evitar leer el Excel en cada rerun
    try:
        ss = st.session_state
        if ss.get("_cortes_cache_path") == path and ss.get("_cortes_cache_skus") is not None:
            return set(ss.get("_cortes_cache_skus") or [])
    except Exception:
        pass

    try:
        if not path or not os.path.exists(path):
            return set()
        df = pd.read_excel(path, dtype=str)
    except Exception:
        return set()

    try:
        cols = {str(c).strip().upper(): c for c in df.columns}
        col_sku = cols.get("SKU") or cols.get("SKUS") or cols.get("CODIGO") or cols.get("CÓDIGO")
        if not col_sku:
            col_sku = df.columns[0]

        skus = set()
        for v in df[col_sku].fillna("").tolist():
            s = normalize_sku(v)
            if s:
                skus.add(s)

        try:
            st.session_state["_cortes_cache_path"] = path
            st.session_state["_cortes_cache_skus"] = list(skus)
        except Exception:
            pass

        return skus
    except Exception:
        return set()
# =========================
# PARSER PDF MANIFIESTO
# =========================

def parse_manifest_pdf(uploaded_file) -> pd.DataFrame:
    """
    Parser robusto para Manifiesto PDF (etiquetas).

    Cubre casos reales de ML donde el PDF puede traer, en cualquier orden:
      - "Venta: <id> SKU:<sku>" en el mismo renglón
      - "Pack ID: ... SKU:<sku>" en un renglón y luego "Venta: <id> Cantidad: <n>" en el siguiente
      - "SKU:<sku>" en un renglón y "Cantidad:<n>" en el siguiente
      - Varias ocurrencias de SKU/Cantidad dentro de un mismo renglón

    Regla: cada vez que se detecta una Cantidad, si existe un SKU "vigente" y una Venta vigente,
    se crea un registro (línea) para esa venta+sku.
    """
    if not HAS_PDF_LIB:
        raise RuntimeError("Falta pdfplumber. Agrega 'pdfplumber' a requirements.txt")

    records: list[dict] = []

    re_venta = re.compile(r"\bVenta\s*[:#]?\s*([0-9]+)\b", re.IGNORECASE)
    re_sku = re.compile(r"\bSKU\s*[:#]?\s*([0-9A-Za-z.\-]+)\b", re.IGNORECASE)
    re_qty = re.compile(r"\bCantidad\s*[:#]?\s*([0-9]+)\b", re.IGNORECASE)

    def _is_noise_line(s: str) -> bool:
        low = (s or "").strip().lower()
        if not low:
            return True
        bad = [
            "código carrier", "codigo carrier", "firma carrier",
            "fecha y hora", "despacha tus productos", "identifi",
        ]
        if any(b in low for b in bad):
            return True
        if re.fullmatch(r"[0-9 .:/\-]+", low):
            return True
        return False

    def _maybe_buyer(line: str) -> str:
        # Quitamos cosas típicas que se pegan al nombre (ej: "Diámetro de la cupla: ...")
        # sin ser demasiado agresivos.
        cut_tokens = ["diámetro", "diametro", "color:", "acabado:", "pack id", "sku", "cantidad", "venta:"]
        low = line.lower()
        for tok in cut_tokens:
            idx = low.find(tok)
            if idx > 0:
                return line[:idx].strip()
        return line.strip()

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "").replace("\r", "\n")
            lines = [ln.strip() for ln in text.splitlines() if ln and str(ln).strip()]

            current_order: str | None = None
            current_buyer: str = ""

            # SKU "vigente" para el próximo "Cantidad"
            sku_current: str | None = None

            # SKU visto antes de que aparezca la Venta (caso: "Pack ID ... SKU:xxxx" y luego "Venta ... Cantidad ...")
            pending_sku_before_order: str | None = None

            for line in lines:
                if _is_noise_line(line):
                    continue

                # Capturar Venta (no reseteamos SKU aquí; hay PDFs donde el SKU viene en la línea anterior)
                mv = re_venta.search(line)
                if mv:
                    current_order = mv.group(1).strip()
                    current_buyer = ""
                    # Si hay un SKU pendiente (visto antes de la venta), lo activamos
                    if pending_sku_before_order and not sku_current:
                        sku_current = pending_sku_before_order
                        pending_sku_before_order = None

                # Buyer: primera línea razonable después de "Venta:" que no sea metadata
                if current_order and not current_buyer:
                    low = line.lower()
                    if (not _is_noise_line(line)
                        and "venta" not in low
                        and "sku" not in low
                        and "cantidad" not in low
                        and ":" not in line  # evita "Color:" etc
                        and len(line) <= 120):
                        cand = _maybe_buyer(line)
                        if cand and len(cand) >= 3:
                            current_buyer = cand

                # Tokenizar SKU y Cantidad en orden de aparición en el renglón
                tokens = []
                for ms in re_sku.finditer(line):
                    tokens.append((ms.start(), "SKU", normalize_sku(ms.group(1))))
                for mq in re_qty.finditer(line):
                    try:
                        qv = int(mq.group(1))
                    except Exception:
                        qv = 0
                    tokens.append((mq.start(), "QTY", qv))
                tokens.sort(key=lambda x: x[0])

                for _, kind, val in tokens:
                    if kind == "SKU":
                        if current_order:
                            sku_current = val
                        else:
                            pending_sku_before_order = val
                    elif kind == "QTY":
                        qty = int(val) if val is not None else 0
                        if current_order and sku_current and qty > 0:
                            records.append(
                                {
                                    "ml_order_id": str(current_order).strip(),
                                    "buyer": str(current_buyer or "").strip(),
                                    "sku_ml": str(sku_current).strip(),
                                    "title_ml": "",
                                    "qty": qty,
                                }
                            )
                            # Importante: NO limpiamos sku_current aquí, porque puede venir otra Cantidad asociada
                            # al mismo SKU en el mismo bloque (raro, pero seguro).

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


    # SKUs que se van a CORTES (no aparecen en picking)
    cortes_set = load_cortes_set()

    # Reset corrida (no borra histórico; eso lo hace admin reset total)
    c.execute("DELETE FROM picking_tasks;")
    c.execute("DELETE FROM picking_incidences;")
    c.execute("DELETE FROM cortes_tasks;")
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
            if sku in cortes_set:
                c.execute(
                    "INSERT INTO cortes_tasks (ot_id, sku_ml, title_ml, title_tec, qty_total, created_at) VALUES (?,?,?,?,?,?)",
                    (ot_id, sku, title, title_tec_any, int(total), now_iso())
                )
            else:
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

    st.markdown('<div class="lobbybtn">', unsafe_allow_html=True)
    if st.button("🧮 Contador de paquetes", key="mode_pkg_counter"):
        st.session_state.app_mode = "PKG_COUNT"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Escanea etiquetas y cuenta paquetes; evita duplicados.")
def page_import(inv_map_sku: dict):
    st.header("Importar ventas")
    # Bloqueo duro: no permitir cargar otra tanda si hay una en curso
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(1) FROM picking_ots WHERE status='OPEN'")
        open_ots = int(c.fetchone()[0] or 0)
        c.execute("SELECT COUNT(1) FROM picking_tasks WHERE status='PENDING'")
        pending_tasks = int(c.fetchone()[0] or 0)
    except Exception:
        open_ots, pending_tasks = 0, 0
    conn.close()

    if open_ots > 0 or pending_tasks > 0:
        st.warning("⚠️ Ya hay una tanda de Picking en curso. Para cargar otra, ve a **Administrador** y reinicia/borra la tanda actual.")
        return

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
# UI: CORTES (PDF de la tanda)
# =========================
def page_cortes_pdf_batch():
    st.header("Cortes de la tanda (PDF)")
    st.caption("Lista de productos que requieren corte manual (rollos). No aparecen en el picking PDA.")

    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT po.ot_code,
               ct.sku_ml,
               COALESCE(NULLIF(ct.title_tec,''), ct.title_ml) AS title,
               ct.qty_total
        FROM cortes_tasks ct
        JOIN picking_ots po ON po.id = ct.ot_id
        ORDER BY po.ot_code, CAST(ct.sku_ml AS INTEGER), ct.sku_ml
    """)
    rows = c.fetchall()
    conn.close()

    if not rows:
        st.info("No hay SKUs de corte en la tanda actual.")
        return

    df_raw = pd.DataFrame(rows, columns=["OT", "SKU", "Producto", "Cantidad"])

    # Consolidar por SKU (mismo producto) sumando cantidades
    df = (
        df_raw.groupby(["SKU", "Producto"], as_index=False)
        .agg(Cantidad=("Cantidad", "sum"), OTs=("OT", lambda s: sorted(set(map(str, s)))))
    )
    df["OTs"] = df["OTs"].apply(lambda xs: ", ".join(xs))
    # Orden por SKU numérico si aplica
    try:
        df["_sku_num"] = pd.to_numeric(df["SKU"], errors="coerce")
        df = df.sort_values(["_sku_num", "SKU"]).drop(columns=["_sku_num"])
    except Exception:
        df = df.sort_values(["SKU"])

    st.dataframe(df[["SKU", "Producto", "Cantidad"]], use_container_width=True, hide_index=True)

    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    import textwrap

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    y = h - 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Ferretería Aurora - Cortes")
    y -= 18
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Generado: {to_chile_display(now_iso())}")
    y -= 22

    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, y, "SKU")
    pdf.drawString(140, y, "Producto")
    pdf.drawString(520, y, "Cant.")
    y -= 14

    pdf.setFont("Helvetica", 10)
    for _, r in df.iterrows():
        if y < 60:
            pdf.showPage()
            y = h - 40
            pdf.setFont("Helvetica-Bold", 14)
            pdf.drawString(40, y, "Ferretería Aurora - Cortes")
            y -= 18
            pdf.setFont("Helvetica", 10)
            pdf.drawString(40, y, f"Generado: {to_chile_display(now_iso())}")
            y -= 22
            pdf.setFont("Helvetica-Bold", 10)
            pdf.drawString(40, y, "SKU")
            pdf.drawString(140, y, "Producto")
            pdf.drawString(460, y, "OTs")
            pdf.drawString(540, y, "Cant.")
            y -= 14
            pdf.setFont("Helvetica", 10)

        sku = str(r["SKU"])
        title_full = str(r["Producto"])
        qty = str(int(r["Cantidad"]))

        # Envolver título en 2 líneas máximo para que no se corte ni se mezcle con la cantidad
        wrap_width = 62  # aprox. caracteres para la columna de Producto en A4
        lines = textwrap.wrap(title_full, width=wrap_width)[:2]
        if not lines:
            lines = [""]

        # Línea 1: SKU + Producto + Cantidad
        pdf.drawString(40, y, sku)
        pdf.drawString(140, y, lines[0])
        pdf.drawRightString(565, y, qty)
        y -= 12

        # Línea 2 (si aplica): continuación del producto (sin tocar cantidad)
        if len(lines) > 1:
            pdf.drawString(140, y, lines[1])
            y -= 12

    pdf.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()

    st.download_button(
        "⬇️ Descargar PDF de Cortes (tanda)",
        data=pdf_bytes,
        file_name=f"cortes_tanda_{now_iso().replace(':','-')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


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

        # Autofocus en PDA: después de elegir desde la lista, dejar listo el campo de escaneo
        if st.session_state.get("focus_scan", False):
            components.html(
                "<script>"
                "setTimeout(function(){"
                "const el=document.querySelector('input[type=\"text\"]');"
                "if(el){el.focus(); if(el.select){el.select();}}"
                "}, 50);"
                "</script>",
                height=0,
            )
            st.session_state["focus_scan"] = False
        force_tel_keyboard(scan_label)
        # Autofocus inteligente:
        # - Si ya validó el producto (confirmed), llevar el foco a "Cantidad"
        # - Si no, mantener foco en "Escaneo"
        if s.get("confirmed", False):
            autofocus_input("Cantidad")
        else:
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
        if st.button("Siguiente"):
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
                # Si el picker usó "Sin EAN", lo registramos en incidencias para trazabilidad
                if str(s.get("confirm_mode") or "") == "MANUAL_NO_EAN":
                    try:
                        c.execute("""INSERT INTO picking_incidences
                                     (ot_id, sku_ml, qty_total, qty_picked, qty_missing, reason, note, created_at)
                                     VALUES (?,?,?,?,?,?,?,?)""",
                                  (ot_id, sku_expected, int(qty_total), int(q), 0, "SIN_EAN", "", now_iso()))
                    except Exception:
                        pass

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
            # Incidencia con nota (igual que Sorting): pedir motivo antes de guardar
            if "pick_inc_pending" not in st.session_state:
                st.session_state["pick_inc_pending"] = None

            pending = st.session_state.get("pick_inc_pending")
            is_pending = bool(pending and pending.get("task_id") == task_id)

            if (not is_pending) and st.button("A incidencias y seguir"):
                st.session_state["pick_inc_pending"] = {"task_id": task_id}
                st.rerun()

            if is_pending:
                st.warning("Incidencia: escribe el motivo antes de guardar.")
                note_val = st.text_area("Motivo / Nota", key=f"pick_inc_note_{task_id}", height=90,
                                        placeholder="Ej: Falta producto, no se encontró en ubicación, etc.")
                c1, c2 = st.columns([1, 1])
                if c1.button("💾 Guardar incidencia", key=f"pick_inc_save_{task_id}"):
                    q = int(s["qty_input"])
                    missing = int(qty_total) - q

                    c.execute("""INSERT INTO picking_incidences
                                 (ot_id, sku_ml, qty_total, qty_picked, qty_missing, reason, note, created_at)
                                 VALUES (?,?,?,?,?,?,?,?)""",
                              (ot_id, sku_expected, int(qty_total), q, missing, "FALTANTE", note_val or "", now_iso()))

                    c.execute("""UPDATE picking_tasks
                                 SET qty_picked=?, status='INCIDENCE', decided_at=?, confirm_mode=?
                                 WHERE id=?""",
                              (q, now_iso(), s["confirm_mode"], task_id))

                    conn.commit()
                    st.session_state["pick_inc_pending"] = None
                    state.pop(str(task_id), None)
                    st.success("Enviado a incidencias. Siguiente…")
                    st.rerun()

                if c2.button("Cancelar", key=f"pick_inc_cancel_{task_id}"):
                    st.session_state["pick_inc_pending"] = None
                    st.rerun()


        with colB:
            if st.button("Reintentar"):
                s["needs_decision"] = False
                st.info("Ajusta cantidad y confirma nuevamente.")
    # =========================
    
    # =========================
    # LISTA DE SKUS DE ESTA OT
    # =========================
    st.markdown("---")


    force_close_key = f"pick_force_close_list_{ot_id}"
    if force_close_key not in st.session_state:
        st.session_state[force_close_key] = False

    label_list = "📋 Lista de SKUs de esta OT" + ("\u200b" if st.session_state.get(force_close_key, False) else "")
    with st.expander(label_list, expanded=False):

        # Forzar cierre en la próxima recarga (especial PDA)
        st.session_state[force_close_key] = False

        st.caption("Toca un SKU pendiente para ponerlo como el próximo a escanear. Luego sigues normal.")

        # Pendientes primero
        ordered = sorted(
            tasks,
            key=lambda t: (0 if t[6] == "PENDING" else 1, str(t[1]))
        )

        for t in ordered:
            _tid, _sku, _title_ml, _title_tec, _qty_total, _qty_picked, _status = t

            raw_master_t = master_raw_title_lookup(MASTER_FILE, _sku)
            _title_show = raw_master_t if raw_master_t else (
                _title_tec if _title_tec not in (None, "") else (_title_ml or "")
            )

            disabled = (_status != "PENDING") or (_tid == task_id)
            label = f"{_title_show} [{_sku}]"

            if st.button(label, disabled=disabled, key=f"jump_{ot_id}_{_tid}"):

                try:
                    c.execute(
                        "SELECT COALESCE(MIN(defer_rank), 0) FROM picking_tasks WHERE ot_id=? AND status='PENDING'",
                        (ot_id,)
                    )
                    min_rank = c.fetchone()[0] or 0
                    new_rank = int(min_rank) - 1

                    c.execute(
                        "UPDATE picking_tasks SET defer_rank=?, defer_at=? WHERE id=?",
                        (new_rank, now_iso(), _tid)
                    )
                    conn.commit()

                except Exception:
                    pass

                if "pick_state" in st.session_state:
                    st.session_state.pick_state.pop(str(task_id), None)
                    st.session_state.pick_state.pop(str(_tid), None)

                st.session_state[force_close_key] = True
                st.session_state['focus_scan'] = True
                st.rerun()

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

    if st.session_state.get("scroll_to_scan", False):
        components.html(
            "<script>const el=document.getElementById('scan_top'); if(el){el.scrollIntoView({behavior:'smooth', block:'start'});}</script>",
            height=0,
        )
        st.session_state["scroll_to_scan"] = False

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
    batch_name = f"FULL_{(datetime.now(CL_TZ) if CL_TZ else datetime.now()).strftime('%Y-%m-%d_%H%M')}"

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

    # Respaldo/Restauración SOLO FULL (no afecta otros módulos)
    _render_module_backup_ui("full", "Full", FULL_TABLES)


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
        # Producto (nombre técnico): usar maestro si existe, si no SKU
        if isinstance(inv_map_sku, dict) and not df_inc.empty:
            def _pname(sku):
                k = str(sku).strip()
                return inv_map_sku.get(k) or master_raw_title_lookup(MASTER_FILE, k) or k
            df_inc["Producto"] = df_inc["SKU"].apply(_pname)
        else:
            df_inc["Producto"] = df_inc["SKU"].astype(str)
        df_inc = df_inc[["OT","Picker","SKU","Producto","Solicitado","Pickeado","Faltante","Motivo","Nota","Hora"]]
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


    # =========================
    # PERSISTENCIA (Streamlit Community Cloud)
    # =========================
    st.subheader("Persistencia / Respaldo — PICKING")
    _render_module_backup_ui("picking", "Picking", PICKING_TABLES)

    st.divider()

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
        SELECT po.ot_code, pk.name, pi.sku_ml, pi.qty_total, pi.qty_picked, pi.qty_missing, pi.reason, pi.note, pi.created_at
        FROM picking_incidences pi
        JOIN picking_ots po ON po.id = pi.ot_id
        JOIN pickers pk ON pk.id = po.picker_id
        ORDER BY pi.created_at DESC
    """)
    inc_rows = c.fetchall()
    if inc_rows:
        df_inc = pd.DataFrame(inc_rows, columns=["OT","Picker","SKU","Solicitado","Pickeado","Faltante","Motivo","Nota","Hora"])
        # Producto (título técnico): maestro si existe; si no, SKU
        try:
            df_inc["Producto"] = df_inc["SKU"].apply(lambda x: (master_raw_title_lookup(MASTER_FILE, str(x).strip()) or str(x).strip()))
        except Exception:
            df_inc["Producto"] = df_inc["SKU"].astype(str)

        df_inc["Hora"] = df_inc["Hora"].apply(to_chile_display)
        # Orden de columnas más útil
        try:
            df_inc = df_inc[["OT","Picker","SKU","Producto","Solicitado","Pickeado","Faltante","Motivo","Nota","Hora"]]
        except Exception:
            pass
        st.dataframe(df_inc, use_container_width=True, hide_index=True)
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

def mark_manifest_done(manifest_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE sorting_manifests SET status='DONE' WHERE id=?;", (manifest_id,))
    conn.commit()
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
    """Parsea Control.pdf (Flex/Colecta) por página.

    Soporta 2 formatos principales:
    - **Colecta / Identificación Productos**: bloques con Pack ID / Venta / (Comprador) / SKU / Cantidad (puede traer múltiples SKU por venta)
    - **Flex (y a veces Colecta)**: líneas con envío (shipment id) y luego Venta/Pack/SKU/Cantidad.

    Devuelve:
      [{"page_no": int, "items": [ {shipment_id, ml_order_id, pack_id, sku, qty, title_ml, buyer} ... ]}]
    """
    if not HAS_PDF_LIB:
        st.error("Falta pdfplumber en el entorno.")
        return None

    def looks_like_name(s: str) -> bool:
        s = (s or "").strip()
        if not s:
            return False
        if len(s) > 60:
            return False
        # Evitar líneas tipo "Color: Blanco", etc.
        if re.search(r"\b(color|acabado|modelo|di[aá]metro|voltaje|dise[nñ]o|tipo)\b\s*:", s, flags=re.I):
            return False
        # Debe tener letras
        return bool(re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", s))

    pages = []
    with pdfplumber.open(pdf_file) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]

            items = []

            # Contexto (se mantiene mientras cambian SKU/Cantidad)
            ctx = {
                "shipment_id": "",
                "ml_order_id": None,
                "pack_id": None,
                "buyer": "",
                "title_ml": "",
            }

            current_sku = None
            current_title = ""

            def push_item(sku, qty):
                if not ctx.get("ml_order_id") or not sku or not qty:
                    return
                try:
                    q = int(qty)
                except Exception:
                    return
                items.append({
                    "shipment_id": ctx.get("shipment_id", "") or "",
                    "ml_order_id": str(ctx.get("ml_order_id")),
                    "pack_id": (str(ctx.get("pack_id")) if ctx.get("pack_id") else None),
                    "sku": str(sku),
                    "qty": q,
                    "title_ml": (ctx.get("title_ml") or current_title or "")[:200],
                    "buyer": (ctx.get("buyer") or "")[:120],
                })

            # Heurística: en algunos PDFs vienen títulos al final; guardamos el último "título largo" como fallback.
            for ln in lines:
                # 1) shipment id (Flex) dentro de una línea tipo "4638.... <texto>"
                m_ship = re.match(r"^(\d{8,15})\s+(.+)$", ln)
                if m_ship and not ln.lower().startswith("venta"):
                    ctx["shipment_id"] = m_ship.group(1)
                    title = m_ship.group(2).strip()
                    if title and len(title) >= 8:
                        ctx["title_ml"] = title[:200]
                    continue

                # 2) Pack ID / Venta
                m_pack = re.search(r"\bPack\s*ID:\s*([0-9]{10,20})\b", ln, flags=re.I)
                if m_pack:
                    ctx["pack_id"] = m_pack.group(1)
                    # a veces trae SKU en la misma línea
                    m_pack_sku = re.search(r"\bSKU:\s*([0-9A-Za-z_-]+)\b", ln, flags=re.I)
                    if m_pack_sku:
                        current_sku = m_pack_sku.group(1)
                    continue

                m_sale = re.search(r"\bVenta:\s*([0-9]{10,20})\b", ln, flags=re.I)
                if m_sale:
                    ctx["ml_order_id"] = m_sale.group(1)
                    # En algunos casos viene un SKU y Cantidad en la misma línea
                    m_sale_sku = re.search(r"\bSKU:\s*([0-9A-Za-z_-]+)\b", ln, flags=re.I)
                    if m_sale_sku:
                        current_sku = m_sale_sku.group(1)
                    m_sale_qty = re.search(r"\bCantidad:\s*(\d+)\b", ln, flags=re.I)
                    if m_sale_qty and current_sku:
                        push_item(current_sku, m_sale_qty.group(1))
                        current_sku = None
                    continue

                # 3) SKU (línea sola)
                m_sku = re.match(r"^SKU:\s*([0-9A-Za-z_-]+)\b", ln, flags=re.I)
                if m_sku:
                    current_sku = m_sku.group(1)
                    continue

                # 4) Cantidad (línea sola) -> si hay current_sku, crea item
                m_qty = re.match(r"^Cantidad:\s*(\d+)\b", ln, flags=re.I)
                if m_qty:
                    if current_sku:
                        push_item(current_sku, m_qty.group(1))
                        current_sku = None
                    continue

                # 5) Comprador (suele venir justo después de Venta)
                if looks_like_name(ln):
                    # Si aún no hay buyer y ya hay venta, lo tomamos
                    if ctx.get("ml_order_id") and not ctx.get("buyer"):
                        ctx["buyer"] = ln[:120]
                        continue

                # 6) Guardar posible título largo como fallback
                if len(ln) >= 18 and ":" not in ln and not re.match(r"^(Despacha|Identif|Pack\s*ID|Venta:|SKU:|Cantidad:)", ln, flags=re.I):
                    current_title = ln[:200]

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
            "INSERT OR IGNORE INTO sorting_runs (manifest_id, page_no, mesa, status, created_at) VALUES (?,?,?,?,?);",
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



def _s2_now_iso():
    # Timestamp en hora Chile con offset
    if CL_TZ is not None:
        return datetime.now(CL_TZ).isoformat(timespec="seconds")
    return datetime.now().isoformat(timespec="seconds")

def _s2_create_tables():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS s2_manifests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        status TEXT NOT NULL DEFAULT 'ACTIVE',
        created_at TEXT NOT NULL
    );""")
    c.execute("""CREATE TABLE IF NOT EXISTS s2_files (
        manifest_id INTEGER PRIMARY KEY,
        control_pdf BLOB,
        labels_txt BLOB,
        control_name TEXT,
        labels_name TEXT,
        updated_at TEXT NOT NULL
    );""")
    c.execute("""CREATE TABLE IF NOT EXISTS s2_page_assign (
        manifest_id INTEGER NOT NULL,
        page_no INTEGER NOT NULL,
        mesa INTEGER NOT NULL,
        PRIMARY KEY (manifest_id, page_no)
    );""")
    c.execute("""CREATE TABLE IF NOT EXISTS s2_sales (
        manifest_id INTEGER NOT NULL,
        sale_id TEXT NOT NULL,
        shipment_id TEXT,
        page_no INTEGER NOT NULL,
        mesa INTEGER,
        status TEXT NOT NULL DEFAULT 'NEW',
        opened_at TEXT,
        closed_at TEXT,
        PRIMARY KEY (manifest_id, sale_id)
    );""")
    c.execute("""CREATE TABLE IF NOT EXISTS s2_items (
        manifest_id INTEGER NOT NULL,
        sale_id TEXT NOT NULL,
        sku TEXT NOT NULL,
        description TEXT,
        qty INTEGER NOT NULL,
        picked INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'PENDING',
        PRIMARY KEY (manifest_id, sale_id, sku)
    );""")
    c.execute("""CREATE TABLE IF NOT EXISTS s2_labels (
        manifest_id INTEGER NOT NULL,
        shipment_id TEXT NOT NULL,
        raw TEXT,
        PRIMARY KEY (manifest_id, shipment_id)
    );""")

    # --- Migraciones suaves (SQLite) ---
    try:
        cols = [r[1] for r in c.execute("PRAGMA table_info(s2_sales);").fetchall()]
        if "pack_id" not in cols:
            c.execute("ALTER TABLE s2_sales ADD COLUMN pack_id TEXT;")
        if "customer" not in cols:
            c.execute("ALTER TABLE s2_sales ADD COLUMN customer TEXT;")
    except Exception:
        pass

    # s2_items: guardar confirm_mode para trazabilidad (ej: MANUAL_NO_EAN)
    try:
        cols_i = [r[1] for r in c.execute("PRAGMA table_info(s2_items);").fetchall()]
        if "confirm_mode" not in cols_i:
            c.execute("ALTER TABLE s2_items ADD COLUMN confirm_mode TEXT;")
        if "updated_at" not in cols_i:
            c.execute("ALTER TABLE s2_items ADD COLUMN updated_at TEXT;")
    except Exception:
        pass


    # Mapa Pack ID -> Shipment ID (necesario para Colecta)
    c.execute("""CREATE TABLE IF NOT EXISTS s2_pack_ship (
        manifest_id INTEGER NOT NULL,
        pack_id TEXT NOT NULL,
        shipment_id TEXT NOT NULL,
        PRIMARY KEY (manifest_id, pack_id)
    );""")

    conn.commit()
    conn.close()

def _s2_get_active_manifest_id():
    _s2_create_tables()
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id FROM s2_manifests WHERE status='ACTIVE' ORDER BY id DESC LIMIT 1;")
    row = c.fetchone()
    if row:
        mid = int(row[0])
        conn.close()
        return mid
    c.execute("INSERT INTO s2_manifests(status, created_at) VALUES('ACTIVE', ?);", (_s2_now_iso(),))
    mid = int(c.lastrowid)
    conn.commit()
    conn.close()
    return mid


def _s2_manifest_files_state(mid: int) -> dict:
    """Return whether the active manifest already has Control and/or Labels loaded."""
    _s2_create_tables()
    conn = get_conn()
    c = conn.cursor()
    row = c.execute(
        "SELECT (control_pdf IS NOT NULL AND length(control_pdf)>0) AS has_control, "
        "       (labels_txt  IS NOT NULL AND length(labels_txt)>0)  AS has_labels "
        "FROM s2_files WHERE manifest_id=?;",
        (mid,),
    ).fetchone()
    conn.close()
    if not row:
        return {"has_control": False, "has_labels": False}
    return {"has_control": bool(row[0]), "has_labels": bool(row[1])}

def _s2_close_manifest(mid: int):
    """Marks current manifest as DONE (archived)."""
    _s2_create_tables()
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE s2_manifests SET status='DONE' WHERE id=?;", (int(mid),))
    conn.commit()
    conn.close()

def _s2_create_new_manifest() -> int:
    """Creates a new ACTIVE manifest and returns its id."""
    _s2_create_tables()
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO s2_manifests(status, created_at) VALUES('ACTIVE', ?);", (_s2_now_iso(),))
    mid = int(c.lastrowid)
    conn.commit()
    conn.close()
    return mid


def _s2_parse_label_raw_info(raw: str):
    """Extrae info visible de una etiqueta (nombre, dirección, comuna, etc.) desde el texto raw."""
    import re
    if not raw:
        return {}
    s = str(raw).replace("\r", "\n")
    info = {}
    m = re.search(r"Destinatario\s*:\s*(.+)", s, flags=re.IGNORECASE)
    if m:
        info["destinatario"] = m.group(1).strip()
    m = re.search(r"Direccion\s*:\s*(.+)", s, flags=re.IGNORECASE)
    if m:
        info["direccion"] = m.group(1).strip()
    m = re.search(r"Comuna\s*:\s*(.+)", s, flags=re.IGNORECASE)
    if m:
        info["comuna"] = m.group(1).strip()
    m = re.search(r"Ciudad\s*de\s*destino\s*:\s*(.+)", s, flags=re.IGNORECASE)
    if m:
        info["ciudad_destino"] = m.group(1).strip()
    m = re.search(r"Domicilio\s*:\s*(.+)", s, flags=re.IGNORECASE)
    if m and "direccion" not in info:
        info["direccion"] = m.group(1).strip()
    if "destinatario" not in info:
        m = re.search(r"^\s*([A-ZÁÉÍÓÚÑ][^\n]{3,60})\s*\(([^\n]{2,30})\)\s*$", s, flags=re.M)
        if m:
            info["destinatario"] = m.group(1).strip()
    return info

def _s2_get_label_raw(mid:int, shipment_id:str):
    conn=get_conn()
    c=conn.cursor()
    row=c.execute("SELECT raw FROM s2_labels WHERE manifest_id=? AND shipment_id=?;", (mid, str(shipment_id))).fetchone()
    conn.close()
    return row[0] if row else ""

def _s2_extract_shipment_id(scan_raw: str):
    """Lee el identificador desde el escaneo de etiqueta.

    - Flex: a veces viene como JSON con {"id":"..."}
    - Colecta: puede venir como shipment (10-15 dígitos, suele empezar por 46)
      o como Pack ID (más largo, 10-20 dígitos)

    Devuelve el mejor candidato numérico (string) o None.
    """
    import re, json
    if not scan_raw:
        return None
    s = str(scan_raw).strip()

    # 1) JSON (Flex QR)
    if s.startswith("{") and "id" in s:
        try:
            obj = json.loads(s)
            sid = obj.get("id")
            if sid and re.fullmatch(r"\d{8,20}", str(sid)):
                return str(sid)
        except Exception:
            pass

    # 2) Números: extraer todos los grupos (incluye prefijos tipo >: )
    nums = re.findall(r"(\d{6,20})", s)
    if not nums:
        return None

    # Preferir shipment_id típico (10-15, empieza por 46) si existe
    ship_like = [n for n in nums if 10 <= len(n) <= 15]
    if ship_like:
        ship_like = sorted(ship_like, key=lambda x: (0 if x.startswith("46") else 1, -len(x)))
        return ship_like[0]

    # Si no, devolver el más largo (útil si escanean Pack ID)
    nums_sorted = sorted(nums, key=lambda x: -len(x))
    return nums_sorted[0]



def _s2_parse_control_pdf(pdf_bytes: bytes):
    """Parse Control.pdf (Flex/Colecta) into sales with items.

    Importante (Colecta): el Control a veces NO trae shipment_id al inicio de línea.
    Por eso este parser NO exige shipment_id para contar ventas; lo completa luego
    usando Etiquetas (por Pack ID o por shipment_id cuando venga en el Control).

    Returns: list of dicts:
      {page_no:int, shipment_id:str|None, sale_id:str, pack_id:str|None, customer:str|None,
       items:[{sku:str, qty:int}]}
    """
    import io, re, pdfplumber

    def ship_from_line(s: str):
        # Flex suele venir como número al inicio (p.ej. 4636...)
        m = re.match(r"^(46\d{8,13})\b", (s or "").strip())  # evita capturar códigos no-shipment (ej: 30119784...)
        return m.group(1) if m else None

    def sale_from_line(s: str):
        m = re.search(r"\bVenta\s*:\s*(\d{10,20})\b", s or "", flags=re.IGNORECASE)
        return m.group(1) if m else None

    def pack_from_line(s: str):
        m = re.search(r"\bPack\s*ID\s*:\s*(\d{10,20})\b", s or "", flags=re.IGNORECASE)
        return m.group(1) if m else None

    def skus_from_line(s: str):
        # SKU puede venir con guiones/letras en algunos casos internos, pero en Control suele ser numérico
        return re.findall(r"\bSKU\s*:\s*([0-9A-Za-z_-]{6,20})\b", s or "", flags=re.IGNORECASE)

    def qty_from_line(s: str):
        m = re.search(r"\bCantidad\s*:\s*(\d+)\b", s or "", flags=re.IGNORECASE)
        return int(m.group(1)) if m else None

    def looks_like_name(s: str):
        s = (s or "").strip()
        if not s or len(s) > 70:
            return False
        if re.search(r"\b(color|acabado|modelo|di[aá]metro|voltaje|dise[nñ]o|tipo)\b\s*:", s, flags=re.I):
            return False
        return bool(re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", s))

    sales = []
    cur = {"page_no": None, "shipment_id": None, "sale_id": None, "pack_id": None, "customer": None, "items": []}
    sku_queue = []

    def flush():
        nonlocal cur, sku_queue
        if cur.get("sale_id") and cur.get("items"):
            # sale_id + items es suficiente para contar venta
            sales.append(cur)
        cur = {"page_no": None, "shipment_id": None, "sale_id": None, "pack_id": None, "customer": None, "items": []}
        sku_queue = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
            for ln in lines:
                low = ln.lower()
                if low.startswith("despacha ") or low.startswith("identifi"):
                    continue

                # Flex shipment id en línea
                ship = ship_from_line(ln)
                if ship:
                    if cur.get("shipment_id") and ship != cur.get("shipment_id") and cur.get("sale_id"):
                        flush()
                    if not cur.get("shipment_id"):
                        cur["shipment_id"] = ship
                        if not cur.get("page_no"):
                            cur["page_no"] = pidx

                # Pack ID (ojo: en Colecta a veces Pack+SKU viene ANTES de "Venta:",
                # así que si aparece un Pack ID nuevo y ya tenemos una venta completa, hacemos flush aquí)
                pid = pack_from_line(ln)
                if pid:
                    if cur.get("sale_id") and cur.get("items"):
                        if (cur.get("pack_id") and pid != cur.get("pack_id")) or (cur.get("pack_id") is None):
                            flush()
                    cur["pack_id"] = pid
                    if not cur.get("page_no"):
                        cur["page_no"] = pidx


                # Venta (si cambia, flush)
                sid = sale_from_line(ln)
                if sid:
                    if cur.get("sale_id") and sid != cur.get("sale_id") and cur.get("items"):
                        flush()
                    cur["sale_id"] = sid
                    if not cur.get("page_no"):
                        cur["page_no"] = pidx

                # SKU en línea
                skus = skus_from_line(ln)
                if skus:
                    sku_queue.extend(skus)

                # Cantidad: asigna a primer SKU pendiente
                q = qty_from_line(ln)
                if q is not None:
                    # cliente a veces viene junto a Cantidad
                    if cur.get("sale_id") and not cur.get("customer") and ("venta" not in low) and ("pack" not in low):
                        pre = re.split(r"Cantidad\s*:", ln, flags=re.IGNORECASE)[0].strip()
                        pre = re.sub(r"\bSKU\s*:\s*[0-9A-Za-z_-]{6,20}\b", "", pre, flags=re.IGNORECASE).strip()
                        pre = re.sub(r"^\d{8,15}\b", "", pre).strip()
                        if pre and len(pre) <= 70 and looks_like_name(pre):
                            cur["customer"] = pre

                    if sku_queue:
                        sku = sku_queue.pop(0)
                        cur["items"].append({"sku": sku, "qty": int(q)})
                else:
                    # nombre en línea sola después de Venta
                    if cur.get("sale_id") and not cur.get("customer") and looks_like_name(ln):
                        cur["customer"] = ln[:70]

    flush()
    return sales

def _s2_parse_labels_txt(raw_bytes: bytes):
    """Parsea etiquetas TXT/ZPL de Flex y Colecta.

    Devuelve:
      - pack_to_ship: dict {pack_id(str) -> shipment_id(str)}
      - sale_to_ship: dict {sale_id(str) -> shipment_id(str)}  (fallback cuando no hay Pack ID en Control)
      - shipment_ids: sorted list de shipment_id detectados

    Nota: En Colecta el Pack ID / Venta suelen venir PARTIDOS en dos ^FD:
        ^FDPack ID: 20000^FS  y luego ^FD1128....^FS  -> 200001128....
        ^FDVenta: 20000^FS    y luego ^FD1498....^FS  -> 200001498....
    """
    import re

    try:
        txt = raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        txt = str(raw_bytes)

    # separar etiquetas por bloque ^XA ... ^XZ
    blocks = re.split(r"\^XA", txt)
    pack_to_ship = {}
    sale_to_ship = {}
    shipment_ids = set()

    def clean_num(s):
        return re.sub(r"\D", "", s or "")

    def rebuild_split_id(kind: str, b: str):
        """
        kind: 'Pack' o 'Venta'
        Busca:
          - Completo:  kind ID: 2000011363....
          - Partido:   kind ID: 20000  + siguiente ^FD 11363....
        """
        kind_re = kind
        full = None

        m_full = re.search(rf"{kind_re}\s*(?:ID)?\s*:\s*(\d{{10,20}})", b, flags=re.I)
        if m_full:
            full = clean_num(m_full.group(1))

        if not full:
            m_part = re.search(rf"{kind_re}\s*(?:ID)?\s*:\s*(\d{{4,10}})\s*\^FS", b, flags=re.I)
            if m_part:
                head = clean_num(m_part.group(1))
                tailm = re.search(r"\^FD\s*([0-9 ]{6,20})\s*\^FS", b[m_part.end():])
                if tailm:
                    tail = clean_num(tailm.group(1))
                    cand = head + tail
                    if 10 <= len(cand) <= 20:
                        full = cand
        return full

    for b in blocks:
        if not b.strip():
            continue

        pack_full = rebuild_split_id("Pack", b)
        sale_full = rebuild_split_id("Venta", b)

        # shipment id: preferir JSON con "id":"4638..."
        ship = None
        jm = re.search(r"\"id\"\s*:\s*\"(\d{8,15})\"", b)
        if jm:
            ship = jm.group(1)

        if not ship:
            # buscar números candidatos, priorizando 10-15 dígitos y que empiecen por 46
            nums = re.findall(r"\b\d{10,15}\b", b)
            if nums:
                nums_sorted = sorted(nums, key=lambda x: (0 if x.startswith("46") else 1, -len(x)))
                ship = nums_sorted[0]

        if ship:
            shipment_ids.add(ship)
            if pack_full:
                pack_to_ship[str(pack_full)] = str(ship)
            if sale_full:
                sale_to_ship[str(sale_full)] = str(ship)

    return pack_to_ship, sale_to_ship, sorted(shipment_ids)

def _s2_upsert_control(mid: int, pdf_name: str, pdf_bytes: bytes):
    pages_sales = _s2_parse_control_pdf(pdf_bytes)
    conn = get_conn()
    c = conn.cursor()
    # store file
    c.execute("""INSERT INTO s2_files(manifest_id, control_pdf, control_name, updated_at)
                 VALUES(?, ?, ?, ?)
                 ON CONFLICT(manifest_id) DO UPDATE SET
                    control_pdf=excluded.control_pdf,
                    control_name=excluded.control_name,
                    updated_at=excluded.updated_at;""", (mid, pdf_bytes, pdf_name, _s2_now_iso()))
    # clear previous parsed sales/items
    c.execute("DELETE FROM s2_items WHERE manifest_id=?;", (mid,))
    c.execute("DELETE FROM s2_sales WHERE manifest_id=?;", (mid,))

    n_sales = 0
    for s in pages_sales:
        sale_id = str(s.get("sale_id") or "")
        if not sale_id:
            continue
        n_sales += 1
        shipment_id = s.get("shipment_id")
        page_no = int(s.get("page_no") or 1)
        pack_id = s.get("pack_id")
        customer = s.get("customer")

        c.execute("""INSERT INTO s2_sales(manifest_id, sale_id, shipment_id, page_no, status, pack_id, customer)
                     VALUES(?,?,?,?, 'NEW', ?, ?)
                     ON CONFLICT(manifest_id, sale_id) DO UPDATE SET
                        shipment_id=excluded.shipment_id,
                        page_no=excluded.page_no,
                        status='NEW',
                        mesa=NULL,
                        opened_at=NULL,
                        closed_at=NULL,
                        pack_id=excluded.pack_id,
                        customer=excluded.customer;""",
                  (mid, sale_id, (str(shipment_id) if shipment_id else None), page_no,
                   (str(pack_id) if pack_id else None), (str(customer) if customer else None)))

        for it in s.get("items", []):
            try:
                sku = str(it.get("sku"))
                qty = int(it.get("qty") or 0)
            except Exception:
                continue
            if not sku or qty <= 0:
                continue
            c.execute("""INSERT INTO s2_items(manifest_id, sale_id, sku, description, qty, picked, status)
                         VALUES(?,?,?,?,?,0,'PENDING')
                         ON CONFLICT(manifest_id, sale_id, sku) DO UPDATE SET
                            description=excluded.description,
                            qty=excluded.qty,
                            picked=0,
                            status='PENDING';""", (mid, sale_id, sku, it.get("desc",""), qty))

    conn.commit()
    conn.close()
    return n_sales


def _s2_upsert_labels(mid: int, labels_name: str, labels_bytes: bytes):
    pack_to_ship, sale_to_ship, shipment_ids = _s2_parse_labels_txt(labels_bytes)
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO s2_files(manifest_id, labels_txt, labels_name, updated_at)
                 VALUES(?, ?, ?, ?)
                 ON CONFLICT(manifest_id) DO UPDATE SET
                    labels_txt=excluded.labels_txt,
                    labels_name=excluded.labels_name,
                    updated_at=excluded.updated_at;""", (mid, labels_bytes, labels_name, _s2_now_iso()))

    # limpiar y reinsertar shipment ids
    c.execute("DELETE FROM s2_labels WHERE manifest_id=?;", (mid,))
    for sid in shipment_ids:
        c.execute("INSERT OR REPLACE INTO s2_labels(manifest_id, shipment_id, raw) VALUES(?,?,NULL);", (mid, str(sid)))

    # guardar pack->ship para Colecta
    if pack_to_ship:
        for pack_id, ship_id in pack_to_ship.items():
            c.execute("INSERT OR REPLACE INTO s2_pack_ship(manifest_id, pack_id, shipment_id) VALUES(?,?,?);",
                      (mid, str(pack_id), str(ship_id)))

        # completar shipment_id en ventas usando pack_id si falta
        try:
            c.execute("""UPDATE s2_sales
                           SET shipment_id = (
                               SELECT ps.shipment_id FROM s2_pack_ship ps
                               WHERE ps.manifest_id=s2_sales.manifest_id AND ps.pack_id=s2_sales.pack_id
                           )
                           WHERE manifest_id=? AND (shipment_id IS NULL OR shipment_id='') AND pack_id IS NOT NULL AND pack_id!='';""", (mid,))
        except Exception:
            pass

    # fallback: completar shipment_id por sale_id (cuando el Control no trae Pack ID)
    if sale_to_ship:
        try:
            for sale_id, ship_id in sale_to_ship.items():
                c.execute("""UPDATE s2_sales
                             SET shipment_id=?
                             WHERE manifest_id=? AND sale_id=? AND (shipment_id IS NULL OR shipment_id='');""",
                          (str(ship_id), mid, str(sale_id)))
        except Exception:
            pass

    conn.commit()
    conn.close()
    return len(shipment_ids)

def _s2_get_stats(mid: int):
    """
    Stats del manifiesto (Sorting v2).
    Incluye aliases para UI: ventas/items/etiquetas/... para evitar KeyError.
    Tolerante a cambios de esquema.
    """
    conn = get_conn()
    c = conn.cursor()

    def has_col(table: str, col: str) -> bool:
        try:
            cols = [r[1] for r in c.execute(f"PRAGMA table_info({table});").fetchall()]
            return col in cols
        except Exception:
            return False

    stats = {}

    # Core counts
    sales_total = int(c.execute("SELECT COUNT(*) FROM s2_sales WHERE manifest_id=?;", (mid,)).fetchone()[0] or 0)
    items_total = int(c.execute("SELECT COUNT(*) FROM s2_items WHERE manifest_id=?;", (mid,)).fetchone()[0] or 0)
    labels_total = int(c.execute("SELECT COUNT(*) FROM s2_labels WHERE manifest_id=?;", (mid,)).fetchone()[0] or 0)

    sales_pending = int(c.execute("SELECT COUNT(*) FROM s2_sales WHERE manifest_id=? AND status='PENDING';", (mid,)).fetchone()[0] or 0)
    sales_done    = int(c.execute("SELECT COUNT(*) FROM s2_sales WHERE manifest_id=? AND status='DONE';", (mid,)).fetchone()[0] or 0)

    items_pending = int(c.execute("SELECT COUNT(*) FROM s2_items WHERE manifest_id=? AND status='PENDING';", (mid,)).fetchone()[0] or 0)
    items_done    = int(c.execute("SELECT COUNT(*) FROM s2_items WHERE manifest_id=? AND status='DONE';", (mid,)).fetchone()[0] or 0)
    items_incid   = int(c.execute("SELECT COUNT(*) FROM s2_items WHERE manifest_id=? AND status='INCIDENCE';", (mid,)).fetchone()[0] or 0)

    # Labels with shipment_id
    if has_col("s2_labels", "shipment_id"):
        labels_with_ship = int(c.execute(
            "SELECT COUNT(*) FROM s2_labels WHERE manifest_id=? AND shipment_id IS NOT NULL AND shipment_id!='';",
            (mid,)
        ).fetchone()[0] or 0)
        distinct_ship_labels = int(c.execute(
            "SELECT COUNT(DISTINCT shipment_id) FROM s2_labels WHERE manifest_id=? AND shipment_id IS NOT NULL AND shipment_id!='';",
            (mid,)
        ).fetchone()[0] or 0)
    else:
        labels_with_ship = 0
        distinct_ship_labels = 0

    # Pack ID availability
    # In control, pack_id usually lives in s2_sales.pack_id (not in labels).
    sales_with_pack = 0
    distinct_packs = 0
    if has_col("s2_sales", "pack_id"):
        sales_with_pack = int(c.execute(
            "SELECT COUNT(*) FROM s2_sales WHERE manifest_id=? AND pack_id IS NOT NULL AND pack_id!='';",
            (mid,)
        ).fetchone()[0] or 0)
        distinct_packs = int(c.execute(
            "SELECT COUNT(DISTINCT pack_id) FROM s2_sales WHERE manifest_id=? AND pack_id IS NOT NULL AND pack_id!='';",
            (mid,)
        ).fetchone()[0] or 0)

    labels_with_pack = 0
    if has_col("s2_labels", "pack_id"):
        labels_with_pack = int(c.execute(
            "SELECT COUNT(*) FROM s2_labels WHERE manifest_id=? AND pack_id IS NOT NULL AND pack_id!='';",
            (mid,)
        ).fetchone()[0] or 0)

    labels_with_sale = 0
    if has_col("s2_labels", "sale_id"):
        labels_with_sale = int(c.execute(
            "SELECT COUNT(*) FROM s2_labels WHERE manifest_id=? AND sale_id IS NOT NULL AND sale_id!='';",
            (mid,)
        ).fetchone()[0] or 0)

    # Sales with shipment_id (after matching)
    if has_col("s2_sales", "shipment_id"):
        sales_with_ship = int(c.execute(
            "SELECT COUNT(*) FROM s2_sales WHERE manifest_id=? AND shipment_id IS NOT NULL AND shipment_id!='';",
            (mid,)
        ).fetchone()[0] or 0)
    else:
        sales_with_ship = 0

    missing_ship = sales_total - sales_with_ship

    # Matches by pack (if mapping table exists)
    matched_by_pack = 0
    if "s2_pack_ship" in [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]:
        matched_by_pack = int(c.execute(
            "SELECT COUNT(DISTINCT pack_id) FROM s2_pack_ship WHERE manifest_id=? AND pack_id IS NOT NULL AND pack_id!='' AND shipment_id IS NOT NULL AND shipment_id!='';",
            (mid,)
        ).fetchone()[0] or 0)

    # Fill canonical keys
    stats.update({
        "sales_total": sales_total,
        "sales_pending": sales_pending,
        "sales_done": sales_done,
        "items_total": items_total,
        "items_pending": items_pending,
        "items_done": items_done,
        "items_incidence": items_incid,
        "labels_total": labels_total,
        "labels_with_ship": labels_with_ship,
        "labels_unique_ship": distinct_ship_labels,
        "sales_with_pack": sales_with_pack,
        "distinct_packs": distinct_packs,
        "sales_with_ship": sales_with_ship,
        "sales_missing_ship": missing_ship,
        "labels_with_pack": labels_with_pack,
        "labels_with_sale": labels_with_sale,
        "matched_by_pack": matched_by_pack,
    })

    # Aliases expected by UI (legacy naming)
    stats.update({
        "ventas": sales_total,
        "items": items_total,
        "etiquetas": labels_total,
        "distinct_ship_labels": distinct_ship_labels,
        "ventas_with_pack": sales_with_pack,
        "ventas_with_ship": sales_with_ship,
        "missing_ship": missing_ship,
        "matched_by_pack": matched_by_pack,
    })

    conn.close()
    return stats

def _s2_reset_all_sorting():
    """Hard reset of Sorting module only (keeps other modules intact)."""
    conn = get_conn()
    c = conn.cursor()
    # New (s2_*) tables
    s2_tables = [
        "s2_page_assign",
        "s2_pack_ship",
        "s2_labels",
        "s2_items",
        "s2_sales",
        "s2_files",
        "s2_manifests",
    ]
    for t in s2_tables:
        c.execute(f"DELETE FROM {t};")

    # Legacy sorting tables (kept for backward compat in older code paths)
    legacy = [
        "sorting_run_items",
        "sorting_runs",
        "sorting_labels",
        "sorting_manifests",
        "sorting_status",
    ]
    for t in legacy:
        try:
            c.execute(f"DELETE FROM {t};")
        except Exception:
            pass

    conn.commit()


def _s2_get_pages(mid:int):
    conn=get_conn()
    c=conn.cursor()
    c.execute("SELECT DISTINCT page_no FROM s2_sales WHERE manifest_id=? ORDER BY page_no;", (mid,))
    pages=[int(r[0]) for r in c.fetchall()]
    conn.close()
    return pages

def _s2_auto_assign_pages(mid:int, num_mesas:int=10):
    pages=_s2_get_pages(mid)
    if not pages:
        return 0
    conn=get_conn()
    c=conn.cursor()
    for i,p in enumerate(pages):
        mesa = (i % num_mesas) + 1
        c.execute("""INSERT INTO s2_page_assign(manifest_id, page_no, mesa)
                     VALUES(?,?,?)
                     ON CONFLICT(manifest_id, page_no) DO UPDATE SET mesa=excluded.mesa;""", (mid, p, mesa))
    conn.commit()
    conn.close()
    return len(pages)

def _s2_get_assignments(mid:int):
    conn=get_conn()
    c=conn.cursor()
    c.execute("SELECT page_no, mesa FROM s2_page_assign WHERE manifest_id=? ORDER BY page_no;", (mid,))
    rows=[(int(r[0]), int(r[1])) for r in c.fetchall()]
    conn.close()
    return rows

def _s2_set_assignment(mid:int, page_no:int, mesa:int):
    conn=get_conn()
    c=conn.cursor()
    c.execute("""INSERT INTO s2_page_assign(manifest_id, page_no, mesa)
                 VALUES(?,?,?)
                 ON CONFLICT(manifest_id, page_no) DO UPDATE SET mesa=excluded.mesa;""", (mid, int(page_no), int(mesa)))
    conn.commit()
    conn.close()

def _s2_create_corridas(mid:int):
    # apply mesa from page assignments to sales
    conn=get_conn()
    c=conn.cursor()
    c.execute("SELECT page_no, mesa FROM s2_page_assign WHERE manifest_id=?;", (mid,))
    page_to_mesa = {int(p): int(m) for p,m in c.fetchall()}
    # update sales
    c.execute("SELECT sale_id, page_no FROM s2_sales WHERE manifest_id=?;", (mid,))
    sales = c.fetchall()
    updated=0
    for sale_id, page_no in sales:
        mesa = page_to_mesa.get(int(page_no))
        if mesa is None:
            continue
        c.execute("""UPDATE s2_sales
                     SET mesa=?, status='PENDING', opened_at=NULL, closed_at=NULL
                     WHERE manifest_id=? AND sale_id=?;""", (mesa, mid, sale_id))
        updated += 1
    conn.commit()
    conn.close()
    return updated

def _s2_find_sale_for_scan(mid:int, mesa:int, shipment_id:str):
    conn=get_conn()
    c=conn.cursor()
    c.execute("""SELECT sale_id FROM s2_sales
                 WHERE manifest_id=? AND mesa=? AND shipment_id=? AND status='PENDING'
                 ORDER BY page_no, sale_id
                 LIMIT 1;""", (mid, int(mesa), str(shipment_id)))
    row=c.fetchone()
    conn.close()
    return row[0] if row else None

def _s2_find_sale_for_pack_scan(mid:int, mesa:int, pack_id:str):
    """Fallback: algunos escáneres/etiquetas devuelven Pack ID en vez de Shipment ID (Colecta)."""
    conn=get_conn()
    c=conn.cursor()
    c.execute("""SELECT sale_id FROM s2_sales
                 WHERE manifest_id=? AND mesa=? AND pack_id=? AND status='PENDING'
                 ORDER BY page_no, sale_id
                 LIMIT 1;""", (mid, int(mesa), str(pack_id)))
    row=c.fetchone()
    conn.close()
    return row[0] if row else None


def _s2_sale_items(mid:int, sale_id:str):
    conn=get_conn()
    c=conn.cursor()
    c.execute("""SELECT sku, description, qty, picked, status
                 FROM s2_items WHERE manifest_id=? AND sale_id=? ORDER BY sku;""", (mid, sale_id))
    rows=c.fetchall()
    conn.close()
    return rows

def _s2_apply_pick(mid:int, sale_id:str, sku:str, add_qty:int):
    conn=get_conn()
    c=conn.cursor()
    c.execute("""SELECT qty, picked FROM s2_items
                 WHERE manifest_id=? AND sale_id=? AND sku=?;""", (mid, sale_id, sku))
    row=c.fetchone()
    if not row:
        conn.close()
        return False, "SKU no pertenece a esta venta"
    qty, picked = int(row[0]), int(row[1])
    new_picked = min(qty, picked + int(add_qty))
    status = "DONE" if new_picked >= qty else "PENDING"
    c.execute("""UPDATE s2_items SET picked=?, status=? WHERE manifest_id=? AND sale_id=? AND sku=?;""", 
              (new_picked, status, mid, sale_id, sku))
    # if all done, allow close
    conn.commit()
    conn.close()
    return True, None


def _s2_mark_incidence(mid:int, sale_id:str, sku:str, note:str=""):
    conn=get_conn()
    c=conn.cursor()
    c.execute("UPDATE s2_items SET status='INCIDENCE', confirm_mode='INCIDENCE', updated_at=? WHERE manifest_id=? AND sale_id=? AND sku=?;", (_s2_now_iso(), mid, sale_id, sku))
    conn.commit()
    conn.close()

def _s2_force_done_no_ean(mid:int, sale_id:str, sku:str):
    conn=get_conn()
    c=conn.cursor()
    c.execute("SELECT qty, picked FROM s2_items WHERE manifest_id=? AND sale_id=? AND sku=?;", (mid, sale_id, sku))
    row=c.fetchone()
    if not row:
        conn.close()
        return False
    qty=int(row[0] or 0)
    c.execute("UPDATE s2_items SET picked=?, status='DONE', confirm_mode='MANUAL_NO_EAN', updated_at=? WHERE manifest_id=? AND sale_id=? AND sku=?;", (qty, _s2_now_iso(), mid, sale_id, sku))
    conn.commit()
    conn.close()
    return True

def _s2_is_sale_done(mid:int, sale_id:str):
    conn=get_conn()
    c=conn.cursor()
    c.execute("""SELECT COUNT(1) FROM s2_items WHERE manifest_id=? AND sale_id=? AND status NOT IN ('DONE','INCIDENCE');""", (mid, sale_id))
    rem=int(c.fetchone()[0] or 0)
    conn.close()
    return rem==0

def _s2_close_sale(mid:int, sale_id:str):
    conn=get_conn()
    c=conn.cursor()
    c.execute("""UPDATE s2_sales SET status='DONE', closed_at=? WHERE manifest_id=? AND sale_id=?;""", (_s2_now_iso(), mid, sale_id))
    conn.commit()
    conn.close()

def _s2_reset_all():
    conn=get_conn()
    c=conn.cursor()
    for t in ["s2_labels","s2_items","s2_sales","s2_page_assign","s2_files","s2_manifests"]:
        c.execute(f"DROP TABLE IF EXISTS {t};")
    conn.commit()
    conn.close()

def page_sorting_upload(inv_map_sku, barcode_to_sku):
    _s2_create_tables()
    st.title("Sorting - Carga y Corridas")

    mid = _s2_get_active_manifest_id()
    st.session_state["sorting_manifest_id"] = mid

    st.caption(f"Manifiesto activo: {mid}")

    files_state = _s2_manifest_files_state(mid)
    lock_control = bool(files_state.get("has_control"))
    if lock_control:
        st.warning("🔒 Ya hay un Control cargado en el manifiesto activo. Para cargar un manifiesto nuevo debes **Cerrar** o **Reiniciar** el Sorting desde Administrador.")

    col1, col2 = st.columns(2)
    with col1:
        pdf = st.file_uploader("Control (PDF)", type=["pdf"], key="s2_control_pdf", disabled=lock_control)
    with col2:
        zpl = st.file_uploader("Etiquetas de envío (TXT/ZPL)", type=["txt","zpl"], key="s2_labels_txt")

    if pdf is not None:
        n_sales = _s2_upsert_control(mid, getattr(pdf, "name", "control.pdf"), pdf.getvalue())
        st.success(f"Control cargado. Ventas detectadas: {n_sales}")
        _s2_auto_assign_pages(mid, num_mesas=10)

    if zpl is not None:
        n_labels = _s2_upsert_labels(mid, getattr(zpl, "name", "etiquetas.txt"), zpl.getvalue())
        st.success(f"Etiquetas cargadas. IDs detectados: {n_labels}")

    # Resumen (para evitar confusión: ventas y etiquetas NO siempre coinciden 1:1)
    stats = _s2_get_stats(mid)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ventas (Control)", stats["ventas"])
    c2.metric("Items (líneas)", stats["items"])
    c3.metric("Etiquetas (total)", stats["etiquetas"])
    c4.metric("Envíos únicos (labels)", stats["distinct_ship_labels"])

    with st.expander("Ver detalle de conciliación", expanded=False):
        st.write(
            {
                "Ventas con Pack ID": stats["ventas_with_pack"],
                "Packs distintos (Control)": stats["distinct_packs"],
                "Ventas con Envío (Control)": stats["ventas_with_ship"],
                "Etiquetas con Pack ID": stats["labels_with_pack"],
                "Etiquetas con Venta": stats["labels_with_sale"],
                "Ventas matcheadas por Pack": stats["matched_by_pack"],
                "Ventas sin Envío asignado": stats["missing_ship"],
            }
        )

    pages = _s2_get_pages(mid)
    if not pages:
        st.info("Sube el Control.pdf para continuar.")
        return

    st.subheader("Asignación Página → Mesa")
    assigns = dict(_s2_get_assignments(mid))
    for p in pages:
        cur = assigns.get(p, 1)
        new_mesa = st.number_input(f"Página {p} → Mesa", min_value=1, max_value=50, value=int(cur), key=f"s2_mesa_{p}")
        if int(new_mesa) != int(cur):
            _s2_set_assignment(mid, p, int(new_mesa))

    # Validate all pages assigned
    assigns = dict(_s2_get_assignments(mid))
    missing = [p for p in pages if p not in assigns]
    if missing:
        st.warning(f"Faltan páginas por asignar: {missing}")
        if st.button("Auto-asignar faltantes", use_container_width=True):
            _s2_auto_assign_pages(mid, num_mesas=10)
            st.rerun()

    st.divider()
    if st.button("✅ Crear corridas", use_container_width=True):
        created = _s2_create_corridas(mid)
        if created <= 0:
            st.error("No se crearon corridas. Revisa asignación de páginas.")
        else:
            st.success(f"Corridas creadas/actualizadas: {created}")
            st.session_state["s2_last_created"] = created

def page_sorting_camarero(inv_map_sku, barcode_to_sku):
    _s2_create_tables()
    st.title("Camarero")
    st.caption("Escaneo por etiqueta (Flex/Colecta) y productos por SKU/EAN")
    mid = _s2_get_active_manifest_id()
    st.session_state["sorting_manifest_id"] = mid

    mesa = st.number_input("Mesa", min_value=1, max_value=50, value=int(st.session_state.get("s2_mesa", 1)), key="s2_mesa")
    st.session_state["s2_mesa_int"] = int(mesa)  # store separately; do not overwrite widget key

    # State: current sale
    if "s2_sale_open" not in st.session_state:
        st.session_state["s2_sale_open"] = None

    if st.session_state["s2_sale_open"] is None:
        st.subheader("Escanea etiqueta (QR Flex o barra Colecta)")
        # Limpieza segura del campo de escaneo (evita StreamlitAPIException)
        if st.session_state.get("s2_clear_label_scan"):
            st.session_state["s2_label_scan_widget"] = ""
            st.session_state["s2_clear_label_scan"] = False

        scan = st.text_input("Etiqueta", key="s2_label_scan_widget")
        if scan:
            sid = _s2_extract_shipment_id(scan)
            if not sid:
                st.error("No pude leer el ID de envío desde el escaneo.")
            else:
                sale_id = _s2_find_sale_for_scan(mid, int(mesa), sid)
                if (not sale_id) and sid:
                    # fallback: si el escaneo corresponde a Pack ID (Colecta)
                    sale_id = _s2_find_sale_for_pack_scan(mid, int(mesa), sid)
                if not sale_id:
                    # debug: exists in other mesa?
                    conn=get_conn(); c=conn.cursor()
                    c.execute("SELECT mesa, status FROM s2_sales WHERE manifest_id=? AND shipment_id=? LIMIT 5;", (mid, sid))
                    info=c.fetchall(); conn.close()
                    if info:
                        st.warning(f"Etiqueta encontrada pero no pendiente en mesa {mesa}. Coincidencias: {info}")
                    else:
                        st.error("No encontré esta etiqueta en corridas pendientes.")
                else:
                    st.session_state["s2_sale_open"] = sale_id
                    st.session_state["s2_clear_label_scan"] = True
                    st.rerun()
                    st.rerun()
        return

    sale_id = st.session_state["s2_sale_open"]
    st.info(f"Venta abierta: {sale_id}")


    # Información de la etiqueta / envío
    conn=get_conn(); c=conn.cursor()
    sale_row = c.execute("SELECT shipment_id, pack_id, customer, page_no, mesa, status FROM s2_sales WHERE manifest_id=? AND sale_id=?;", (mid, sale_id)).fetchone()
    conn.close()
    shipment_id = sale_row[0] if sale_row else ""
    pack_id = sale_row[1] if sale_row else ""
    customer = sale_row[2] if sale_row else ""
    page_no = sale_row[3] if sale_row else ""
    mesa_db = sale_row[4] if sale_row else ""

    raw_label = _s2_get_label_raw(mid, shipment_id) if shipment_id else ""
    info = _s2_parse_label_raw_info(raw_label)

    st.markdown("### Etiqueta / Envío")
    a,b,cx = st.columns(3)
    a.metric("Envío", str(shipment_id) if shipment_id else "-")
    b.metric("Pack ID", str(pack_id) if pack_id else "-")
    cx.metric("Mesa / Página", f"{mesa_db}/{page_no}" if page_no else str(mesa_db))

    name = info.get("destinatario") or customer or "-"
    addr = info.get("direccion") or "-"
    comuna = info.get("comuna") or info.get("ciudad_destino") or "-"
    
    
    

    items = _s2_sale_items(mid, sale_id)

    st.markdown("### Productos de la venta")
    total_items = len(items)
    done_items = sum(1 for _sku,_d,_q,_p,stx in items if stx in ("DONE","INCIDENCE"))
    st.progress(0 if total_items==0 else done_items/total_items)
    st.caption(f"{done_items}/{total_items} ítems finalizados (DONE o INCIDENCE)")

    for sku, desc, qty, picked, status in items:
        title = None
        if isinstance(inv_map_sku, dict):
            k = str(sku).strip()
            title = inv_map_sku.get(k)
            if title is None and k.isdigit():
                try:
                    title = inv_map_sku.get(str(int(k)))
                except Exception:
                    pass
        title = title or desc or str(sku)

        remaining = max(0, int(qty) - int(picked))
        row1 = st.columns([6, 2, 2])
        row1[0].markdown(f"### {title}  \nSKU: `{sku}`")
        row1[1].markdown(f"## {int(qty)}")
        row1[2].metric("Hecho", int(picked))

        if status != "DONE" and remaining > 0:
            bcols = st.columns([1,1,6])
            if bcols[0].button("⚠️ Incidencia", key=f"s2_inc_{sale_id}_{sku}"):
                _s2_mark_incidence(mid, sale_id, str(sku))
                st.rerun()
            if bcols[1].button("📝 Sin EAN", key=f"s2_noean_{sale_id}_{sku}"):
                _s2_force_done_no_ean(mid, sale_id, str(sku))
                st.rerun()
        st.divider()

    st.subheader("Escanea SKU/EAN del producto")
    st.caption("Escanea **1 vez**. Luego verificas la cantidad solicitada (sin digitar).")

    # Estado de confirmación por producto
    if "s2_pending_sku" not in st.session_state:
        st.session_state["s2_pending_sku"] = None
        st.session_state["s2_pending_qty"] = 0
        st.session_state["s2_pending_title"] = ""

    pending_sku = st.session_state.get("s2_pending_sku")

    # Limpieza segura del campo de producto (evita StreamlitAPIException)
    if st.session_state.get("s2_clear_prod_scan"):
        st.session_state["s2_prod_scan_widget"] = ""
        st.session_state["s2_clear_prod_scan"] = False

    sku_scan = st.text_input(
        "Producto",
        key="s2_prod_scan_widget",
        disabled=bool(pending_sku)  # mientras confirmas, bloquea nuevo escaneo
    )

    # 1) Al escanear: identificamos el SKU y preparamos la verificación automática de cantidad pendiente
    sku_scan = st.session_state.get("s2_prod_scan_widget", "").strip()
    if sku_scan and not pending_sku:
        sku = resolve_scan_to_sku(sku_scan, barcode_to_sku)

        # Buscar qty/picked del ítem dentro de esta venta
        connx = get_conn()
        cx = connx.cursor()
        cx.execute(
            "SELECT qty, picked, description FROM s2_items WHERE manifest_id=? AND sale_id=? AND sku=?;",
            (mid, sale_id, str(sku))
        )
        row = cx.fetchone()
        connx.close()

        if not row:
            st.error("SKU/EAN no pertenece a esta venta.")
        else:
            qty_req, picked_now, desc_ml = int(row[0]), int(row[1]), row[2]
            remaining = max(0, qty_req - picked_now)

            # Resolver título visible (maestro > descripción > SKU)
            title_show = ""
            if isinstance(inv_map_sku, dict):
                k = str(sku).strip()
                title_show = inv_map_sku.get(k) or inv_map_sku.get(normalize_sku(k)) or ""
            title_show = title_show or (desc_ml or "") or str(sku)

            if remaining <= 0:
                st.info(f"✅ Ya está completo: {title_show}")
                st.session_state["s2_clear_prod_scan"] = True
                st.rerun()
            else:
                st.session_state["s2_pending_sku"] = str(sku)
                st.session_state["s2_pending_qty"] = int(remaining)
                st.session_state["s2_pending_title"] = str(title_show)
                st.session_state["s2_clear_prod_scan"] = True
                st.rerun()

    # 2) Si hay un SKU pendiente: mostrar verificación de cantidad (sin digitar)
    pending_sku = st.session_state.get("s2_pending_sku")
    if pending_sku:
        pending_qty = int(st.session_state.get("s2_pending_qty", 0) or 0)
        pending_title = st.session_state.get("s2_pending_title", "") or pending_sku

        st.warning(f"Verificar **{pending_qty}** unidad(es) para: **{pending_title}**")
        cA, cB = st.columns([2, 1])
        with cA:
            if st.button(f"✅ Verificar {pending_qty} y cerrar producto", key=f"s2_verify_{sale_id}_{pending_sku}", use_container_width=True):
                ok, msg = _s2_apply_pick(mid, sale_id, str(pending_sku), int(pending_qty))
                if not ok:
                    st.error(msg or "No se pudo aplicar.")
                else:
                    st.session_state["s2_pending_sku"] = None
                    st.session_state["s2_pending_qty"] = 0
                    st.session_state["s2_pending_title"] = ""
                    st.rerun()
        with cB:
            if st.button("Cancelar", key=f"s2_verify_cancel_{sale_id}_{pending_sku}", use_container_width=True):
                st.session_state["s2_pending_sku"] = None
                st.session_state["s2_pending_qty"] = 0
                st.session_state["s2_pending_title"] = ""
                st.rerun()

    done = _s2_is_sale_done(mid, sale_id)

    st.subheader("Cerrar venta")
    if done:
        c1, c2 = st.columns([1,2])
        with c1:
            confirm_close = st.checkbox("Confirmo cierre", key=f"s2_confirm_close_{sale_id}")
        with c2:
            if st.button("✅ Cerrar venta y volver a escanear etiqueta", key=f"s2_close_{sale_id}", use_container_width=True, disabled=not confirm_close):
                _s2_close_sale(mid, sale_id)
                st.session_state["s2_sale_open"] = None
                st.session_state["s2_clear_prod_scan"] = True
                st.session_state["s2_clear_label_scan"] = True
                st.rerun()
    else:
        st.info("Para cerrar: completa todos los productos o márcalos como Incidencia / Sin EAN.")



def page_sorting_admin(inv_map_sku, barcode_to_sku):
    _s2_create_tables()
    st.title("Administrador")

    # Respaldo/Restauración SOLO SORTING (no afecta otros módulos)
    _render_module_backup_ui("sorting", "Sorting", SORTING_TABLES)

    # Manifiesto activo
    try:
        mid = _s2_get_active_manifest_id()
    except Exception:
        mid = None

    if not mid:
        st.warning("No hay manifiesto activo. Primero carga Control + Etiquetas y crea corridas.")
        return

    conn = get_conn()
    c = conn.cursor()

    # archivo/control info
    f = c.execute("SELECT control_name, labels_name, updated_at FROM s2_files WHERE manifest_id=?", (mid,)).fetchone()
    stats = _s2_get_stats(mid)

    # ---- Estado del manifiesto (como en Admin Picking: métricas arriba) ----
    st.subheader("Estado del manifiesto activo")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Manifiesto ID", mid)
    colB.metric("Ventas (Control)", stats.get("ventas", 0))
    colC.metric("Items", stats.get("items", 0))
    colD.metric("Etiquetas", stats.get("etiquetas", 0))

    if f:
        control_name, labels_name, updated_at = f
        st.caption(f"Control: {control_name or '-'} · Etiquetas: {labels_name or '-'} · Actualizado: {updated_at or '-'}")
    else:
        st.caption("Aún no se han cargado archivos para este manifiesto.")

    # ---- Trazabilidad ----
    st.divider()
    st.subheader("Trazabilidad")

    rows = c.execute(
        "SELECT mesa, COUNT(*) as ventas, "
        "SUM(CASE WHEN status='DONE' THEN 1 ELSE 0 END) as done "
        "FROM s2_sales WHERE manifest_id=? GROUP BY mesa ORDER BY mesa;",
        (mid,)
    ).fetchall()

    if rows:
        mesa_data = []
        for mesa, ventas, done in rows:
            ventas = int(ventas or 0)
            done = int(done or 0)
            mesa_data.append({
                "Mesa": int(mesa or 0),
                "Ventas": ventas,
                "Cerradas": done,
                "%": 0 if ventas == 0 else round(done * 100 / ventas, 1),
            })
        st.dataframe(mesa_data, use_container_width=True, hide_index=True)
    else:
        st.info("No hay ventas asignadas a mesas todavía.")

    # ---- Incidencias (bajo trazabilidad) ----
    st.divider()
    st.subheader("Incidencias")

    inc_rows = c.execute(
        """SELECT s.sale_id, s.mesa, s.shipment_id,
                  i.sku, i.description, i.qty, i.picked, i.status,
                  COALESCE(i.confirm_mode,'') as confirm_mode,
                  COALESCE(i.updated_at,'') as updated_at
             FROM s2_items i
             JOIN s2_sales s
               ON s.manifest_id=i.manifest_id AND s.sale_id=i.sale_id
            WHERE i.manifest_id=?
              AND (i.status='INCIDENCE' OR i.confirm_mode='MANUAL_NO_EAN')
            ORDER BY s.mesa, s.sale_id, i.sku;""",
        (mid,),
    ).fetchall()

    if inc_rows:
        df_inc = pd.DataFrame(
            inc_rows,
            columns=[
                "Venta", "Mesa", "Envío", "SKU", "Descripción Control",
                "Solicitado", "Verificado", "Estado", "Modo", "Hora"
            ],
        )

        def _title_tec_for_sku(sku_val, fallback_desc=""):
            try:
                if isinstance(inv_map_sku, dict):
                    k = str(sku_val).strip()
                    t = inv_map_sku.get(k) or inv_map_sku.get(normalize_sku(k)) or ""
                    if t:
                        return t
            except Exception:
                pass
            return str(fallback_desc or sku_val or "")

        try:
            df_inc["Producto (técnico)"] = df_inc.apply(
                lambda r: _title_tec_for_sku(r["SKU"], r["Descripción Control"]),
                axis=1,
            )
        except Exception:
            df_inc["Producto (técnico)"] = df_inc["SKU"].astype(str)

        # Orden similar a Admin Picking
        try:
            df_inc = df_inc[[
                "Mesa", "Venta", "Envío", "SKU", "Producto (técnico)",
                "Solicitado", "Verificado", "Estado", "Modo", "Hora"
            ]]
        except Exception:
            pass

        st.dataframe(df_inc, use_container_width=True, hide_index=True)
    else:
        st.info("Sin incidencias ni productos marcados como Sin EAN en este manifiesto.")

    # ---- Ventas pendientes ----
    st.divider()
    st.subheader("Ventas pendientes")

    pend = c.execute(
        "SELECT sale_id, mesa, shipment_id, status FROM s2_sales "
        "WHERE manifest_id=? AND status!='DONE' ORDER BY mesa, sale_id LIMIT 200;",
        (mid,),
    ).fetchall()

    if pend:
        pend_data = []
        for sale_id, mesa, shipment_id, status in pend:
            it = c.execute(
                "SELECT COUNT(*), SUM(CASE WHEN status IN ('DONE','INCIDENCE') THEN 1 ELSE 0 END) "
                "FROM s2_items WHERE manifest_id=? AND sale_id=?;",
                (mid, sale_id),
            ).fetchone()
            total = int(it[0] or 0)
            done = int(it[1] or 0)
            pend_data.append({
                "Venta": str(sale_id),
                "Mesa": int(mesa or 0),
                "Envío": str(shipment_id or ""),
                "Estado": str(status),
                "Items": f"{done}/{total}",
            })
        st.dataframe(pend_data, use_container_width=True, hide_index=True)
    else:
        st.success("No hay ventas pendientes: todo está cerrado.")

    # ---- Conciliación ----
    with st.expander("Conciliación ventas ↔ etiquetas", expanded=False):
        st.write({
            "Envíos únicos (labels)": stats.get("distinct_ship_labels"),
            "Ventas con Pack ID": stats.get("ventas_with_pack"),
            "Packs distintos (Control)": stats.get("distinct_packs"),
            "Etiquetas con Pack ID": stats.get("labels_with_pack"),
            "Etiquetas con Venta": stats.get("labels_with_sale"),
            "Ventas matcheadas por Pack": stats.get("matched_by_pack"),
            "Ventas sin Envío asignado": stats.get("missing_ship"),
        })

        missing = c.execute(
            "SELECT sale_id, page_no, pack_id FROM s2_sales "
            "WHERE manifest_id=? AND (shipment_id IS NULL OR shipment_id='') "
            "ORDER BY page_no, sale_id LIMIT 20",
            (mid,),
        ).fetchall()
        if missing:
            st.warning("Ejemplos de ventas sin envío asignado (primeras 20):")
            st.table([{"venta": a, "pagina": b, "pack_id": cpid or ""} for (a, b, cpid) in missing])

    # ---- Acciones (bloqueo duro + cierre + reinicio) ----
    st.divider()
    st.subheader("Acciones")
    st.caption("🔒 Bloqueo duro: para cargar un nuevo manifiesto debes **Cerrar** o **Reiniciar** el manifiesto activo.")

    close_ok = (
        int(stats.get("sales_total", 0) or 0) > 0
        and int(stats.get("sales_pending", 0) or 0) == 0
        and int(stats.get("items_pending", 0) or 0) == 0
    )
    btn_close = st.button("✅ Cerrar manifiesto (habilitar nuevo)", disabled=not close_ok)
    if not close_ok and int(stats.get("sales_total", 0) or 0) > 0:
        st.info(
            "Para cerrar el manifiesto: todas las **ventas** deben estar cerradas y no deben quedar **ítems pendientes**. "
            "Si necesitas cargar otro manifiesto sin terminar, usa **Reiniciar** (borra todo)."
        )

    if btn_close:
        _s2_close_manifest(mid)
        new_mid = _s2_create_new_manifest()
        for k in list(st.session_state.keys()):
            if k.startswith("s2_") or "sorting" in k:
                del st.session_state[k]
        st.success(f"Manifiesto {mid} cerrado. Nuevo manifiesto activo: {new_mid}")
        st.rerun()

    # Reinicio al final (como en Admin Picking)
    if "s2_reset_armed" not in st.session_state:
        st.session_state["s2_reset_armed"] = False

    arm = st.checkbox("Quiero reiniciar Sorting (entiendo que se borra todo)", value=st.session_state["s2_reset_armed"])
    st.session_state["s2_reset_armed"] = bool(arm)

    confirm_txt = st.text_input("Escribe BORRAR para confirmar", value="", disabled=not arm)
    do_reset = st.button(
        "🗑️ Reiniciar Sorting (borrar todo)",
        type="primary",
        disabled=not (arm and confirm_txt.strip().upper() == "BORRAR"),
    )

    if do_reset:
        _s2_reset_all_sorting()
        for k in list(st.session_state.keys()):
            if k.startswith("s2_") or "sorting" in k:
                del st.session_state[k]
        st.success("Sorting reiniciado completamente.")
        st.rerun()

    conn.close()

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
# CONTADOR DE PAQUETES (Flex/Colecta)
# =========================
def _pkg_norm_label(raw: str) -> str:
    r = str(raw or "").strip()
    d = only_digits(r)
    return d if d else r

def _pkg_get_open_run(kind: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, created_at FROM pkg_counter_runs WHERE kind=? AND status='OPEN' ORDER BY id DESC LIMIT 1;",
        (str(kind),),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"id": int(row[0]), "created_at": row[1]}

def _pkg_create_run(kind: str) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO pkg_counter_runs (kind, status, created_at) VALUES (?, 'OPEN', ?);",
        (str(kind), now_iso()),
    )
    rid = int(c.lastrowid)
    conn.commit()
    conn.close()
    return rid

def _pkg_close_run(run_id: int):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE pkg_counter_runs SET status='DONE', closed_at=? WHERE id=?;", (now_iso(), int(run_id)))
    conn.commit()
    conn.close()

def _pkg_run_count(run_id: int) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(1) FROM pkg_counter_scans WHERE run_id=?;", (int(run_id),))
    n = int(c.fetchone()[0] or 0)
    conn.close()
    return n

def _pkg_last_scans(run_id: int, limit: int = 15):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT label_key, scanned_at FROM pkg_counter_scans WHERE run_id=? ORDER BY id DESC LIMIT ?;",
        (int(run_id), int(limit)),
    )
    rows = c.fetchall()
    conn.close()
    return rows

def _pkg_register_scan(run_id: int, label_key: str, raw: str):
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO pkg_counter_scans (run_id, label_key, raw, scanned_at) VALUES (?, ?, ?, ?);",
            (int(run_id), str(label_key), str(raw or ""), now_iso()),
        )
        conn.commit()
        return True, None
    except Exception as e:
        # SQLite lanza error por UNIQUE(run_id,label_key) => repetido
        msg = str(e).lower()
        if "unique" in msg or "constraint" in msg:
            return False, "DUP"
        return False, str(e)
    finally:
        conn.close()

def _pkg_reset_kind(kind: str):
    """Borra historial COMPLETO de ese tipo (Flex/Colecta): runs + scans."""
    conn = get_conn()
    c = conn.cursor()
    # obtener runs
    c.execute("SELECT id FROM pkg_counter_runs WHERE kind=?;", (str(kind),))
    rids = [int(r[0]) for r in c.fetchall()]
    if rids:
        qmarks = ",".join(["?"] * len(rids))
        c.execute(f"DELETE FROM pkg_counter_scans WHERE run_id IN ({qmarks});", tuple(rids))
    c.execute("DELETE FROM pkg_counter_runs WHERE kind=?;", (str(kind),))
    conn.commit()
    conn.close()

def page_pkg_counter():
    st.header("🧮 Contador de paquetes")
    st.caption("Escanea para contar. Si repites una etiqueta, se avisa y NO suma.")

    KIND = "GENERAL"

    def ensure_run() -> dict:
        run = _pkg_get_open_run(KIND)
        if not run:
            rid = _pkg_create_run(KIND)
            run = {"id": rid, "created_at": now_iso()}
        return run

    def handle_scan(input_key: str):
        raw = str(st.session_state.get(input_key, "") or "").strip()
        if not raw:
            return

        run = ensure_run()
        run_id = int(run["id"])

        label_key = _pkg_norm_label(raw)
        if not label_key:
            st.session_state["pkg_flash"] = ("err", "Etiqueta inválida.")
            _sfx_trigger("ERROR")
            st.session_state[input_key] = ""
            return

        ok, err = _pkg_register_scan(run_id, label_key, raw)
        if ok:
            st.session_state["pkg_flash"] = ("ok", "OK")
            _sfx_trigger("OK")
        else:
            if err == "DUP":
                st.session_state["pkg_flash"] = ("dup", f"Repetida: {label_key}")
                _sfx_trigger("DUPLICADO")
            else:
                st.session_state["pkg_flash"] = ("err", "Error al registrar")
                _sfx_trigger("ERROR")

        # dejar el campo en blanco para el siguiente escaneo
        st.session_state[input_key] = ""

    # Reinicio sin confirmación (manejado antes de crear el widget para evitar StreamlitAPIException)
    if st.session_state.pop("pkg_reset_trigger", False):
        _pkg_reset_kind(KIND)
        _ = _pkg_create_run(KIND)
        # limpiar input del lector antes de renderizar el widget
        try:
            if "pkg_scan_input" in st.session_state:
                del st.session_state["pkg_scan_input"]
        except Exception:
            pass
        st.rerun()

    # asegura corrida activa
    run = ensure_run()
    run_id = int(run["id"])

    # aviso minimalista (una vez)
    if "pkg_flash" in st.session_state:
        k, msg = st.session_state.get("pkg_flash", ("info", ""))
        if msg:
            if k == "ok":
                st.success(msg)
            elif k == "dup":
                st.warning(msg)
            else:
                st.error(msg)
        st.session_state.pop("pkg_flash", None)

    total = _pkg_run_count(run_id)
    st.metric("Paquetes contabilizados", total)

    # Escaneo automático (sin botones)
    input_key = "pkg_scan_input"
    st.text_input(
        "Escaneo (lector)",
        key=input_key,
        on_change=handle_scan,
        args=(input_key,),
    )
    force_tel_keyboard("Escaneo (lector)")
    autofocus_input("Escaneo (lector)")

    # Últimos escaneos
    rows = _pkg_last_scans(run_id, 15)
    if rows:
        df_last = pd.DataFrame(rows, columns=["Etiqueta", "Hora"])
        df_last["Hora"] = df_last["Hora"].apply(to_chile_display)
        st.dataframe(df_last, use_container_width=True, hide_index=True)
    else:
        st.info("Aún no hay paquetes en esta corrida.")

    # Única acción
    if st.button("🔄 Reiniciar corrida", use_container_width=True, key="pkg_reset_now"):
        st.session_state["pkg_reset_trigger"] = True
        st.rerun()

def main():

    st.set_page_config(page_title="Aurora ML – WMS", layout="wide")
    init_db()

    # SFX global (se dispara con _sfx_trigger en los módulos)
    _sfx_render()

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
            "3) Cortes de la tanda (PDF)",
            "4) Administrador",
        ]
        page = st.sidebar.radio("Menú", pages, index=0)

        if page.startswith("1"):
            page_picking()
        elif page.startswith("2"):
            page_import(inv_map_sku)
        elif page.startswith("3"):
            page_cortes_pdf_batch()
        else:
            page_admin()

    elif mode == "SORTING":
        pages = [
            "1) Camarero",
            "2) Cargar manifiesto y asignar mesas",
            "3) Administrador",
        ]
        page = st.sidebar.radio("Menú", pages, index=0)

        if page.startswith("1"):
            page_sorting_camarero(inv_map_sku, barcode_to_sku)
        elif page.startswith("2"):
            page_sorting_upload(inv_map_sku, barcode_to_sku)
        else:
            page_sorting_admin(inv_map_sku, barcode_to_sku)

    # ==========
    # MODO FULL (nuevo módulo completo)
    # ==========
    elif mode == "PKG_COUNT":
        pages = [
            "1) Contador de paquetes",
        ]
        _ = st.sidebar.radio("Menú", pages, index=0)
        page_pkg_counter()

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