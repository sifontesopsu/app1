import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sqlite3
from datetime import datetime
import re

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
    # Guardamos naive ISO (server suele ser UTC en Streamlit Cloud)
    return datetime.now().isoformat(timespec="seconds")


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
        confirm_mode TEXT
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
    df = pd.read_excel(file, header=[4, 5])
    df.columns = [" | ".join([str(x) for x in col if str(x) != "nan"]) for col in df.columns]

    COLUMN_ORDER_ID = "Ventas | # de venta"
    COLUMN_QTY = "Ventas | Unidades"
    COLUMN_SKU = "Publicaciones | SKU"
    COLUMN_TITLE = "Publicaciones | Título de la publicación"
    COLUMN_BUYER = "Compradores | Comprador"

    required = [COLUMN_ORDER_ID, COLUMN_QTY, COLUMN_SKU, COLUMN_TITLE, COLUMN_BUYER]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    work = df[[COLUMN_ORDER_ID, COLUMN_QTY, COLUMN_SKU, COLUMN_TITLE, COLUMN_BUYER]].copy()
    work.columns = ["ml_order_id", "qty", "sku_ml", "title_ml", "buyer"]
    work["qty"] = pd.to_numeric(work["qty"], errors="coerce").fillna(0).astype(int)
    work = work[work["qty"] > 0]
    work["sku_ml"] = work["sku_ml"].apply(normalize_sku)
    return work[["ml_order_id", "buyer", "sku_ml", "title_ml", "qty"]]


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
        .lobbywrap { max-width: 980px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="lobbywrap">', unsafe_allow_html=True)
    colA, colB = st.columns(2)

    with colA:
        st.markdown('<div class="lobbybtn">', unsafe_allow_html=True)
        if st.button("📦 Preparación pedidos Flex y Colecta", key="mode_flex"):
            st.session_state.app_mode = "FLEX"
            st.session_state.pop("selected_picker", None)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Picking por OT, incidencias, admin, etc.")

    with colB:
        st.markdown('<div class="lobbybtn">', unsafe_allow_html=True)
        if st.button("🏷️ Preparación productos Full", key="mode_full"):
            st.session_state.app_mode = "FULL"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Control de acopio Full (escaneo + chequeo vs Excel).")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# UI: IMPORTAR (FLEX)
# =========================
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
        SELECT id, sku_ml, COALESCE(NULLIF(title_tec,''), title_ml) AS producto,
               qty_total, qty_picked, status
        FROM picking_tasks
        WHERE ot_id=?
        ORDER BY CAST(sku_ml AS INTEGER), sku_ml
    """, (ot_id,))
    tasks = c.fetchall()

    total_tasks = len(tasks)
    done_small = sum(1 for t in tasks if t[5] in ("DONE", "INCIDENCE"))
    st.caption(f"Resueltos: {done_small}/{total_tasks}")

    current = next((t for t in tasks if t[5] == "PENDING"), None)
    if current is None:
        st.success("No quedan SKUs pendientes.")
        if st.button("Cerrar OT"):
            c.execute("UPDATE picking_ots SET status='PICKED', closed_at=? WHERE id=?", (now_iso(), ot_id))
            conn.commit()
            st.success("OT cerrada.")
        conn.close()
        return

    task_id, sku_expected, producto, qty_total, qty_picked, status = current

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

    st.markdown(
        f"""
        <div class="hero">
            <div class="smallcap">OT: {ot_code}</div>
            <div class="sku">SKU: {sku_expected}</div>
            <div class="prod">{producto}</div>
            <div class="qty">Solicitado: {qty_total}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if s["scan_status"] == "ok":
        st.markdown(f'<span class="scanok ok">✅ OK</span> {s["scan_msg"]}', unsafe_allow_html=True)
    elif s["scan_status"] == "bad":
        st.markdown(f'<span class="scanok bad">❌ ERROR</span> {s["scan_msg"]}', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])

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

    if s.get("show_manual_confirm", False) and not s["confirmed"]:
        st.info("Confirmación manual")
        st.write(f"✅ {producto}")
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
        state[str(batch_id)] = {"scan_value": "", "sku_current": "", "qty_input": "", "msg": "", "msg_kind": "idle"}

    sst = state[str(batch_id)]

    scan_label = "Escaneo"
    scan = st.text_input(scan_label, value=sst.get("scan_value", ""), key=f"full_scan_{batch_id}")
    force_tel_keyboard(scan_label)
    autofocus_input(scan_label)

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("🔎 Buscar / Validar", key=f"full_find_{batch_id}"):
            sku = resolve_scan_to_sku(scan, barcode_to_sku)
            sst["scan_value"] = scan
            sst["sku_current"] = sku
            sst["qty_input"] = ""

            if not sku:
                sst["msg_kind"] = "bad"
                sst["msg"] = "No se pudo leer el código."
                st.rerun()

            # Validar si existe en el lote
            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                SELECT sku_ml, COALESCE(NULLIF(title,''),''), qty_required, qty_checked
                FROM full_batch_items
                WHERE batch_id=? AND sku_ml=?
            """, (batch_id, sku))
            row = c.fetchone()
            conn.close()

            if not row:
                sst["msg_kind"] = "bad"
                sst["msg"] = f"{sku} no pertenece a este lote."
                sst["sku_current"] = ""
            else:
                sst["msg_kind"] = "ok"
                sst["msg"] = "SKU encontrado."
            st.rerun()

    with colB:
        if st.button("🧹 Limpiar", key=f"full_clear_{batch_id}"):
            sst["scan_value"] = ""
            sst["sku_current"] = ""
            sst["qty_input"] = ""
            sst["msg_kind"] = "idle"
            sst["msg"] = ""
            try:
                st.session_state[f"full_scan_{batch_id}"] = ""
                st.session_state[f"full_qty_{batch_id}"] = ""
            except Exception:
                pass
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
        SELECT sku_ml, COALESCE(NULLIF(title,''),''), qty_required, qty_checked
        FROM full_batch_items
        WHERE batch_id=? AND sku_ml=?
    """, (batch_id, sku_cur))
    row = c.fetchone()
    conn.close()

    if not row:
        st.warning("El SKU no está en el lote (vuelve a validar).")
        return

    sku_db, title_db, qty_req, qty_chk = row
    title_clean = str(title_db or "").strip()
    # Seguro por si por algún motivo llega como Series u otro objeto raro
    if hasattr(title_db, "iloc"):
        try:
            title_clean = str(title_db.iloc[0]).strip()
        except Exception:
            title_clean = str(title_db).strip()
    if not title_clean:
        title_clean = inv_map_sku.get(sku_db, "")

    pending = int(qty_req) - int(qty_chk)
    if pending < 0:
        pending = 0

    st.markdown(
        f"""
        <div class="hero2">
            <div class="sku">SKU: {sku_db}</div>
            <div class="prod">{title_clean}</div>
            <div class="qty">Solicitado: {int(qty_req)} • Acopiado: {int(qty_chk)} • Pendiente: {pending}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    qty_label = "Cantidad a acopiar"
    qty_in = st.text_input(qty_label, value=sst.get("qty_input",""), key=f"full_qty_{batch_id}")
    force_tel_keyboard(qty_label)

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

            conn = get_conn()
            c = conn.cursor()
            c.execute("""
                UPDATE full_batch_items
                SET qty_checked = COALESCE(qty_checked,0) + ?
                WHERE batch_id=? AND sku_ml=?
            """, (q, batch_id, sku_db))
            conn.commit()
            conn.close()

            # Limpiar para siguiente escaneo
            sst["scan_value"] = ""
            sst["sku_current"] = ""
            sst["qty_input"] = ""
            sst["msg_kind"] = "idle"
            sst["msg"] = ""
            try:
                st.session_state[f"full_scan_{batch_id}"] = ""
                st.session_state[f"full_qty_{batch_id}"] = ""
            except Exception:
                pass

            st.success(f"Acopiado: {q} unidades.")
            st.rerun()

    with colD:
        # Duplicamos "Limpiar" abajo por ergonomía
        if st.button("🧹 Limpiar campos", key=f"full_clear2_{batch_id}"):
            sst["scan_value"] = ""
            sst["sku_current"] = ""
            sst["qty_input"] = ""
            sst["msg_kind"] = "idle"
            sst["msg"] = ""
            try:
                st.session_state[f"full_scan_{batch_id}"] = ""
                st.session_state[f"full_qty_{batch_id}"] = ""
            except Exception:
                pass
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
    st.dataframe(df)

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

    mode = st.session_state.get("app_mode", "FLEX")

    # ==========
    # MODO FLEX / COLECTA (lo actual)
    # ==========
    if mode == "FLEX":
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
