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

    c.execute("""
    CREATE TABLE IF NOT EXISTS sku_barcodes (
        barcode TEXT PRIMARY KEY,
        sku_ml TEXT
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
# IMPORTAR VENTAS
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
# UI: LOBBY APP (NUEVO)
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
        .lobbywrap { max-width: 860px; margin: 0 auto; }
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
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Picking por OT, incidencias, admin, etc.")

    with colB:
        st.markdown('<div class="lobbybtn">', unsafe_allow_html=True)
        if st.button("🏷️ Preparación productos Full", key="mode_full"):
            st.session_state.app_mode = "FULL"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Nuevo módulo (control de acopio Full).")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# UI: IMPORTAR
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
# UI: PICKING (LOBBY + PDA)
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
# UI: FULL (placeholder por ahora)
# =========================
def page_full_placeholder():
    st.header("Preparación productos Full")
    st.info("Módulo en desarrollo: Control de acopio Full (escaneo + chequeo vs Excel).")
    st.write("Siguiente paso: aquí agregamos carga del Excel Full + pantalla supervisor.")


# =========================
# UI: ADMIN
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
    # MODO FULL (nuevo módulo)
    # ==========
    else:
        pages = [
            "1) Full – Preparación",
        ]
        page = st.sidebar.radio("Menú", pages, index=0)
        page_full_placeholder()


if __name__ == "__main__":
    main()
