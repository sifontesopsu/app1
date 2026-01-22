import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from io import BytesIO
import re
import math

# PDF manifiestos
try:
    import pdfplumber
    HAS_PDF_LIB = True
except ImportError:
    HAS_PDF_LIB = False

# PDF (reportlab) - se mantiene por si lo reactivan más adelante
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.graphics.barcode import code128

DB_NAME = "aurora_ml.db"
ADMIN_PASSWORD = "aurora123"  # cambia si quieres
NUM_MESAS = 4
VENTAS_POR_HOJA = 10


# ----------------- UTILIDADES -----------------
def now_iso():
    return datetime.now().isoformat(timespec="seconds")


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
    """Soporta múltiples códigos en una celda, separados por espacio/coma/; o saltos."""
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
    # únicos preservando orden
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


# ----------------- BASE DE DATOS -----------------
def _add_column_if_missing(c, table, col_name, col_def):
    c.execute(f"PRAGMA table_info({table});")
    cols = [r[1] for r in c.fetchall()]
    if col_name not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def};")


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
        status TEXT,              -- OPEN / PICKED
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
        status TEXT DEFAULT 'PENDING',   -- PENDING / DONE / INCIDENCE
        decided_at TEXT,
        confirm_mode TEXT              -- SCAN / MANUAL_NO_EAN
    );
    """)
    _add_column_if_missing(c, "picking_tasks", "confirm_mode", "TEXT")

    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_incidences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ot_id INTEGER,
        sku_ml TEXT,
        qty_total INTEGER,
        qty_picked INTEGER,
        qty_missing INTEGER,
        reason TEXT,             -- FALTANTE
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

    # Se mantiene aunque sorting esté oculto
    c.execute("""
    CREATE TABLE IF NOT EXISTS sorting_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ot_id INTEGER,
        order_id INTEGER,
        status TEXT,           -- PENDING / READY
        marked_at TEXT,
        mesa INTEGER,
        printed_at TEXT
    );
    """)
    _add_column_if_missing(c, "sorting_status", "mesa", "INTEGER")
    _add_column_if_missing(c, "sorting_status", "printed_at", "TEXT")

    c.execute("""
    CREATE TABLE IF NOT EXISTS sku_barcodes (
        barcode TEXT PRIMARY KEY,
        sku_ml TEXT
    );
    """)

    conn.commit()
    conn.close()


# ----------------- PARSER PDF MANIFIESTO -----------------
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


# ----------------- CARGA MAESTRO (SKU + BARCODE) -----------------
def load_master(inv_file) -> tuple[dict, dict, list]:
    inv_map_sku = {}
    barcode_to_sku = {}
    conflicts = []

    if inv_file is None:
        return inv_map_sku, barcode_to_sku, conflicts

    df = pd.read_excel(inv_file, dtype=str)
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
    for cand in ["codigo de barras", "código de barras", "barcode"]:
        if cand in lower:
            barcode_col = cols[lower.index(cand)]
            break

    # fallback (sin headers)
    if sku_col is None or tech_col is None:
        df0 = pd.read_excel(inv_file, header=None, dtype=str)
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
            barcode_col = None

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


# ----------------- IMPORTAR VENTAS -----------------
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

    # reinicia corrida operativa
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

    # pickers + ots
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

    # repartir ventas equitativas por orden
    unique_orders = sales_df[["ml_order_id"]].drop_duplicates().reset_index(drop=True)
    assignments = {}
    for idx, row in unique_orders.iterrows():
        ot_id = ot_ids[idx % len(ot_ids)]
        assignments[str(row["ml_order_id"]).strip()] = ot_id

    # mesa round robin (aunque sorting esté oculto)
    for idx, (ml_order_id, ot_id) in enumerate(assignments.items()):
        order_id = order_id_by_ml[ml_order_id]
        mesa = (idx % NUM_MESAS) + 1
        c.execute("INSERT INTO ot_orders (ot_id, order_id) VALUES (?,?)", (ot_id, order_id))
        c.execute("""
            INSERT INTO sorting_status (ot_id, order_id, status, marked_at, mesa, printed_at)
            VALUES (?,?,?,?,?,?)
        """, (ot_id, order_id, "PENDING", None, mesa, None))

    # tasks por OT agrupado por SKU (ruta ordenada)
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


# ----------------- UI: IMPORTAR -----------------
def page_import():
    st.header("1) Importar ventas")

    origen = st.radio("Origen", ["Excel Mercado Libre", "Manifiesto PDF (etiquetas)"], horizontal=True)
    num_pickers = st.number_input("Cantidad de pickeadores", min_value=1, max_value=20, value=5, step=1)

    st.subheader("Maestro de SKUs y nombres técnicos (opcional, recomendado)")
    st.caption("Soporta columna SKU y (opcional) columna 'Codigo de barras' con múltiples códigos por celda.")
    inv_file = st.file_uploader("Maestro SKU (xlsx)", type=["xlsx"], key="inv_master")

    inv_map_sku, barcode_to_sku, conflicts = load_master(inv_file)
    if inv_file is not None:
        st.success(f"Maestro cargado: {len(inv_map_sku)} SKUs con nombre técnico, {len(barcode_to_sku)} barcodes.")
        if conflicts:
            st.warning("Conflictos de barcode detectados (barcode asignado a más de 1 SKU). Se usará el primero:")
            st.dataframe(pd.DataFrame(conflicts, columns=["Barcode", "SKU existente", "SKU nuevo"]).head(50))
        upsert_barcodes_to_db(barcode_to_sku)

    sales_df = None

    if origen == "Excel Mercado Libre":
        file = st.file_uploader("Ventas ML (xlsx)", type=["xlsx"], key="ml_excel")
        if not file:
            st.info("Sube el Excel de ventas.")
            return
        try:
            sales_df = import_sales_excel(file)
        except Exception as e:
            st.error(f"Error leyendo Excel: {e}")
            return
    else:
        pdf_file = st.file_uploader("Manifiesto PDF", type=["pdf"], key="ml_pdf")
        if not pdf_file:
            st.info("Sube el PDF manifiesto.")
            return
        try:
            sales_df = parse_manifest_pdf(pdf_file)
        except Exception as e:
            st.error(f"No se pudo procesar el PDF: {e}")
            return

    if sales_df is not None:
        st.subheader("Vista previa ventas")
        st.dataframe(sales_df.head(30))

        if st.button("Cargar y generar OTs"):
            save_orders_and_build_ots(sales_df, inv_map_sku, int(num_pickers))
            st.success("OTs creadas. Ya puedes ir a Picking.")


# ----------------- UI: PICKING (PDA) -----------------
def page_picking():
    st.header("2) Picking PDA por OT (1 escaneo + cantidad digitada + decisión)")

    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT barcode, sku_ml FROM sku_barcodes")
    barcode_to_sku = {r[0]: r[1] for r in c.fetchall()}

    c.execute("""
        SELECT po.id, po.ot_code, pk.name, po.status
        FROM picking_ots po
        JOIN pickers pk ON pk.id = po.picker_id
        ORDER BY po.ot_code
    """)
    ots = c.fetchall()
    if not ots:
        st.info("No hay OTs. Importa ventas primero.")
        conn.close()
        return

    labels = [f"{ot_code} – {picker} – {status}" for (ot_id, ot_code, picker, status) in ots]
    sel = st.selectbox("Selecciona tu OT", labels, index=0)
    ot_id = ots[labels.index(sel)][0]

    c.execute("SELECT ot_code, status FROM picking_ots WHERE id=?", (ot_id,))
    ot_code, ot_status = c.fetchone()

    if ot_status == "PICKED":
        st.success("Esta OT ya está cerrada (PICKED).")
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
    st.caption(f"Resueltos (DONE/INCIDENCE): {done_small}/{total_tasks}")

    current = next((t for t in tasks if t[5] == "PENDING"), None)
    if current is None:
        st.success("No quedan SKUs pendientes. Puedes cerrar la OT.")
        if st.button("Cerrar OT"):
            c.execute("UPDATE picking_ots SET status='PICKED', closed_at=? WHERE id=?", (now_iso(), ot_id))
            conn.commit()
            st.success("OT cerrada.")
        conn.close()
        return

    task_id, sku_expected, producto, qty_total, qty_picked, status = current

    st.markdown(
        """
        <style>
        .hero { padding: 12px 14px; border-radius: 12px; background: rgba(0,0,0,0.04); margin-bottom: 12px; }
        .hero .sku { font-size: 30px; font-weight: 900; }
        .hero .prod { font-size: 28px; font-weight: 800; margin-top: 4px; }
        .hero .qty { font-size: 34px; font-weight: 900; margin-top: 8px; }
        .smallcap { font-size: 12px; opacity: 0.75; }
        </style>
        """,
        unsafe_allow_html=True
    )

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
            "show_manual_confirm": False
        }
    s = state[str(task_id)]

    st.divider()

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        scan = st.text_input("Escanea SKU / Código de barras", value=s["scan_value"], key=f"scan_{task_id}")

    with col2:
        if st.button("Validar escaneo"):
            sku_detected = resolve_scan_to_sku(scan, barcode_to_sku)
            if not sku_detected:
                st.error("No se pudo interpretar el escaneo.")
                s["confirmed"] = False
                s["confirm_mode"] = None
            elif sku_detected != sku_expected:
                st.error(f"Escaneo corresponde a SKU {sku_detected}, pero el producto actual es {sku_expected}.")
                s["confirmed"] = False
                s["confirm_mode"] = None
            else:
                s["confirmed"] = True
                s["confirm_mode"] = "SCAN"
                s["scan_value"] = scan
                s["show_manual_confirm"] = False
                st.success("Producto validado por escaneo. Ingresa la cantidad pickeada.")
                st.rerun()

    with col3:
        if st.button("Producto sin EAN"):
            # muestra confirmación manual en pantalla
            s["show_manual_confirm"] = True

    if s.get("show_manual_confirm", False) and not s["confirmed"]:
        st.info("Confirmación manual (sin EAN)")
        st.write(f"✅ **Producto:** {producto}")
        if st.button("Confirmar manual", key=f"confirm_manual_{task_id}"):
            s["confirmed"] = True
            s["confirm_mode"] = "MANUAL_NO_EAN"
            s["show_manual_confirm"] = False
            st.success("Producto confirmado manualmente. Ingresa la cantidad.")
            # IMPORTANTE: rerun para que el input de cantidad aparezca habilitado
            st.rerun()

    qty_in = st.text_input(
        "Cantidad pickeada (digitada)",
        value=s["qty_input"],
        disabled=not s["confirmed"],
        key=f"qty_{task_id}"
    )

    if st.button("Confirmar cantidad", disabled=not s["confirmed"]):
        try:
            q = int(str(qty_in).strip())
        except Exception:
            st.error("Ingresa un número válido.")
            q = None

        if q is not None:
            s["qty_input"] = str(q)

            if q > int(qty_total):
                st.error(f"La cantidad ingresada ({q}) supera la solicitada ({qty_total}). Corrige el valor.")
                s["needs_decision"] = False

            elif q == int(qty_total):
                c.execute("""
                    UPDATE picking_tasks
                    SET qty_picked=?, status='DONE', decided_at=?, confirm_mode=?
                    WHERE id=?
                """, (q, now_iso(), s["confirm_mode"], task_id))
                conn.commit()
                state.pop(str(task_id), None)
                st.success("Producto completado. Pasando al siguiente…")
                st.rerun()

            else:
                missing = int(qty_total) - q
                s["needs_decision"] = True
                s["missing"] = missing
                st.warning(f"Faltan {missing} unidades. Debes decidir: incidencias o reintentar. No puedes avanzar sin decisión.")

    if s["needs_decision"]:
        st.error(f"DECISIÓN OBLIGATORIA: faltan {s['missing']} unidades.")
        colA, colB = st.columns(2)

        with colA:
            if st.button("Enviar a incidencias y continuar"):
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
                st.success("Enviado a incidencias. Continuando…")
                st.rerun()

        with colB:
            if st.button("Reintentar (seguir buscando)"):
                s["needs_decision"] = False
                st.info("Reintentar: ajusta la cantidad y vuelve a confirmar.")

    conn.close()


# ----------------- UI: ADMIN -----------------
def page_admin():
    st.header("3) Administrador")

    pwd = st.text_input("Contraseña", type="password")
    if pwd != ADMIN_PASSWORD:
        st.info("Ingresa contraseña para administrar.")
        return

    conn = get_conn()
    c = conn.cursor()

    st.subheader("Resumen general")
    c.execute("SELECT COUNT(*) FROM orders")
    n_orders = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM order_items")
    n_items = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM picking_ots")
    n_ots = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM picking_incidences")
    n_inc = c.fetchone()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ventas (histórico)", n_orders)
    col2.metric("Líneas (histórico)", n_items)
    col3.metric("OTs (corrida)", n_ots)
    col4.metric("Incidencias (corrida)", n_inc)

    st.subheader("Estado de OTs")
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
        "SKUs pendientes", "SKUs resueltos", "Confirmaciones sin EAN"
    ])
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
        st.dataframe(df_inc)
    else:
        st.info("Sin incidencias registradas en la corrida actual.")

    st.divider()
    st.subheader("Acciones")
    if st.button("Reiniciar corrida operativa (borra OTs, tasks, sorting, incidencias; mantiene histórico ventas)"):
        c.execute("DELETE FROM picking_tasks;")
        c.execute("DELETE FROM picking_incidences;")
        c.execute("DELETE FROM sorting_status;")
        c.execute("DELETE FROM ot_orders;")
        c.execute("DELETE FROM picking_ots;")
        c.execute("DELETE FROM pickers;")
        conn.commit()
        st.success("Corrida reiniciada.")
        st.rerun()

    conn.close()


# ----------------- MAIN -----------------
def main():
    st.set_page_config(page_title="Aurora ML – WMS Picking", layout="wide")
    init_db()

    st.sidebar.title("Ferretería Aurora – WMS")

    # Sorting oculto por ahora
    page = st.sidebar.radio("Menú", [
        "1) Importar ventas",
        "2) Picking PDA (por OT)",
        "3) Administrador"
    ])

    if page.startswith("1"):
        page_import()
    elif page.startswith("2"):
        page_picking()
    else:
        page_admin()


if __name__ == "__main__":
    main()
