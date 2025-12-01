import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from io import BytesIO
import re  # para parsing PDF y normalizar SKUs

# ========= CÓDIGOS DE BARRAS =========
HAS_BARCODE_LIB = False
try:
    from barcode import Code128
    from barcode.writer import ImageWriter
    HAS_BARCODE_LIB = True
except ImportError:
    HAS_BARCODE_LIB = False

# PDF desde manifiestos
try:
    import pdfplumber
    HAS_PDF_LIB = True
except ImportError:
    HAS_PDF_LIB = False

# PDF de hojas de picking (reportlab)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

DB_NAME = "aurora_ml.db"
ADMIN_PASSWORD = "aurora123"  # 🔐 cámbiala si quieres


# ---------- NORMALIZAR SKU ----------
def normalize_sku(value) -> str:
    """
    Normaliza un SKU para que maestro y ventas calcen:
      - Convierte a string
      - Quita espacios
      - Elimina sufijo '.0'
      - Convierte notación científica a entero (si aplica)
    """
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    # quitar .0 típico de floats
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    # notación científica (ej: 1.80201401001E11)
    if re.fullmatch(r"\d+(\.\d+)?[eE][+-]?\d+", s):
        try:
            s = str(int(float(s)))
        except Exception:
            pass
    return s


# ---------- HELPERS DB ----------
def get_conn():
    return sqlite3.connect(DB_NAME, check_same_thread=False)


def init_db():
    conn = get_conn()
    c = conn.cursor()

    # Pedidos ML
    c.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ml_order_id TEXT,
        buyer TEXT,
        created_at TEXT
    );
    """)

    # Líneas de cada pedido
    c.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        sku_ml TEXT,
        mlc_id TEXT,
        title_ml TEXT,
        title_tec TEXT,
        qty INTEGER
    );
    """)

    # Picking global por SKU / MLC
    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_global (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku_ml TEXT,
        mlc_id TEXT,
        title_ml TEXT,
        title_tec TEXT,
        qty_total INTEGER,
        qty_picked INTEGER DEFAULT 0,
        picker_id INTEGER,
        ot_id INTEGER
    );
    """)

    # Escaneos finales (no usados ahora)
    c.execute("""
    CREATE TABLE IF NOT EXISTS packages_scan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracking_code TEXT,
        scanned_at TEXT
    );
    """)

    # Piqueadores
    c.execute("""
    CREATE TABLE IF NOT EXISTS pickers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    );
    """)

    # OTs
    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_ots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        picker_id INTEGER,
        ot_code TEXT UNIQUE,
        created_at TEXT
    );
    """)

    # Asegurar columnas nuevas
    c.execute("PRAGMA table_info(order_items);")
    cols_oi = [row[1] for row in c.fetchall()]
    if "mlc_id" not in cols_oi:
        c.execute("ALTER TABLE order_items ADD COLUMN mlc_id TEXT;")
    if "title_tec" not in cols_oi:
        c.execute("ALTER TABLE order_items ADD COLUMN title_tec TEXT;")

    c.execute("PRAGMA table_info(picking_global);")
    cols_pg = [row[1] for row in c.fetchall()]
    if "mlc_id" not in cols_pg:
        c.execute("ALTER TABLE picking_global ADD COLUMN mlc_id TEXT;")
    if "title_tec" not in cols_pg:
        c.execute("ALTER TABLE picking_global ADD COLUMN title_tec TEXT;")
    if "picker_id" not in cols_pg:
        c.execute("ALTER TABLE picking_global ADD COLUMN picker_id INTEGER;")
    if "ot_id" not in cols_pg:
        c.execute("ALTER TABLE picking_global ADD COLUMN ot_id INTEGER;")

    # Tabla de imágenes por MLC
    c.execute("""
    CREATE TABLE IF NOT EXISTS sku_images (
        mlc_id TEXT PRIMARY KEY,
        image_url TEXT
    );
    """)

    conn.commit()
    conn.close()


# ---------- CÓDIGOS DE BARRAS ----------
def generate_barcode_bytes(data: str):
    """Genera un Code128 en memoria y devuelve los bytes de la imagen."""
    if not HAS_BARCODE_LIB:
        return None
    rv = BytesIO()
    Code128(data, writer=ImageWriter()).write(rv)
    return rv.getvalue()


# ---------- PDF HOJAS DE PICKING ----------
def build_picklist_pdf(ot_data_list):
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    margin_left = 20 * mm
    margin_right = 190 * mm
    margin_top = 20 * mm
    margin_bottom = 20 * mm

    col_sku = margin_left
    col_prod = 60 * mm
    col_qty = 170 * mm
    table_width = margin_right - margin_left

    header_height = 8 * mm
    row_height = 6 * mm

    for idx, ot in enumerate(ot_data_list):
        if idx > 0:
            c.showPage()

        ot_code = ot["ot_code"]
        picker_name = ot["picker_name"]
        created_at = ot["created_at"]
        items = ot["items"]

        # Encabezado
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin_left, height - margin_top, f"OT: {ot_code}")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin_left, height - margin_top - 6 * mm, f"Piqueador: {picker_name}")
        c.setFont("Helvetica", 10)
        c.drawString(margin_left, height - margin_top - 11 * mm, f"Creada: {created_at}")

        y = height - margin_top - 25 * mm

        # Código de barras
        img_bytes = generate_barcode_bytes(ot_code)
        if img_bytes is not None:
            try:
                img = ImageReader(BytesIO(img_bytes))
                c.drawImage(
                    img,
                    margin_left,
                    y - 20 * mm,
                    width=60 * mm,
                    preserveAspectRatio=True,
                    mask='auto'
                )
                y -= 32 * mm
            except Exception:
                y -= 10 * mm
        else:
            y -= 5 * mm

        def draw_header_row(y_top):
            c.setFont("Helvetica-Bold", 9)
            c.rect(margin_left, y_top - header_height, table_width, header_height, stroke=1, fill=0)
            c.line(col_prod, y_top - header_height, col_prod, y_top)
            c.line(col_qty, y_top - header_height, col_qty, y_top)
            c.drawString(col_sku + 2 * mm, y_top - header_height + 2 * mm, "SKU")
            c.drawString(col_prod + 2 * mm, y_top - header_height + 2 * mm, "Producto")
            c.drawString(col_qty + 2 * mm, y_top - header_height + 2 * mm, "Cant.")
            return y_top - header_height

        y = draw_header_row(y)
        c.setFont("Helvetica", 9)

        for sku, producto, qty in items:
            if y - row_height < margin_bottom:
                c.showPage()
                c.setFont("Helvetica-Bold", 12)
                c.drawString(margin_left, height - margin_top, f"OT: {ot_code} (continuación)")
                c.setFont("Helvetica", 10)
                c.drawString(margin_left, height - margin_top - 6 * mm, f"Piqueador: {picker_name}")
                y = height - margin_top - 12 * mm
                y = draw_header_row(y)
                c.setFont("Helvetica", 9)

            y_row_bottom = y - row_height
            c.rect(margin_left, y_row_bottom, table_width, row_height, stroke=1, fill=0)
            c.line(col_prod, y_row_bottom, col_prod, y)
            c.line(col_qty, y_row_bottom, col_qty, y)

            # Truncar producto para que no tape la cantidad
            prod_text = str(producto)
            max_len = 60
            if len(prod_text) > max_len:
                prod_text = prod_text[:max_len - 3] + "..."

            c.drawString(col_sku + 2 * mm, y_row_bottom + 2 * mm, str(sku)[:20])
            c.drawString(col_prod + 2 * mm, y_row_bottom + 2 * mm, prod_text)
            c.drawRightString(col_qty + (margin_right - col_qty) - 2 * mm,
                              y_row_bottom + 2 * mm,
                              str(qty))
            y = y_row_bottom

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# ---------- PARSER MANIFIESTO PDF ----------
def parse_manifest_pdf(uploaded_file):
    if not HAS_PDF_LIB:
        raise RuntimeError(
            "La librería pdfplumber no está instalada. "
            "Agrega 'pdfplumber' a requirements.txt en Streamlit."
        )

    import pdfplumber as _pdfplumber
    records = []

    with _pdfplumber.open(uploaded_file) as pdf:
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
                try:
                    qty = int(m_qty.group(1))
                except Exception:
                    continue

                sku = None
                order = None
                buyer = ""
                start = max(0, i - 10)

                # Buscar hacia arriba SKU y Venta
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

                # Buyer entre Venta y Cantidad
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
                            "código carrier", "firma carrier",
                            "fecha y hora de retiro"
                        ]):
                            continue
                        if re.fullmatch(r"[0-9 .:/-]+", cand):
                            continue
                        buyer = cand
                        break

                records.append(
                    {
                        "ml_order_id": order,
                        "buyer": buyer,
                        "sku_ml": sku,
                        "mlc_id": None,
                        "title_ml": "",
                        "qty": qty,
                    }
                )

    if not records:
        return pd.DataFrame(
            columns=["ml_order_id", "buyer", "sku_ml", "mlc_id", "title_ml", "qty"]
        )

    return pd.DataFrame(records)


# ---------- PÁGINA ADMIN ----------
def page_admin():
    st.header("4) Panel administrador")

    pwd = st.text_input("Contraseña de administrador", type="password")
    if pwd != ADMIN_PASSWORD:
        st.info("Ingresa la contraseña para ver las opciones de administración.")
        return

    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT COUNT(*), COALESCE(SUM(qty),0) FROM order_items;")
    total_lineas, total_unidades = c.fetchone()

    c.execute("SELECT COUNT(DISTINCT sku_ml) FROM order_items;")
    total_skus = c.fetchone()[0] or 0

    c.execute("SELECT COALESCE(SUM(qty_total),0), COALESCE(SUM(qty_picked),0) FROM picking_global;")
    total_picking, total_picked = c.fetchone()

    col1, col2, col3 = st.columns(3)
    col1.metric("Órdenes / líneas", total_lineas)
    col2.metric("SKUs distintos", total_skus)
    col3.metric("Unidades picking vs pickeadas", f"{total_picked}/{total_picking}")

    st.subheader("OTs de picking y estado")
    c.execute("""
        SELECT po.id,
               po.ot_code,
               COALESCE(pk.name, '—') AS picker,
               po.created_at,
               COUNT(pg.id) AS skus_totales,
               SUM(CASE WHEN pg.qty_picked >= pg.qty_total THEN 1 ELSE 0 END) AS skus_completos
        FROM picking_ots po
        LEFT JOIN pickers pk ON pk.id = po.picker_id
        LEFT JOIN picking_global pg ON pg.ot_id = po.id
        GROUP BY po.id, po.ot_code, pk.name, po.created_at
        ORDER BY po.ot_code;
    """)
    rows = c.fetchall()
    if rows:
        df_ots = pd.DataFrame(
            rows,
            columns=[
                "ID OT", "Código OT", "Piqueador",
                "Creada", "SKUs totales", "SKUs completos"
            ],
        )
        st.dataframe(df_ots)
    else:
        st.info("No hay OTs generadas en esta base.")

    st.subheader("Acciones sobre picking actual")
    if st.button("Reiniciar picking actual (mantiene ventas históricas)"):
        c.execute("DELETE FROM picking_global;")
        c.execute("DELETE FROM packages_scan;")
        c.execute("DELETE FROM pickers;")
        c.execute("DELETE FROM sku_images;")
        c.execute("DELETE FROM picking_ots;")
        conn.commit()
        st.success("Picking actual reiniciado. Las ventas (orders y order_items) se mantienen.")
    conn.close()


# ---------- PÁGINA IMPORTAR VENTAS ----------
def page_import_ml():
    st.header("1) Importar ventas")

    origen = st.radio(
        "Origen de datos de ventas",
        ["Excel Mercado Libre", "Manifiesto PDF (etiquetas)"],
        horizontal=True,
    )

    num_pickers = st.number_input(
        "Cantidad de piqueadores para esta corrida",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
    )

    st.markdown("### Maestro de SKUs y nombres técnicos (opcional, recomendado)")
    inv_file = st.file_uploader(
        "Archivo maestro de inventario (.xlsx) (LibroInventario o maestro sku)",
        type=["xlsx"],
        key="inv_uploader",
    )

    st.markdown("### Archivo de imágenes por MLC (opcional)")
    st.caption("Archivo de publicaciones de MELI con columna MLC (ID de publicación) y URL de imagen.")
    img_file = st.file_uploader(
        "Archivo de imágenes (xlsx o csv)",
        type=["xlsx", "csv"],
        key="img_uploader",
    )

    img_df = None
    img_mlc_col = None
    img_url_col = None

    if img_file is not None:
        try:
            if img_file.name.lower().endswith(".csv"):
                img_df = pd.read_csv(img_file)
            else:
                img_df = pd.read_excel(img_file)
            st.success(f"Archivo de imágenes cargado con {len(img_df)} filas.")
            st.dataframe(img_df.head())
            if len(img_df.columns) >= 2:
                img_mlc_col = st.selectbox(
                    "Columna con MLC (ID de publicación)",
                    img_df.columns,
                    key="img_mlc_col",
                )
                img_url_col = st.selectbox(
                    "Columna con URL de la imagen",
                    img_df.columns,
                    key="img_url_col",
                )
            else:
                st.warning("El archivo de imágenes debe tener al menos 2 columnas (MLC y URL).")
        except Exception as e:
            st.error(f"No se pudo leer el archivo de imágenes: {e}")
            img_df = None
            img_mlc_col = None
            img_url_col = None

    sales_df = None

    # ------- EXCEL ML -------
    if origen == "Excel Mercado Libre":
        st.write("Sube el archivo de ventas del día exportado desde Mercado Libre (XLSX).")
        file = st.file_uploader("Archivo de ventas ML (.xlsx)", type=["xlsx"], key="ventas_xlsx")

        if file is None:
            st.info("Esperando archivo de ventas de Mercado Libre...")
            return

        try:
            df = pd.read_excel(file, header=[4, 5])
        except Exception as e:
            st.error(f"Error leyendo el Excel de ML: {e}")
            st.stop()

        df.columns = [
            " | ".join([str(x) for x in col if str(x) != "nan"])
            for col in df.columns
        ]

        COLUMN_ORDER_ID = "Ventas | # de venta"
        COLUMN_QTY = "Ventas | Unidades"
        COLUMN_SKU = "Publicaciones | SKU"
        COLUMN_TITLE = "Publicaciones | Título de la publicación"
        COLUMN_BUYER = "Compradores | Comprador"

        required = [COLUMN_ORDER_ID, COLUMN_QTY, COLUMN_SKU, COLUMN_TITLE, COLUMN_BUYER]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Faltan columnas en el archivo de Mercado Libre: {missing}")
            st.stop()

        mlc_candidates = [
            "Publicaciones | # de publicación",
            "Publicaciones | ID de publicación",
            "Publicaciones | # publicación",
            "# de publicación",
            "# publicación",
            "ID de publicación",
        ]
        mlc_col_found = None
        for cand in mlc_candidates:
            if cand in df.columns:
                mlc_col_found = cand
                break

        cols_to_copy = [COLUMN_ORDER_ID, COLUMN_QTY, COLUMN_SKU, COLUMN_TITLE, COLUMN_BUYER]
        col_names = ["ml_order_id", "qty", "sku_ml", "title_ml", "buyer"]

        if mlc_col_found:
            cols_to_copy.append(mlc_col_found)
            col_names.append("mlc_id")

        work_df = df[cols_to_copy].copy()
        work_df.columns = col_names

        if "mlc_id" not in work_df.columns:
            st.warning(
                "No se encontró una columna MLC (# publicación / ID de publicación) en el Excel de ventas. "
                "Las imágenes por MLC no se podrán enlazar para estas ventas."
            )
            work_df["mlc_id"] = None

        work_df["qty"] = pd.to_numeric(work_df["qty"], errors="coerce").fillna(0).astype(int)
        work_df = work_df[work_df["qty"] > 0]

        if work_df.empty:
            st.error("Después de limpiar las cantidades, no quedó ninguna línea con qty > 0.")
       

Let's continue the code (I got cut).
::contentReference[oaicite:0]{index=0}
