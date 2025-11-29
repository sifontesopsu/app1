import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from io import BytesIO
import re  # para parsing del PDF

# ========= CÓDIGOS DE BARRAS (SIN INTEGRACIONES EXTERNAS) =========
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
ADMIN_PASSWORD = "aurora123"  # 🔐 CAMBIA ESTA CLAVE A LA QUE QUIERAS


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

    # Picking global por SKU / MLC (incluye picker_id y ot_id)
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

    # Paquetes escaneados en conteo final (no usados ahora, pero dejamos la tabla)
    c.execute("""
    CREATE TABLE IF NOT EXISTS packages_scan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracking_code TEXT,
        scanned_at TEXT
    );
    """)

    # Tabla de piqueadores
    c.execute("""
    CREATE TABLE IF NOT EXISTS pickers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    );
    """)

    # Tabla de OTs de picking (una por piqueador por corrida)
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


# ---------- PDF HOJAS DE PICKING (TABLA CON CELDAS) ----------
def build_picklist_pdf(ot_data_list):
    """
    ot_data_list: lista de dicts:
      {
        "ot_code": str,
        "picker_name": str,
        "created_at": str,
        "items": [(sku, producto, qty), ...]
      }
    """
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

        # Posición inicial para contenido
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

            c.drawString(col_sku + 2 * mm, y_row_bottom + 2 * mm, str(sku)[:20])
            c.drawString(col_prod + 2 * mm, y_row_bottom + 2 * mm, str(producto)[:80])
            c.drawRightString(col_qty + (margin_right - col_qty) - 2 * mm,
                              y_row_bottom + 2 * mm,
                              str(qty))
            y = y_row_bottom

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes


# ---------- PARSER MANIFIESTO PDF (robusto, multi-página) ----------
def parse_manifest_pdf(uploaded_file):
    """
    Devuelve un DataFrame con:
      ml_order_id, buyer, sku_ml, mlc_id(None), title_ml(''), qty

    Lógica:
      - Recorre todas las páginas.
      - Cada vez que ve "Cantidad: X", mira hacia arriba (hasta 10 líneas)
        y busca el SKU y la Venta más cercanos.
      - El buyer se toma como la primera línea "de texto normal" entre Venta y Cantidad.
    """
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
                            sku = m_sku.group(1).strip()

                    if order is None and "Venta" in l:
                        m_ord = re.search(r"Venta\s*[:#]?\s*([0-9]+)", l)
                        if m_ord:
                            order = m_ord.group(1).strip()

                if not (order and sku):
                    continue

                # Buscar buyer entre la línea de Venta y la de Cantidad
                venta_idx = None
                for k in range(start, i):
                    if "Venta" in lines[k]:
                        venta_idx = k
                        break

                if venta_idx is not None:
                    for k in range(venta_idx + 1, i):
                        cand = lines[k].strip()
                        low = cand.lower()
                        if any(tok in low for tok in ["venta", "sku", "pack id", "cantidad",
                                                      "código carrier", "firma carrier",
                                                      "fecha y hora de retiro"]):
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


# ---------- PÁGINA: ADMIN (pausada) ----------
def page_admin():
    st.header("Panel de administrador (PAUSADO)")
    st.info("Este módulo está oculto del menú por ahora.")


# ---------- PÁGINA: IMPORTAR VENTAS ----------
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
        "Archivo maestro de inventario (.xlsx) con columnas 'SKU' y 'Artículo'",
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
            st.stop()

        sales_df = work_df[["ml_order_id", "buyer", "sku_ml", "mlc_id", "title_ml", "qty"]].copy()
        st.subheader("Vista previa (ventas procesadas)")
        st.dataframe(sales_df.head())

    # ------- MANIFIESTO PDF -------
    else:
        st.write("Sube el manifiesto PDF con las etiquetas.")
        pdf_file = st.file_uploader("Manifiesto PDF", type=["pdf"], key="ventas_pdf")

        if pdf_file is None:
            st.info("Esperando archivo PDF de manifiesto...")
            return

        try:
            sales_df = parse_manifest_pdf(pdf_file)
        except Exception as e:
            st.error(f"No se pudo procesar el PDF: {e}")
            st.stop()

        if sales_df.empty:
            st.error("No se encontraron ventas válidas en el PDF.")
            st.stop()

        st.subheader("Vista previa de ventas detectadas en PDF")
        st.dataframe(sales_df.head())

    # ---- Maestro de inventario: mapa SKU -> nombre técnico ----
    inv_map = {}
    if inv_file is not None:
        try:
            inv_df = pd.read_excel(inv_file)
            cols_inv = inv_df.columns.tolist()
            if "SKU" in cols_inv and "Artículo" in cols_inv:
                inv_df = inv_df.dropna(subset=["SKU", "Artículo"])
                for _, row in inv_df.iterrows():
                    sku_key = str(row["SKU"]).strip()
                    art = str(row["Artículo"]).strip()
                    if sku_key:
                        inv_map[sku_key] = art
                st.success(f"Maestro de inventario cargado ({len(inv_map)} SKUs con nombre técnico).")
            else:
                st.warning("El maestro no tiene columnas 'SKU' y 'Artículo'. Se continuará sin nombres técnicos.")
        except Exception as e:
            st.warning(f"No se pudo leer el maestro de inventario: {e}")

    # ---- Cargar en DB ----
    if st.button("Cargar ventas en el sistema"):
        conn = get_conn()
        c = conn.cursor()

        # Limpiar solo parte operativa de picking
        c.execute("DELETE FROM picking_global;")
        c.execute("DELETE FROM packages_scan;")
        c.execute("DELETE FROM pickers;")
        c.execute("DELETE FROM sku_images;")
        c.execute("DELETE FROM picking_ots;")

        # Insertar pedidos y líneas (histórico se mantiene)
        for ml_order_id, grupo in sales_df.groupby("ml_order_id"):
            c.execute("SELECT id FROM orders WHERE ml_order_id = ?;", (str(ml_order_id),))
            row_exist = c.fetchone()
            if row_exist:
                continue

            buyer = str(grupo["buyer"].iloc[0]) if "buyer" in grupo.columns else ""
            created_at = datetime.now().isoformat()

            c.execute("""
                INSERT INTO orders (ml_order_id, buyer, created_at)
                VALUES (?, ?, ?)
            """, (str(ml_order_id), buyer, created_at))
            order_id = c.lastrowid

            for _, row in grupo.iterrows():
                sku = str(row["sku_ml"]).strip()
                title_ml_raw = str(row["title_ml"]) if "title_ml" in row and str(row["title_ml"]) not in ["nan"] else ""
                mlc_id_raw = row.get("mlc_id", None)

                mlc_id = None
                if mlc_id_raw is not None and str(mlc_id_raw).lower() != "nan":
                    mlc_id = str(mlc_id_raw).strip()

                title_tec = inv_map.get(sku)
                title_ml = title_tec if (not title_ml_raw and title_tec) else title_ml_raw

                qty = int(row["qty"])
                if qty <= 0:
                    continue

                c.execute("""
                    INSERT INTO order_items (order_id, sku_ml, mlc_id, title_ml, title_tec, qty)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (order_id, sku, mlc_id, title_ml, title_tec, qty))

        # Generar picking_global agrupado por SKU
        c.execute("""
            SELECT sku_ml, mlc_id, title_ml, title_tec, SUM(qty) as total
            FROM order_items
            GROUP BY sku_ml, mlc_id, title_ml, title_tec
        """)
        rows = c.fetchall()
        for sku, mlc_id, title_ml, title_tec, total in rows:
            c.execute("""
                INSERT INTO picking_global (sku_ml, mlc_id, title_ml, title_tec, qty_total, qty_picked, picker_id, ot_id)
                VALUES (?, ?, ?, ?, ?, 0, NULL, NULL)
            """, (sku, mlc_id, title_ml, title_tec, total))

        # Crear piqueadores y repartir SKUs
        num_pickers_int = int(num_pickers)
        for i in range(num_pickers_int):
            name = f"P{i+1}"
            c.execute("INSERT INTO pickers (name) VALUES (?);", (name,))

        c.execute("SELECT id, name FROM pickers ORDER BY id;")
        pickers = c.fetchall()
        total_pickers = len(pickers)

        if total_pickers > 0:
            c.execute("""
                SELECT id FROM picking_global
                ORDER BY 
                    CASE WHEN title_tec IS NULL OR title_tec = '' THEN 1 ELSE 0 END,
                    title_tec,
                    title_ml
            """)
            skus_pg = c.fetchall()
            for idx, (pg_id,) in enumerate(skus_pg):
                picker_id = pickers[idx % total_pickers][0]
                c.execute("UPDATE picking_global SET picker_id = ? WHERE id = ?;", (picker_id, pg_id))

            for pid, pname in pickers:
                created_at = datetime.now().isoformat()
                c.execute("""
                    INSERT INTO picking_ots (picker_id, ot_code, created_at)
                    VALUES (?, ?, ?)
                """, (pid, "", created_at))
                ot_id = c.lastrowid
                ot_code = f"OT{ot_id:06d}"
                c.execute("UPDATE picking_ots SET ot_code = ? WHERE id = ?;", (ot_code, ot_id))

                c.execute("""
                    UPDATE picking_global
                    SET ot_id = ?
                    WHERE picker_id = ?
                """, (ot_id, pid))

        # Imágenes por MLC
        if img_df is not None and img_mlc_col and img_url_col:
            inserted = 0
            for _, row in img_df.iterrows():
                mlc_val = str(row[img_mlc_col]).strip()
                url_val = str(row[img_url_col]).strip()
                if mlc_val and url_val:
                    c.execute("""
                        INSERT OR REPLACE INTO sku_images (mlc_id, image_url)
                        VALUES (?, ?)
                    """, (mlc_val, url_val))
                    inserted += 1
            st.success(f"Se cargaron {inserted} imágenes en la tabla sku_images (por MLC).")

        conn.commit()
        conn.close()

        st.success("Ventas cargadas, picking generado, OTs creadas y distribuidas entre piqueadores correctamente.")
        st.info("Las ventas anteriores se mantienen en la base de datos para trazabilidad. Solo se reinició el picking de esta corrida.")


# ---------- PÁGINA: HOJAS DE PICKING ----------
def page_hojas_picking():
    st.header("2) Hojas de picking (papel por OT)")

    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT po.id, po.ot_code, pk.name, po.created_at
        FROM picking_ots po
        JOIN pickers pk ON pk.id = po.picker_id
        ORDER BY po.id
    """)
    ot_rows = c.fetchall()

    if not ot_rows:
        st.info("No hay OTs generadas. Primero importa ventas en '1) Importar ventas'.")
        conn.close()
        return

    opciones = ["Todas las OTs"]
    ot_map = {}
    for ot_id, ot_code, picker_name, created_at in ot_rows:
        label = f"{ot_code} – {picker_name}"
        opciones.append(label)
        ot_map[label] = (ot_id, ot_code, picker_name, created_at)

    seleccion = st.selectbox("Selecciona OT para ver/imprimir", opciones, index=0)

    if seleccion == "Todas las OTs":
        ots_a_mostrar = ot_rows
    else:
        ot_id, ot_code, picker_name, created_at = ot_map[seleccion]
        ots_a_mostrar = [(ot_id, ot_code, picker_name, created_at)]

    st.write("Estas son las hojas de picking que puedes imprimir o descargar en PDF.")

    ot_data_list = []

    for ot_id, ot_code, picker_name, created_at in ots_a_mostrar:
        st.markdown("---")
        st.subheader(f"OT: {ot_code} – Piqueador: {picker_name}")
        st.caption(f"Creada: {created_at}")

        img_bytes = generate_barcode_bytes(ot_code)
        if img_bytes is not None:
            try:
                st.image(img_bytes, caption=f"Código de barras OT {ot_code}", use_container_width=False)
            except Exception as e:
                st.write(f"Error generando código de barras para {ot_code}: {e}")
        else:
            st.markdown(f"**Código OT:** `{ot_code}`")

        c.execute("""
            SELECT sku_ml,
                   COALESCE(title_tec, title_ml) AS producto,
                   qty_total
            FROM picking_global
            WHERE ot_id = ?
            ORDER BY producto
        """, (ot_id,))
        rows = c.fetchall()

        if not rows:
            st.write("No hay SKUs asignados a esta OT.")
            continue

        df = pd.DataFrame(rows, columns=["SKU", "Producto", "Cantidad a pickear"])
        st.table(df)

        items = [(r[0], r[1], r[2]) for r in rows]
        ot_data_list.append(
            {
                "ot_code": ot_code,
                "picker_name": picker_name,
                "created_at": created_at,
                "items": items,
            }
        )

    conn.close()

    if ot_data_list:
        pdf_bytes = build_picklist_pdf(ot_data_list)
        st.download_button(
            "📄 Descargar PDF de hojas de picking",
            data=pdf_bytes,
            file_name="hojas_picking_aurora.pdf",
            mime="application/pdf",
        )


# ---------- PÁGINA: CERRAR OT ----------
def page_cerrar_ot():
    st.header("3) Cerrar OT (escaneo único por piqueador)")

    conn = get_conn()
    c = conn.cursor()

    st.write("""
    El piqueador termina su recorrido con la hoja en papel.
    Aquí escanea el **código de barras de la OT** (o escribe el código OT)
    para marcar como pickeados todos los productos asignados a esa OT.
    """)

    code = st.text_input("Escanee o escriba el código de la OT (ej: OT000001)", key="scan_ot")

    if st.button("Cerrar OT"):
        if not code:
            st.warning("Escanee o escriba un código de OT primero.")
        else:
            code = code.strip()

            c.execute("""
                SELECT po.id, po.picker_id, pk.name
                FROM picking_ots po
                JOIN pickers pk ON pk.id = po.picker_id
                WHERE po.ot_code = ?
            """, (code,))
            ot_row = c.fetchone()

            if not ot_row:
                st.error("OT no encontrada.")
            else:
                ot_id, picker_id, picker_name = ot_row

                c.execute("""
                    SELECT COUNT(*),
                           SUM(CASE WHEN qty_picked >= qty_total THEN 1 ELSE 0 END)
                FROM picking_global
                WHERE ot_id = ?
                """, (ot_id,))
                total_skus, skus_completos = c.fetchone()
                total_skus = total_skus or 0
                skus_completos = skus_completos or 0

                if total_skus == 0:
                    st.warning("Esta OT no tiene SKUs asignados.")
                elif skus_completos == total_skus:
                    st.info(f"La OT {code} ya estaba completamente cerrada (todos los SKUs pickeados).")
                else:
                    c.execute("""
                        UPDATE picking_global
                        SET qty_picked = qty_total
                        WHERE ot_id = ?
                    """, (ot_id,))
                    conn.commit()
                    st.success(
                        f"OT {code} cerrada para {picker_name}: "
                        f"{total_skus} SKUs marcados como pickeados."
                    )

    conn.close()


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="Aurora ML – Picking por OT", layout="wide")
    init_db()

    st.sidebar.title("Aurora ML – Picking OT")
    page = st.sidebar.radio(
        "Menú",
        [
            "1) Importar ventas",
            "2) Hojas de picking (papel por OT)",
            "3) Cerrar OT (escaneo)",
        ],
    )

    if page.startswith("1"):
        page_import_ml()
    elif page.startswith("2"):
        page_hojas_picking()
    elif page.startswith("3"):
        page_cerrar_ot()


if __name__ == "__main__":
    main()
