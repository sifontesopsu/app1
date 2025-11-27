import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

# Intentar importar pdfplumber para leer manifiestos PDF
try:
    import pdfplumber
    HAS_PDF_LIB = True
except ImportError:
    HAS_PDF_LIB = False

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

    # Picking global por SKU / MLC (incluye picker_id para distribución)
    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_global (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku_ml TEXT,
        mlc_id TEXT,
        title_ml TEXT,
        title_tec TEXT,
        qty_total INTEGER,
        qty_picked INTEGER DEFAULT 0,
        picker_id INTEGER
    );
    """)

    # Paquetes escaneados en conteo final
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

    # 🔄 Asegurar columnas nuevas en tablas existentes
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

    # 🔄 sku_images: siempre basada en MLC
    c.execute("DROP TABLE IF EXISTS sku_images;")
    c.execute("""
    CREATE TABLE IF NOT EXISTS sku_images (
        mlc_id TEXT PRIMARY KEY,
        image_url TEXT
    );
    """)

    conn.commit()
    conn.close()


# ---------- PARSER MANIFIESTO PDF ----------
def parse_manifest_pdf(uploaded_file):
    """
    Lee un PDF de manifiesto con etiquetas y devuelve un DataFrame con columnas:
    ml_order_id, buyer, sku_ml, mlc_id (None), title_ml (''), qty.
    """
    if not HAS_PDF_LIB:
        raise RuntimeError(
            "La librería pdfplumber no está instalada. "
            "Agrega 'pdfplumber' a requirements.txt en Streamlit."
        )

    import pdfplumber as _pdfplumber  # asegurar

    records = []
    with _pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            lines = text.splitlines()

            current_order_id = None
            current_buyer = None
            next_is_buyer = False
            current_sku = None

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Ej: "Venta: 2000014009097738"
                if line.startswith("Venta:"):
                    current_order_id = line.split("Venta:")[1].strip()
                    current_buyer = None
                    next_is_buyer = True
                    current_sku = None
                    continue

                # La línea siguiente después de "Venta:" es el comprador
                if next_is_buyer and current_order_id:
                    current_buyer = line.strip()
                    next_is_buyer = False
                    continue

                # Ej: "SKU: 2444103"
                if line.startswith("SKU:") and current_order_id:
                    current_sku = line.split("SKU:")[1].strip()
                    continue

                # Ej: "Cantidad: 1"
                if line.startswith("Cantidad:") and current_order_id and current_sku:
                    qty_part = line.split("Cantidad:")[1].strip()
                    try:
                        qty = int(qty_part.split()[0])
                    except ValueError:
                        continue

                    records.append(
                        {
                            "ml_order_id": current_order_id,
                            "buyer": current_buyer or "",
                            "sku_ml": current_sku,
                            "mlc_id": None,
                            "title_ml": "",
                            "qty": qty,
                        }
                    )
                    current_sku = None
                    continue

    if not records:
        return pd.DataFrame(
            columns=["ml_order_id", "buyer", "sku_ml", "mlc_id", "title_ml", "qty"]
        )
    return pd.DataFrame(records)


# ---------- PÁGINA: ADMIN ----------
def page_admin():
    st.header("Panel de administrador")

    # Login simple de admin
    if not st.session_state.get("admin_authenticated", False):
        st.info("Ingresa la clave de administrador para acceder a las opciones avanzadas.")
        pwd = st.text_input("Clave de administrador", type="password")
        if st.button("Entrar como administrador"):
            if pwd == ADMIN_PASSWORD:
                st.session_state["admin_authenticated"] = True
                st.success("Sesión de administrador iniciada.")
                st.rerun()
            else:
                st.error("Clave incorrecta.")
        return

    st.success("Estás en modo administrador.")

    conn = get_conn()
    c = conn.cursor()

    # Resumen rápido del sistema
    st.subheader("Resumen del sistema")

    c.execute("SELECT COUNT(*) FROM orders;")
    total_orders = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM order_items;")
    total_items = c.fetchone()[0] or 0

    c.execute("SELECT COUNT(*) FROM picking_global;")
    total_skus = c.fetchone()[0] or 0

    c.execute("SELECT COALESCE(SUM(qty_total), 0), COALESCE(SUM(qty_picked), 0) FROM picking_global;")
    total_qty, total_picked = c.fetchone()
    total_qty = total_qty or 0
    total_picked = total_picked or 0
    avance = 0
    if total_qty > 0:
        avance = round((total_picked / total_qty) * 100, 1)

    c.execute("SELECT COUNT(*) FROM packages_scan;")
    total_packages_scanned = c.fetchone()[0] or 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pedidos cargados", total_orders)
    col2.metric("Líneas de ítems", total_items)
    col3.metric("SKUs en picking", total_skus)
    col4.metric("Paquetes escaneados", total_packages_scanned)

    st.metric("Avance global de picking", f"{avance} %", help="Unidades pickeadas / unidades totales del día")

    st.markdown("---")
    st.subheader("Acciones de administración")

    col_a, col_b, col_c = st.columns(3)

    # Resetear solo cantidades pickeadas
    with col_a:
        st.write("**Resetear SOLO picking (qty_picked = 0)**")
        st.caption("No borra pedidos ni ítems, solo deja todas las cantidades pickeadas en 0.")
        if st.button("Resetear cantidades pickeadas"):
            c.execute("UPDATE picking_global SET qty_picked = 0;")
            conn.commit()
            st.success("Se reiniciaron todas las cantidades pickeadas.")

    # Resetear solo paquetes
    with col_b:
        st.write("**Resetear SOLO paquetes (tracking)**")
        st.caption("Borra todos los códigos escaneados, pero mantiene pedidos e ítems.")
        if st.button("Resetear paquetes escaneados"):
            c.execute("DELETE FROM packages_scan;")
            conn.commit()
            st.warning("Se borraron todos los paquetes escaneados.")

    # Resetear TODO
    with col_c:
        st.write("**Resetear TODO el sistema**")
        st.caption("Borra pedidos, ítems, picking, imágenes y paquetes. Úsalo solo al cambiar de día o en pruebas.")
        confirm = st.checkbox("Confirmo que quiero borrar TODOS los datos", key="confirm_reset_all")
        if st.button("BORRAR TODO (sistema completo)"):
            if confirm:
                c.execute("DELETE FROM order_items;")
                c.execute("DELETE FROM orders;")
                c.execute("DELETE FROM picking_global;")
                c.execute("DELETE FROM packages_scan;")
                c.execute("DELETE FROM sku_images;")
                c.execute("DELETE FROM pickers;")
                conn.commit()
                st.warning("Se borraron TODOS los datos del sistema.")
            else:
                st.error("Marca la casilla de confirmación antes de borrar todo.")

    st.markdown("---")
    if st.button("Cerrar sesión de administrador"):
        st.session_state["admin_authenticated"] = False
        st.success("Sesión de administrador cerrada.")
        st.rerun()

    conn.close()


# ---------- PÁGINA: IMPORTAR VENTAS (EXCEL / PDF) ----------
def page_import_ml():
    st.header("1) Importar ventas")

    origen = st.radio(
        "Origen de datos de ventas",
        ["Excel Mercado Libre", "Manifiesto PDF (etiquetas)"],
        horizontal=True,
    )

    # Cantidad de piqueadores
    num_pickers = st.number_input(
        "Cantidad de piqueadores para hoy",
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
            st.markdown("Vista previa archivo imágenes:")
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

    # ------- ORIGEN EXCEL ML -------
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

        # Aplanar MultiIndex
        df.columns = [
            " | ".join([str(x) for x in col if str(x) != "nan"])
            for col in df.columns
        ]

        # Columnas mínimas necesarias
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

     # Buscar la columna de MLC (# publicación) entre varios candidatos
mlc_candidates = [
    "Publicaciones | # de publicación",   # 👈 ESTE ES EL QUE REALMENTE TIENES
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

        # Normalizar cantidades
        work_df["qty"] = pd.to_numeric(work_df["qty"], errors="coerce").fillna(0).astype(int)
        work_df = work_df[work_df["qty"] > 0]

        if work_df.empty:
            st.error("Después de limpiar las cantidades, no quedó ninguna línea con qty > 0.")
            st.stop()

        sales_df = work_df[["ml_order_id", "buyer", "sku_ml", "mlc_id", "title_ml", "qty"]].copy()

        st.subheader("Vista previa (ventas procesadas)")
        st.dataframe(sales_df.head())

    # ------- ORIGEN MANIFIESTO PDF -------
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

    # ---- Botón para cargar todo en DB ----
    if st.button("Cargar ventas en el sistema"):
        conn = get_conn()
        c = conn.cursor()

        # Limpiar datos anteriores
        c.execute("DELETE FROM order_items;")
        c.execute("DELETE FROM orders;")
        c.execute("DELETE FROM picking_global;")
        c.execute("DELETE FROM packages_scan;")
        c.execute("DELETE FROM pickers;")
        c.execute("DELETE FROM sku_images;")

        # Insertar pedidos y sus líneas
        for ml_order_id, grupo in sales_df.groupby("ml_order_id"):
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

                # Normalizar mlc_id
                mlc_id = None
                if mlc_id_raw is not None and str(mlc_id_raw).lower() != "nan":
                    mlc_id = str(mlc_id_raw).strip()

                title_tec = inv_map.get(sku)

                if not title_ml_raw and title_tec:
                    title_ml = title_tec
                else:
                    title_ml = title_ml_raw

                qty = int(row["qty"])
                if qty <= 0:
                    continue

                c.execute("""
                    INSERT INTO order_items (order_id, sku_ml, mlc_id, title_ml, title_tec, qty)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (order_id, sku, mlc_id, title_ml, title_tec, qty))

        # Generar picking_global agrupando todos los SKUs/MLC del día
        c.execute("""
            SELECT sku_ml, mlc_id, title_ml, title_tec, SUM(qty) as total
            FROM order_items
            GROUP BY sku_ml, mlc_id, title_ml, title_tec
        """)
        rows = c.fetchall()
        for sku, mlc_id, title_ml, title_tec, total in rows:
            c.execute("""
                INSERT INTO picking_global (sku_ml, mlc_id, title_ml, title_tec, qty_total, qty_picked, picker_id)
                VALUES (?, ?, ?, ?, ?, 0, NULL)
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

        # Cargar imágenes por MLC (si tenemos archivo)
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

        st.session_state["pick_index"] = 0
        st.success("Ventas cargadas, picking generado y distribuido entre piqueadores correctamente.")


# ---------- PÁGINA: PICKING GLOBAL ----------
def page_picking():
    st.header("2) Picking por producto")

    conn = get_conn()
    c = conn.cursor()

    # Selector de piqueador
    c.execute("SELECT id, name FROM pickers ORDER BY id;")
    picker_rows = c.fetchall()
    picker_options = ["Todos"]
    picker_id_map = {}
    for pid, pname in picker_rows:
        picker_options.append(pname)
        picker_id_map[pname] = pid

    selected_picker = st.selectbox("Lista de trabajo:", picker_options, index=0)

    # Traer productos (incluyendo imagen por MLC)
    c.execute("""
        SELECT pg.id,
               pg.sku_ml,
               pg.mlc_id,
               pg.title_ml,
               pg.title_tec,
               pg.qty_total,
               pg.qty_picked,
               si.image_url,
               pg.picker_id
        FROM picking_global pg
        LEFT JOIN sku_images si
               ON si.mlc_id = pg.mlc_id
        ORDER BY 
            CASE WHEN pg.title_tec IS NULL OR pg.title_tec = '' THEN 1 ELSE 0 END,
            pg.title_tec,
            pg.title_ml
    """)
    rows = c.fetchall()

    # Filtrar por piqueador
    if selected_picker != "Todos":
        target_id = picker_id_map.get(selected_picker)
        rows = [r for r in rows if r[8] == target_id]

    if not rows:
        st.info("No hay productos asignados con el filtro actual.")
        conn.close()
        return

    if "pick_index" not in st.session_state:
        st.session_state["pick_index"] = 0

    total_productos = len(rows)
    idx = st.session_state["pick_index"]

    if idx < 0:
        idx = 0
    if idx >= total_productos:
        idx = total_productos - 1

    st.session_state["pick_index"] = idx

    pg_id, sku_ml, mlc_id, title_ml, title_tec, qty_total, qty_picked, image_url, picker_id = rows[idx]
    restantes = qty_total - qty_picked

    # Título a mostrar
    display_title = title_tec if title_tec and str(title_tec).strip().lower() not in ["", "nan"] else title_ml

    # ======== ESTILO VISUAL ==========
    st.markdown("""
        <style>
        .title-big {
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 6px;
        }
        .sku-big {
            font-size: 18px;
            color: #555;
            margin-bottom: 4px;
        }
        .ml-name {
            font-size: 14px;
            color: #777;
            margin-bottom: 15px;
            font-style: italic;
        }

        .kpi-box {
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
        }
        .kpi-label {
            font-size: 14px;
        }
        .kpi-total { background-color: #e6f0ff; color: #0047ab; }
        .kpi-picked { background-color: #e6ffe6; color: #0a7a0a; }
        .kpi-rest { background-color: #fff1e6; color: #b34700; }

        .btn-big button {
            height: 80px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            width: 100% !important;
        }

        @media (max-width: 768px) {
            .title-big {
                font-size: 26px;
            }
            .sku-big {
                font-size: 16px;
            }
            .kpi-box {
                font-size: 24px;
                padding: 16px;
            }
            .btn-big button {
                height: 90px !important;
                font-size: 22px !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # ======== HEADER DEL PRODUCTO =========
    st.markdown(f"<div class='title-big'>{display_title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sku-big'>SKU: {sku_ml}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='ml-name'>Nombre ML: {title_ml}</div>", unsafe_allow_html=True)

    info_sub = f"Producto {idx+1} de {total_productos}"
    if selected_picker != "Todos":
        info_sub += f" · Lista de {selected_picker}"
    st.markdown(info_sub)

    col_info, col_img = st.columns([2, 1])

    with col_img:
        if image_url and isinstance(image_url, str) and image_url.strip():
            st.image(image_url, use_container_width=True)
        else:
            st.write("Sin imagen")

    # ======== KPIs =========
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"<div class='kpi-box kpi-total'><div class='kpi-label'>TOTAL</div>{qty_total}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<div class='kpi-box kpi-picked'><div class='kpi-label'>PICKEADO</div>{qty_picked}</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"<div class='kpi-box kpi-rest'><div class='kpi-label'>RESTANTE</div>{restantes}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ======== BOTONES GRANDES =========
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("<div class='btn-big'>", unsafe_allow_html=True)
        if st.button("➕ SUMAR"):
            if qty_picked < qty_total:
                c.execute("""
                    UPDATE picking_global
                    SET qty_picked = qty_picked + 1
                    WHERE id = ?
                """, (pg_id,))
                conn.commit()
                st.rerun()
            else:
                st.warning("Cantidad completa.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='btn-big'>", unsafe_allow_html=True)
        if st.button("➖ RESTAR"):
            if qty_picked > 0:
                c.execute("""
                    UPDATE picking_global
                    SET qty_picked = qty_picked - 1
                    WHERE id = ?
                """, (pg_id,))
                conn.commit()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown("<div class='btn-big'>", unsafe_allow_html=True)
        if st.button("✅ COMPLETAR"):
            c.execute("""
                UPDATE picking_global
                SET qty_picked = ?
                WHERE id = ?
            """, (qty_total, pg_id))
            conn.commit()
            st.session_state["pick_index"] = min(idx + 1, total_productos - 1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        st.markdown("<div class='btn-big'>", unsafe_allow_html=True)
        if st.button("🔄 REINICIAR"):
            c.execute("""
                UPDATE picking_global
                SET qty_picked = 0
                WHERE id = ?
            """, (pg_id,))
            conn.commit()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ======== NAVEGACIÓN =========
    col_prev, col_next = st.columns(2)

    with col_prev:
        st.markdown("<div class='btn-big'>", unsafe_allow_html=True)
        if st.button("⬅️ ANTERIOR"):
            st.session_state["pick_index"] = max(idx - 1, 0)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_next:
        st.markdown("<div class='btn-big'>", unsafe_allow_html=True)
        if st.button("➡️ SIGUIENTE"):
            st.session_state["pick_index"] = min(idx + 1, total_productos - 1)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    conn.close()


# ---------- PÁGINA: CONTEO FINAL ----------
def page_conteo_final():
    st.header("3) Conteo final de paquetes (tracking / número de venta)")

    st.write("""
    Después de embalar y etiquetar, en la zona de despacho
    escanea una sola vez el **número de venta de Mercado Libre** (o el código que uses
    en la etiqueta) para marcar ese pedido como contado.
    """)

    conn = get_conn()
    c = conn.cursor()

    # Pedidos esperados
    c.execute("SELECT DISTINCT ml_order_id FROM orders;")
    orders_all = [row[0] for row in c.fetchall()]
    expected_orders = len(orders_all)

    # Paquetes escaneados
    c.execute("SELECT DISTINCT tracking_code FROM packages_scan;")
    scanned_codes = [row[0] for row in c.fetchall()]
    scanned_packages = len(scanned_codes)

    st.subheader("Resumen")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pedidos (ventas ML)", expected_orders)
    col2.metric("Pedidos escaneados", scanned_packages)
    diff = scanned_packages - expected_orders
    col3.metric("Diferencia (escaneados - pedidos)", diff)

    st.markdown("---")
    st.subheader("Escanear número de venta / tracking")

    tracking = st.text_input(
        "Escanee o escriba el **número de venta de ML** (o código que uses para identificar el pedido)",
        key="scan_tracking"
    )

    col4, col5 = st.columns(2)
    with col4:
        if st.button("Registrar paquete"):
            if not tracking:
                st.warning("Escanee o escriba un número de venta primero.")
            else:
                c.execute("SELECT COUNT(*) FROM packages_scan WHERE tracking_code = ?;", (tracking,))
                exists = c.fetchone()[0]
                if exists:
                    st.error("Este código ya fue escaneado antes (pedido duplicado).")
                else:
                    c.execute("""
                        INSERT INTO packages_scan (tracking_code, scanned_at)
                        VALUES (?, ?)
                    """, (tracking, datetime.now().isoformat()))
                    conn.commit()
                    st.success(f"Código {tracking} registrado como escaneado.")

    with col5:
        if st.button("Resetear conteo de paquetes del día"):
            c.execute("DELETE FROM packages_scan;")
            conn.commit()
            st.warning("Se borraron todos los pedidos escaneados del día.")

    st.markdown("---")
    st.subheader("Pedidos que **faltan** por escanear")

    scanned_set = set(scanned_codes)

    missing_orders = []
    c.execute("SELECT id, ml_order_id, buyer FROM orders;")
    for order_id, ml_order_id, buyer in c.fetchall():
        if ml_order_id not in scanned_set:
            missing_orders.append((order_id, ml_order_id, buyer))

    if not missing_orders:
        st.success("Todos los pedidos están escaneados. 🎉")
    else:
        st.warning(f"Faltan {len(missing_orders)} pedidos por escanear.")
        for order_id, ml_order_id, buyer in missing_orders:
            with st.expander(f"Venta {ml_order_id} · Cliente: {buyer}"):
                c.execute("""
                    SELECT sku_ml,
                           COALESCE(title_tec, title_ml) AS nombre_producto,
                           qty
                    FROM order_items
                    WHERE order_id = ?
                """, (order_id,))
                rows_items = c.fetchall()
                if rows_items:
                    df_items = pd.DataFrame(
                        rows_items,
                        columns=["SKU", "Producto", "Cantidad"]
                    )
                    st.table(df_items)
                else:
                    st.write("Sin líneas de productos registradas para este pedido.")

    conn.close()


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="Aurora ML – MVP Picking", layout="wide")
    init_db()

    st.sidebar.title("Aurora ML – MVP")
    page = st.sidebar.radio(
        "Menú",
        [
            "1) Importar ventas",
            "2) Picking global (producto por producto)",
            "3) Conteo final paquetes",
            "4) Admin",
        ],
    )

    if page.startswith("1"):
        page_import_ml()
    elif page.startswith("2"):
        page_picking()
    elif page.startswith("3"):
        page_conteo_final()
    elif page.startswith("4"):
        page_admin()


if __name__ == "__main__":
    main()

