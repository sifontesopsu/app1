import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

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
        title_ml TEXT,
        title_tec TEXT,
        qty INTEGER
    );
    """)

    # Picking global por SKU (incluye picker_id para distribución)
    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_global (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku_ml TEXT,
        title_ml TEXT,
        title_tec TEXT,
        qty_total INTEGER,
        qty_picked INTEGER DEFAULT 0,
        picker_id INTEGER
    );
    """)

    # Paquetes escaneados en conteo final (guardamos aquí el número de venta / tracking)
    c.execute("""
    CREATE TABLE IF NOT EXISTS packages_scan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracking_code TEXT,
        scanned_at TEXT
    );
    """)

    # Imágenes opcionales por SKU (para el futuro)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sku_images (
        sku_ml TEXT PRIMARY KEY,
        image_url TEXT
    );
    """)

    # Tabla de piqueadores
    c.execute("""
    CREATE TABLE IF NOT EXISTS pickers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    );
    """)

    # Asegurar columnas por si vienes de una versión vieja
    c.execute("PRAGMA table_info(order_items);")
    cols_oi = [row[1] for row in c.fetchall()]
    if "title_tec" not in cols_oi:
        c.execute("ALTER TABLE order_items ADD COLUMN title_tec TEXT;")

    c.execute("PRAGMA table_info(picking_global);")
    cols_pg = [row[1] for row in c.fetchall()]
    if "title_tec" not in cols_pg:
        c.execute("ALTER TABLE picking_global ADD COLUMN title_tec TEXT;")
    if "picker_id" not in cols_pg:
        c.execute("ALTER TABLE picking_global ADD COLUMN picker_id INTEGER;")

    conn.commit()
    conn.close()


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

    # Si ya está autenticado:
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
        st.caption("Borra todos los códigos de tracking escaneados, pero mantiene pedidos e ítems.")
        if st.button("Resetear paquetes escaneados"):
            c.execute("DELETE FROM packages_scan;")
            conn.commit()
            st.warning("Se borraron todos los paquetes escaneados.")

    # Resetear TODO el sistema
    with col_c:
        st.write("**Resetear TODO el sistema**")
        st.caption("Borra pedidos, ítems, picking, paquetes escaneados e imágenes. Úsalo solo al cambiar de día o en pruebas.")
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


# ---------- PÁGINA: IMPORTAR VENTAS ML ----------
def page_import_ml():
    st.header("1) Importar ventas Mercado Libre")

    st.write("Sube el archivo de ventas del día exportado desde Mercado Libre (XLSX).")
    file = st.file_uploader("Archivo de ventas ML (.xlsx)", type=["xlsx"])

    # Cantidad de piqueadores para distribución
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

    if file is None:
        st.info("Esperando archivo de ventas de Mercado Libre...")
        return

    # 1) LEER EL EXCEL DE ML CON DOBLE ENCABEZADO (FORMATO REAL ML)
    try:
        df = pd.read_excel(file, header=[4, 5])
    except Exception as e:
        st.error(f"Error leyendo el Excel de ML: {e}")
        st.stop()

    # Aplanar MultiIndex de columnas: "Ventas | # de venta", etc.
    df.columns = [
        " | ".join([str(x) for x in col if str(x) != "nan"])
        for col in df.columns
    ]

    # 2) COLUMNAS REALES QUE VAMOS A USAR DE TU ARCHIVO
    COLUMN_ORDER_ID = "Ventas | # de venta"
    COLUMN_QTY = "Ventas | Unidades"
    COLUMN_SKU = "Publicaciones | SKU"
    COLUMN_TITLE = "Publicaciones | Título de la publicación"
    COLUMN_BUYER = "Compradores | Comprador"

    required_cols = [COLUMN_ORDER_ID, COLUMN_QTY, COLUMN_SKU, COLUMN_TITLE, COLUMN_BUYER]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Faltan columnas en el archivo de Mercado Libre: {missing}")
        st.stop()

    st.subheader("Vista previa (columnas relevantes)")
    st.dataframe(df[required_cols].head())

    # 3) RENOMBRAR A NOMBRES INTERNOS SIMPLES
    work_df = df[required_cols].copy()
    work_df.columns = ["ml_order_id", "qty", "sku_ml", "title_ml", "buyer"]

    # LIMPIEZA DE CANTIDADES
    work_df["qty"] = pd.to_numeric(work_df["qty"], errors="coerce").fillna(0).astype(int)
    work_df = work_df[work_df["qty"] > 0]

    if work_df.empty:
        st.error("Después de limpiar las cantidades, no quedó ninguna línea con qty > 0.")
        st.stop()

    # Cargar maestro de SKUs → dict {sku: nombre_técnico}
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

    if st.button("Cargar ventas en el sistema"):
        conn = get_conn()
        c = conn.cursor()

        # Limpiar datos anteriores (MVP diario)
        c.execute("DELETE FROM order_items;")
        c.execute("DELETE FROM orders;")
        c.execute("DELETE FROM picking_global;")
        c.execute("DELETE FROM packages_scan;")
        c.execute("DELETE FROM pickers;")
        # sku_images se deja tal cual (no estamos usando fotos aún)

        # Insertar pedidos y sus líneas
        for ml_order_id, grupo in work_df.groupby("ml_order_id"):
            buyer = str(grupo["buyer"].iloc[0])
            created_at = datetime.now().isoformat()

            c.execute("""
                INSERT INTO orders (ml_order_id, buyer, created_at)
                VALUES (?, ?, ?)
            """, (str(ml_order_id), buyer, created_at))
            order_id = c.lastrowid

            for _, row in grupo.iterrows():
                sku = str(row["sku_ml"]).strip()
                title = str(row["title_ml"])
                qty = int(row["qty"])
                if qty <= 0:
                    continue

                title_tec = inv_map.get(sku)  # puede ser None si no está en maestro

                c.execute("""
                    INSERT INTO order_items (order_id, sku_ml, title_ml, title_tec, qty)
                    VALUES (?, ?, ?, ?, ?)
                """, (order_id, sku, title, title_tec, qty))

        # Generar picking_global agrupando todos los SKUs del día
        c.execute("""
            SELECT sku_ml, title_ml, title_tec, SUM(qty) as total
            FROM order_items
            GROUP BY sku_ml, title_ml, title_tec
        """)
        rows = c.fetchall()
        for sku, title_ml, title_tec, total in rows:
            c.execute("""
                INSERT INTO picking_global (sku_ml, title_ml, title_tec, qty_total, qty_picked, picker_id)
                VALUES (?, ?, ?, ?, 0, NULL)
            """, (sku, title_ml, title_tec, total))

        # Crear piqueadores y repartir SKUs en forma equitativa
        num_pickers_int = int(num_pickers)
        for i in range(num_pickers_int):
            name = f"P{i+1}"
            c.execute("INSERT INTO pickers (name) VALUES (?);", (name,))

        c.execute("SELECT id, name FROM pickers ORDER BY id;")
        pickers = c.fetchall()
        total_pickers = len(pickers)

        if total_pickers > 0:
            # Ordenamos por nombre técnico si existe, si no por título ML
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

        conn.commit()
        conn.close()

        # Resetear índice de producto actual del pickeador
        st.session_state["pick_index"] = 0

        st.success("Ventas cargadas, picking generado y distribuido entre piqueadores correctamente.")


# ---------- PÁGINA: PICKING GLOBAL (PRODUCTO POR PRODUCTO) ----------
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

    # Traer productos (incluyendo nombre técnico)
    c.execute("""
        SELECT pg.sku_ml, pg.title_ml, pg.title_tec, pg.qty_total, pg.qty_picked, si.image_url, pg.picker_id
        FROM picking_global pg
        LEFT JOIN sku_images si ON si.sku_ml = pg.sku_ml
        ORDER BY 
            CASE WHEN pg.title_tec IS NULL OR pg.title_tec = '' THEN 1 ELSE 0 END,
            pg.title_tec,
            pg.title_ml
    """)
    rows = c.fetchall()

    # Filtrar por piqueador
    if selected_picker != "Todos":
        target_id = picker_id_map.get(selected_picker)
        rows = [r for r in rows if r[6] == target_id]

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

    sku_ml, title_ml, title_tec, qty_total, qty_picked, image_url, picker_id = rows[idx]
    restantes = qty_total - qty_picked
    is_complete = restantes <= 0

    # Título a mostrar: técnico si existe, si no el de ML
    display_title = title_tec if title_tec and str(title_tec).strip().lower() not in ["", "nan"] else title_ml

    # ======== ESTILO VISUAL Y RESPONSIVE ==========
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

        .kpi-complete {
            background-color: #e1ffe1 !important;
            color: #0a7a0a !important;
            border: 2px solid #0a7a0a;
        }

        .btn-big button {
            height: 80px !important;
            font-size: 20px !important;
            font-weight: bold !important;
            width: 100% !important;
        }

        /* En pantallas más angostas (PDA / móvil), agrandar aún más */
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

    # ======== KPIs GRANDES =========
    col1, col2, col3 = st.columns(3)

    extra_class = " kpi-complete" if is_complete else ""

    with col1:
        st.markdown(
            f"<div class='kpi-box kpi-total{extra_class}'><div class='kpi-label'>TOTAL</div>{qty_total}</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<div class='kpi-box kpi-picked{extra_class}'><div class='kpi-label'>PICKEADO</div>{qty_picked}</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"<div class='kpi-box kpi-rest{extra_class}'><div class='kpi-label'>RESTANTE</div>{restantes}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ======== BOTONES GRANDES DE ACCIÓN =========
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("<div class='btn-big'>", unsafe_allow_html=True)
        if st.button("➕ SUMAR"):
            if qty_picked < qty_total:
                c.execute("""
                    UPDATE picking_global
                    SET qty_picked = qty_picked + 1
                    WHERE sku_ml = ?
                """, (sku_ml,))
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
                    WHERE sku_ml = ?
                """, (sku_ml,))
                conn.commit()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown("<div class='btn-big'>", unsafe_allow_html=True)
        if st.button("✅ COMPLETAR"):
            c.execute("""
                UPDATE picking_global
                SET qty_picked = ?
                WHERE sku_ml = ?
            """, (qty_total, sku_ml))
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
                WHERE sku_ml = ?
            """, (sku_ml,))
            conn.commit()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ======== NAVEGACIÓN GRANDE =========
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


# ---------- PÁGINA: CONTEO FINAL DE PAQUETES ----------
def page_conteo_final():
    st.header("3) Conteo final de paquetes (tracking / número de venta)")

    st.write("""
    Después de embalar y etiquetar, en la zona de despacho
    escanea una sola vez el **número de venta de Mercado Libre** (o el código que uses
    en la etiqueta) para marcar ese pedido como contado.
    """)

    conn = get_conn()
    c = conn.cursor()

    # Pedidos esperados (ventas ML)
    c.execute("SELECT DISTINCT ml_order_id FROM orders;")
    orders_all = [row[0] for row in c.fetchall()]
    expected_orders = len(orders_all)

    # Paquetes escaneados (usamos tracking_code como "número de venta escaneado")
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
                # Evitar duplicados exactos (mismo código)
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

    # Cálculo de pedidos no escaneados:
    # Asumimos que tracking_code = ml_order_id (o al menos lo que tú escaneas coincide)
    missing_orders = []
    scanned_set = set(scanned_codes)

    c.execute("SELECT id, ml_order_id, buyer FROM orders;")
    for order_id, ml_order_id, buyer in c.fetchall():
        if ml_order_id not in scanned_set:
            missing_orders.append((order_id, ml_order_id, buyer))

    if not missing_orders:
        st.success("Todos los pedidos están escaneados. 🎉")
    else:
        st.warning(f"Faltan {len(missing_orders)} pedidos por escanear.")
        # Mostrar detalle: número de venta, cliente y productos
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
            "1) Importar ventas ML",
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
