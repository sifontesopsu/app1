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
        qty INTEGER
    );
    """)

    # Picking global por SKU
    c.execute("""
    CREATE TABLE IF NOT EXISTS picking_global (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku_ml TEXT,
        title_ml TEXT,
        qty_total INTEGER,
        qty_picked INTEGER DEFAULT 0
    );
    """)

    # Paquetes escaneados en conteo final (solo tracking)
    c.execute("""
    CREATE TABLE IF NOT EXISTS packages_scan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracking_code TEXT,
        scanned_at TEXT
    );
    """)

    # Imágenes opcionales por SKU (para mostrar foto en el picking)
    c.execute("""
    CREATE TABLE IF NOT EXISTS sku_images (
        sku_ml TEXT PRIMARY KEY,
        image_url TEXT
    );
    """)

    conn.commit()
    conn.close()


# ---------- PÁGINA: ADMIN ----------
def page_admin():
    st.header("0) Panel de administrador")

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

    c.execute("SELECT COUNT(*) FROM packages_scan;")
    total_packages_scanned = c.fetchone()[0] or 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pedidos cargados", total_orders)
    col2.metric("Líneas de ítems", total_items)
    col3.metric("SKUs en picking_global", total_skus)
    col4.metric("Paquetes escaneados (tracking)", total_packages_scanned)

    st.markdown("---")
    st.subheader("Acciones de administración")

    col_a, col_b = st.columns(2)

    # Resetear solo cantidades pickeadas
    with col_a:
        st.write("**Resetear SOLO picking (qty_picked = 0)**")
        st.caption("No borra pedidos ni ítems, solo deja todas las cantidades pickeadas en 0.")
        if st.button("Resetear cantidades pickeadas"):
            c.execute("UPDATE picking_global SET qty_picked = 0;")
            conn.commit()
            st.success("Se reiniciaron todas las cantidades pickeadas.")

    # Resetear TODO el sistema
    with col_b:
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

    st.markdown("""
    Opcionalmente, puedes subir un archivo CSV con columnas:
    **sku_ml, image_url** para mostrar la foto del producto al piqueador.
    """)
    img_file = st.file_uploader(
        "Archivo opcional de imágenes (CSV con columnas sku_ml,image_url)",
        type=["csv"],
        key="img_uploader",
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

    if st.button("Cargar ventas en el sistema"):
        conn = get_conn()
        c = conn.cursor()

        # Limpiar datos anteriores (MVP diario)
        c.execute("DELETE FROM order_items;")
        c.execute("DELETE FROM orders;")
        c.execute("DELETE FROM picking_global;")
        c.execute("DELETE FROM packages_scan;")

        # También limpiamos imágenes si se va a subir archivo nuevo
        if img_file is not None:
            c.execute("DELETE FROM sku_images;")

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
                sku = str(row["sku_ml"])
                title = str(row["title_ml"])
                qty = int(row["qty"])
                if qty <= 0:
                    continue

                c.execute("""
                    INSERT INTO order_items (order_id, sku_ml, title_ml, qty)
                    VALUES (?, ?, ?, ?)
                """, (order_id, sku, title, qty))

        # Generar picking_global agrupando todos los SKUs del día
        c.execute("""
            SELECT sku_ml, title_ml, SUM(qty) as total
            FROM order_items
            GROUP BY sku_ml, title_ml
        """)
        rows = c.fetchall()
        for sku, title, total in rows:
            c.execute("""
                INSERT INTO picking_global (sku_ml, title_ml, qty_total, qty_picked)
                VALUES (?, ?, ?, 0)
            """, (sku, title, total))

        # Si se subió archivo de imágenes, guardarlo en BD
        if img_file is not None:
            try:
                img_df = pd.read_csv(img_file)
                if "sku_ml" in img_df.columns and "image_url" in img_df.columns:
                    for _, r in img_df.iterrows():
                        sku_img = str(r["sku_ml"])
                        url = str(r["image_url"])
                        if sku_img and url:
                            c.execute("""
                                INSERT OR REPLACE INTO sku_images (sku_ml, image_url)
                                VALUES (?, ?)
                            """, (sku_img, url))
                else:
                    st.warning("El archivo de imágenes no tiene columnas sku_ml,image_url. Se ignoró.")
            except Exception as e:
                st.warning(f"No se pudo procesar el archivo de imágenes: {e}")

        conn.commit()
        conn.close()

        # Resetear índice de producto actual del pickeador
        st.session_state["pick_index"] = 0

        st.success("Ventas cargadas y lista de picking generada correctamente.")


# ---------- PÁGINA: PICKING GLOBAL (PRODUCTO POR PRODUCTO) ----------
def page_picking():
    st.header("2) Picking global – Producto por producto")

    st.write("""
    Esta pantalla está pensada para el piqueador:
    ve **un producto a la vez**, con su título, SKU, cantidad total y pickeada,
    puede sumar/restar con botones (1, 2, 3 clics) y pasar al siguiente producto.
    """)

    conn = get_conn()
    c = conn.cursor()

    # Traer todos los productos de picking_global, junto con URL de imagen si existe
    c.execute("""
        SELECT pg.sku_ml, pg.title_ml, pg.qty_total, pg.qty_picked, si.image_url
        FROM picking_global pg
        LEFT JOIN sku_images si ON si.sku_ml = pg.sku_ml
        ORDER BY pg.title_ml
    """)
    rows = c.fetchall()

    if not rows:
        st.info("Aún no hay lista de picking. Primero importa ventas de Mercado Libre.")
        conn.close()
        return

    # Controlar índice de producto actual en session_state
    if "pick_index" not in st.session_state:
        st.session_state["pick_index"] = 0

    total_productos = len(rows)
    idx = st.session_state["pick_index"]

    if idx < 0:
        idx = 0
    if idx >= total_productos:
        idx = total_productos - 1
    st.session_state["pick_index"] = idx

    # Producto actual
    sku_ml, title_ml, qty_total, qty_picked, image_url = rows[idx]
    restantes = qty_total - qty_picked

    # Mostrar info del producto actual
    st.subheader(f"Producto {idx+1} de {total_productos}")

    col_info, col_img = st.columns([2, 1])

    with col_info:
        st.markdown(f"**SKU ML:** `{sku_ml}`")
        st.markdown(f"**Título (ML):** {title_ml}")
        st.markdown(f"**Cantidad total a pickear:** {qty_total}")
        st.markdown(f"**Cantidad pickeada:** {qty_picked}")
        st.markdown(f"**Restantes:** {restantes}")

    with col_img:
        if image_url and isinstance(image_url, str) and image_url.strip():
            try:
                st.image(image_url, caption="Imagen del producto", use_container_width=True)
            except Exception:
                st.write("No se pudo cargar la imagen.")
                st.write(image_url)
        else:
            st.write("Sin imagen configurada para este SKU.")

    st.markdown("---")

    # Controles de picking tipo "1, 2, 3 clics"
    st.write("**Actualizar cantidad pickeada**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("+1 unidad"):
            if qty_picked < qty_total:
                c.execute("""
                    UPDATE picking_global
                    SET qty_picked = qty_picked + 1
                    WHERE sku_ml = ?
                """, (sku_ml,))
                conn.commit()
                st.rerun()
            else:
                st.warning("Ya alcanzaste la cantidad total de este producto.")

    with col2:
        if st.button("-1 unidad"):
            if qty_picked > 0:
                c.execute("""
                    UPDATE picking_global
                    SET qty_picked = qty_picked - 1
                    WHERE sku_ml = ?
                """, (sku_ml,))
                conn.commit()
                st.rerun()
            else:
                st.warning("La cantidad pickeada ya es 0.")

    with col3:
        if st.button("Marcar completo"):
            if qty_picked < qty_total:
                c.execute("""
                    UPDATE picking_global
                    SET qty_picked = ?
                    WHERE sku_ml = ?
                """, (qty_total, sku_ml))
                conn.commit()
            # Pasar automáticamente al siguiente producto
            st.session_state["pick_index"] = min(idx + 1, total_productos - 1)
            st.rerun()

    with col4:
        if st.button("Poner en 0"):
            c.execute("""
                UPDATE picking_global
                SET qty_picked = 0
                WHERE sku_ml = ?
            """, (sku_ml,))
            conn.commit()
            st.rerun()

    st.markdown("---")
    st.write("**Navegación entre productos**")
    col_prev, col_next = st.columns(2)

    with col_prev:
        if st.button("⬅️ Producto anterior"):
            st.session_state["pick_index"] = max(idx - 1, 0)
            st.rerun()

    with col_next:
        if st.button("Producto siguiente ➡️"):
            st.session_state["pick_index"] = min(idx + 1, total_productos - 1)
            st.rerun()

    conn.close()


# ---------- PÁGINA: CONTEO FINAL DE PAQUETES ----------
def page_conteo_final():
    st.header("3) Conteo final de paquetes (tracking)")

    st.write("""
    Después de embalar y etiquetar, en la zona de despacho
    escanea una sola vez el código de tracking de cada paquete.
    La app compara cuántos pedidos hay vs cuántos paquetes escaneados.
    """)

    conn = get_conn()
    c = conn.cursor()

    # Pedidos esperados (ventas ML)
    c.execute("SELECT COUNT(DISTINCT ml_order_id) FROM orders;")
    expected_orders = c.fetchone()[0] or 0

    # Paquetes escaneados
    c.execute("SELECT COUNT(*) FROM packages_scan;")
    scanned_packages = c.fetchone()[0] or 0

    st.subheader("Resumen")
    col1, col2, col3 = st.columns(3)
    col1.metric("Pedidos (ventas ML)", expected_orders)
    col2.metric("Paquetes escaneados", scanned_packages)
    diff = scanned_packages - expected_orders
    col3.metric("Diferencia (escaneados - pedidos)", diff)

    st.markdown("---")
    st.subheader("Escanear tracking de paquetes")

    tracking = st.text_input("Escanee código de tracking aquí", key="scan_tracking")

    col4, col5 = st.columns(2)
    with col4:
        if st.button("Registrar paquete"):
            if not tracking:
                st.warning("Escanee o escriba un tracking primero.")
            else:
                # Evitar duplicados exactos (mismo tracking)
                c.execute("SELECT COUNT(*) FROM packages_scan WHERE tracking_code = ?;", (tracking,))
                exists = c.fetchone()[0]
                if exists:
                    st.error("Este tracking ya fue escaneado antes (paquete duplicado).")
                else:
                    c.execute("""
                        INSERT INTO packages_scan (tracking_code, scanned_at)
                        VALUES (?, ?)
                    """, (tracking, datetime.now().isoformat()))
                    conn.commit()
                    st.success(f"Tracking {tracking} registrado.")

    with col5:
        if st.button("Resetear conteo de paquetes del día"):
            c.execute("DELETE FROM packages_scan;")
            conn.commit()
            st.warning("Se borraron todos los paquetes escaneados.")

    conn.close()


# ---------- MAIN ----------
def main():
    st.set_page_config(page_title="Aurora ML – MVP Picking", layout="wide")
    init_db()

    st.sidebar.title("Aurora ML – MVP")
    page = st.sidebar.radio(
        "Menú",
        [
            "0) Admin",
            "1) Importar ventas ML",
            "2) Picking global (producto por producto)",
            "3) Conteo final paquetes",
        ],
    )

    if page.startswith("0"):
        page_admin()
    elif page.startswith("1"):
        page_import_ml()
    elif page.startswith("2"):
        page_picking()
    elif page.startswith("3"):
        page_conteo_final()


if __name__ == "__main__":
    main()
