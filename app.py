# app.py
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

DB_NAME = "aurora_ml.db"

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

    # Paquetes escaneados en conteo final
    c.execute("""
    CREATE TABLE IF NOT EXISTS packages_scan (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracking_code TEXT,
        scanned_at TEXT
    );
    """)

    conn.commit()
    conn.close()

# ---------- PÁGINA: IMPORTAR VENTAS ML ----------
def page_import_ml():
    st.header("1) Importar ventas Mercado Libre")

    st.write("Sube el archivo de ventas del día exportado desde Mercado Libre (CSV o XLSX).")

    file = st.file_uploader("Archivo de ventas ML", type=["csv", "xlsx"])

    st.markdown("""
    **Suposición básica de columnas (puedes adaptar en el código si cambian):**
    - `order_id` → ID del pedido ML  
    - `title` → título del producto en la venta  
    - `seller_custom_field` (o `sku`) → SKU que usas para escanear  
    - `buyer_nickname` (o similar) → comprador  
    - `quantity` → cantidad vendida de ese ítem
    """)

    if file is not None:
        # Leer archivo
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.subheader("Vista previa de las primeras filas")
        st.dataframe(df.head())

        # Aquí mapeamos columnas reales a las que usaremos internamente.
        # Ajusta estos nombres según EXACTAMENTE cómo viene tu archivo de ML.
        COLUMN_ORDER_ID = "order_id"
        COLUMN_TITLE = "title"
        COLUMN_SKU = "seller_custom_field"  # o "sku" si usas otra columna
        COLUMN_BUYER = "buyer_nickname"
        COLUMN_QTY = "quantity"

        # Verificar que existan
        missing_cols = [
            col for col in [COLUMN_ORDER_ID, COLUMN_TITLE, COLUMN_SKU, COLUMN_BUYER, COLUMN_QTY]
            if col not in df.columns
        ]
        if missing_cols:
            st.error(f"Faltan columnas en el archivo ML: {missing_cols}. Ajusta los nombres en el código.")
            return

        if st.button("Cargar ventas en el sistema"):
            conn = get_conn()
            c = conn.cursor()

            # Limpiamos tablas del día anterior (para MVP simple)
            c.execute("DELETE FROM order_items;")
            c.execute("DELETE FROM orders;")
            c.execute("DELETE FROM picking_global;")
            c.execute("DELETE FROM packages_scan;")

            # Insertar pedidos y líneas
            for ml_order_id, grupo in df.groupby(COLUMN_ORDER_ID):
                buyer = grupo[COLUMN_BUYER].iloc[0]
                created_at = datetime.now().isoformat()

                c.execute("""
                    INSERT INTO orders (ml_order_id, buyer, created_at)
                    VALUES (?, ?, ?)
                """, (str(ml_order_id), str(buyer), created_at))
                order_id = c.lastrowid

                for _, row in grupo.iterrows():
                    sku = str(row[COLUMN_SKU])
                    title = str(row[COLUMN_TITLE])
                    qty = int(row[COLUMN_QTY])

                    c.execute("""
                        INSERT INTO order_items (order_id, sku_ml, title_ml, qty)
                        VALUES (?, ?, ?, ?)
                    """, (order_id, sku, title, qty))

            # Generar picking_global
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

            conn.commit()
            conn.close()

            st.success("Ventas cargadas y lista de picking generada.")

# ---------- PÁGINA: PICKING GLOBAL ----------
def page_picking():
    st.header("2) Picking global por producto")

    st.write("Usa el PDA para escanear el SKU (seller_custom_field) cada vez que bajas un producto a la zona.")

    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT sku_ml, title_ml, qty_total, qty_picked
        FROM picking_global
        ORDER BY title_ml
    """)
    rows = c.fetchall()

    if not rows:
        st.info("Aún no hay lista de picking. Primero importa ventas ML.")
        conn.close()
        return

    df = pd.DataFrame(rows, columns=["SKU ML", "Título producto", "Cantidad total", "Cantidad pickeada"])
    st.subheader("Resumen de picking")
    st.dataframe(df, use_container_width=True)

    st.markdown("---")
    st.subheader("Registrar escaneo")

    scanned_sku = st.text_input("Escanee SKU aquí (PDA)", key="scan_picking")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Registrar escaneo"):
            if not scanned_sku:
                st.warning("Escanee o escriba un SKU ML primero.")
            else:
                # Buscar SKU en picking_global
                c.execute("""
                    SELECT id, qty_total, qty_picked, title_ml
                    FROM picking_global
                    WHERE sku_ml = ?
                """, (scanned_sku,))
                row = c.fetchone()
                if not row:
                    st.error("Ese SKU ML no está en la lista de picking.")
                else:
                    pg_id, qty_total, qty_picked, title_ml = row
                    if qty_picked >= qty_total:
                        st.warning(f"Ya pickeaste todas las unidades de este producto ({qty_picked}/{qty_total}).")
                    else:
                        new_picked = qty_picked + 1
                        c.execute("""
                            UPDATE picking_global
                            SET qty_picked = ?
                            WHERE id = ?
                        """, (new_picked, pg_id))
                        conn.commit()
                        st.success(
                            f"{title_ml} | SKU {scanned_sku} → {new_picked}/{qty_total} unidades pickeadas."
                        )

    with col2:
        if st.button("Resetear pickeos (solo visual, cuidado)"):
            c.execute("UPDATE picking_global SET qty_picked = 0;")
            conn.commit()
            st.warning("Se reiniciaron las cantidades pickeadas (para el día actual).")

    conn.close()

# ---------- PÁGINA: CONTEO FINAL DE PAQUETES ----------
def page_conteo_final():
    st.header("3) Conteo final de paquetes (tracking)")

    st.write("""
    Idea: después de embalar y etiquetar, en la zona de despacho
    escanean **una vez** el código de tracking de cada paquete.
    La app compara cuántos pedidos hay vs cuántos paquetes escaneados.
    """)

    conn = get_conn()
    c = conn.cursor()

    # Pedidos esperados
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
    page = st.sidebar.radio("Menú", ["1) Importar ventas ML", "2) Picking global", "3) Conteo final paquetes"])

    if page.startswith("1"):
        page_import_ml()
    elif page.startswith("2"):
        page_picking()
    elif page.startswith("3"):
        page_conteo_final()

if __name__ == "__main__":
    main()
