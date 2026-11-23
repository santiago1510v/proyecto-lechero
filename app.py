"""Aplicación de registro y análisis básico de una finca lechera.

Lee los archivos produccion_diaria.csv y gastos.csv. Los archivos pueden estar
junto a app.py o dentro de la carpeta data.
"""
from pathlib import Path
import json
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" if (BASE / "data").exists() else BASE
ARCHIVO_PRODUCCION = DATA / "produccion_diaria.csv"
ARCHIVO_GASTOS = DATA / "gastos.csv"
ARCHIVO_VACAS = DATA / "vacas.csv"
ARCHIVO_CONFIG = DATA / "config_finca.json"

CONFIG_POR_DEFECTO = {
    "vacas_en_produccion": 20,
    "gasolina_mensual": 400000,
    "precio_litro": 2200,
}
DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

st.set_page_config(page_title="Proyecto Lechero", layout="wide")


# -----------------------------------------------------------------------------
# Funciones generales y lectura de archivos
# -----------------------------------------------------------------------------
def normalizar_texto(valor):
    texto = str(valor).strip().lower()
    return "".join(
        caracter
        for caracter in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(caracter)
    )


def dinero(valor):
    return f"${float(valor):,.0f}".replace(",", ".")


def leer_csv_seguro(ruta, columnas):
    """Lee un CSV existente o devuelve una tabla vacía con las columnas esperadas."""
    ruta = Path(ruta)
    if not ruta.exists():
        return pd.DataFrame(columns=columnas)
    try:
        return pd.read_csv(ruta, encoding="utf-8-sig").reindex(columns=columnas)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columnas)



@st.cache_data
def cargar_produccion(ruta):
    df = leer_csv_seguro(ruta, ["fecha", "litros"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["litros"] = pd.to_numeric(df["litros"], errors="coerce")
    df = df.dropna(subset=["fecha", "litros"])
    df = df[df["litros"] >= 0]

    if df.empty:
        return pd.DataFrame(columns=["fecha", "litros", "mes", "dia_semana"])

    # Varios registros de una misma fecha se suman.
    df = df.groupby("fecha", as_index=False)["litros"].sum()
    df = df.sort_values("fecha").reset_index(drop=True)
    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["dia_semana"] = df["fecha"].dt.dayofweek.map(
        lambda n: DIAS_SEMANA[n]
    )

    return df


def cargar_gastos(ruta):
    df = leer_csv_seguro(ruta, ["fecha", "concepto", "valor"])
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["concepto"] = df["concepto"].fillna("").astype(str).str.strip()
    df = df.dropna(subset=["fecha", "valor"])
    df = df[
        (df["valor"] >= 0) & (df["concepto"] != "")
    ].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "fecha",
                "concepto",
                "valor",
                "mes",
                "concepto_normalizado"
            ]
        )

    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["concepto_normalizado"] = df["concepto"].map(normalizar_texto)

    return df



def guardar_csv(df, ruta, columnas):
    DATA.mkdir(parents=True, exist_ok=True)
    salida = df.copy().reindex(columns=columnas)
    salida.to_csv(ruta, index=False, encoding="utf-8-sig")
    st.cache_data.clear()


def cargar_configuracion():
    config = CONFIG_POR_DEFECTO.copy()
    if ARCHIVO_CONFIG.exists():
        try:
            with open(ARCHIVO_CONFIG, "r", encoding="utf-8") as archivo:
                guardada = json.load(archivo)
            for clave in CONFIG_POR_DEFECTO:
                if clave in guardada:
                    config[clave] = guardada[clave]
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return config


def guardar_configuracion(config):
    DATA.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVO_CONFIG, "w", encoding="utf-8") as archivo:
        json.dump(config, archivo, ensure_ascii=False, indent=2)


def cargar_vacas():
    columnas = ["codigo", "nombre", "estado", "fecha_ingreso", "observaciones"]
    df = leer_csv_seguro(ARCHIVO_VACAS, columnas)
    df["fecha_ingreso"] = pd.to_datetime(df["fecha_ingreso"], errors="coerce")
    for columna in ["codigo", "nombre", "estado", "observaciones"]:
        df[columna] = df[columna].fillna("").astype(str)
    return df


def dias_del_periodo_en_mes(mes, inicio, fin):
    periodo = pd.Period(mes, freq="M")
    primer_dia = max(pd.Timestamp(inicio), periodo.start_time.normalize())
    ultimo_dia = min(pd.Timestamp(fin), periodo.end_time.normalize())
    if primer_dia > ultimo_dia:
        return 0
    return (ultimo_dia - primer_dia).days + 1


# -----------------------------------------------------------------------------
# Resumen financiero
# -----------------------------------------------------------------------------
def resumen_mensual(produccion, gastos, precio_litro, gasolina_mensual, inicio, fin):
    if produccion.empty:
        return pd.DataFrame()

    mensual = (
        produccion.assign(mes=produccion["fecha"].dt.to_period("M").astype(str))
        .groupby("mes", as_index=False)
        .agg(dias_registrados=("litros", "count"), litros=("litros", "sum"))
    )
    mensual["ingresos_estimados"] = mensual["litros"] * precio_litro

    if gastos.empty:
        compras = pd.DataFrame(columns=["mes", "concepto_normalizado", "valor"])
    else:
        compras = gastos.copy()
        compras["mes"] = compras["fecha"].dt.to_period("M").astype(str)
        compras["concepto_normalizado"] = compras["concepto"].map(normalizar_texto)

    por_concepto = (
        compras.groupby(["mes", "concepto_normalizado"])["valor"].sum()
        if not compras.empty else pd.Series(dtype=float)
    )

    def gasto_del_mes(mes, concepto):
        return float(por_concepto.get((mes, concepto), 0.0))

    mensual["alimento_comprado"] = mensual["mes"].map(
        lambda mes: gasto_del_mes(mes, "alimento")
    )
    mensual["medicamentos"] = mensual["mes"].map(
        lambda mes: gasto_del_mes(mes, "medicamentos")
    )
    mensual["gasolina_registrada"] = mensual["mes"].map(
        lambda mes: gasto_del_mes(mes, "gasolina")
    )

    fracciones = {}
    for mes in mensual["mes"]:
        dias_seleccionados = dias_del_periodo_en_mes(mes, inicio, fin)
        dias_mes = pd.Period(mes).days_in_month
        fracciones[mes] = dias_seleccionados / dias_mes
    mensual["fraccion_mes"] = mensual["mes"].map(fracciones)

    # Si se registró gasolina real en un mes, se usa ese registro. Si no,
    # se usa el valor mensual configurado, distribuido según los días seleccionados.
    mensual["gasolina"] = np.where(
        mensual["gasolina_registrada"] > 0,
        mensual["gasolina_registrada"],
        gasolina_mensual * mensual["fraccion_mes"],
    )

    conceptos_conocidos = {"alimento", "gasolina", "medicamentos"}
    otros = (
        compras[~compras["concepto_normalizado"].isin(conceptos_conocidos)]
        .groupby("mes")["valor"].sum()
        if not compras.empty else pd.Series(dtype=float)
    )
    mensual["otros_gastos"] = mensual["mes"].map(otros).fillna(0.0)
    mensual["costos_totales"] = (
        mensual["alimento_comprado"]
        + mensual["gasolina"]
        + mensual["medicamentos"]
        + mensual["otros_gastos"]
    )
    mensual["resultado_estimado"] = mensual["ingresos_estimados"] - mensual["costos_totales"]
    return mensual


def calcular_estadisticas(valores):
    datos = np.asarray(valores, dtype=float)
    if datos.size == 0:
        return None
    return {
        "promedio": float(np.mean(datos)),
        "mediana": float(np.median(datos)),
        "desviacion": float(np.std(datos, ddof=1)) if datos.size > 1 else 0.0,
        "minimo": float(np.min(datos)),
        "maximo": float(np.max(datos)),
    }


# -----------------------------------------------------------------------------
# Formularios para registrar información
# -----------------------------------------------------------------------------
def registrar_datos():
    st.subheader("Registrar información")
    tab_leche, tab_gastos = st.tabs(["Producción de leche", "Registrar gasto"])

    with tab_leche:
        st.write("Registra por separado los dos ordeños. El sistema guarda la suma del día.")
        with st.form("form_produccion", clear_on_submit=True):
            fecha = st.date_input("Fecha", value=pd.Timestamp.today().date())
            manana, tarde = st.columns(2)
            with manana:
                litros_manana = st.number_input("Litros de la mañana", min_value=0.0, step=0.5)
            with tarde:
                litros_tarde = st.number_input("Litros de la tarde", min_value=0.0, step=0.5)
            st.write(f"Total del día: **{litros_manana + litros_tarde:,.1f} litros**")
            guardar = st.form_submit_button("Guardar producción", type="primary")

        if guardar:
            total = float(litros_manana + litros_tarde)
            if total <= 0:
                st.error("Ingresa una cantidad mayor que cero.")
            else:
                base = leer_csv_seguro(ARCHIVO_PRODUCCION, ["fecha", "litros"])
                base["fecha"] = pd.to_datetime(base["fecha"], errors="coerce")
                base["litros"] = pd.to_numeric(base["litros"], errors="coerce")
                base = base.dropna(subset=["fecha", "litros"])
                # Un registro diario: si la fecha ya existe, se reemplaza por el nuevo total.
                base = base[base["fecha"].dt.date != fecha]
                nuevo = pd.DataFrame({"fecha": [pd.Timestamp(fecha)], "litros": [total]})
                base = pd.concat([base, nuevo], ignore_index=True).sort_values("fecha")
                base["fecha"] = base["fecha"].dt.strftime("%Y-%m-%d")
                guardar_csv(base, ARCHIVO_PRODUCCION, ["fecha", "litros"])
                st.success(f"Producción guardada: {total:,.1f} litros.")
                st.rerun()

    with tab_gastos:
        with st.form("form_gasto", clear_on_submit=True):
            fecha_gasto = st.date_input("Fecha del gasto", value=pd.Timestamp.today().date())
            tipo = st.selectbox("Concepto", ["Alimento", "Gasolina", "Medicamentos", "Otros"])
            otro = st.text_input("Describe el gasto", disabled=(tipo != "Otros"))
            valor = st.number_input("Valor pagado (COP)", min_value=0, step=1000)
            guardar_gasto = st.form_submit_button("Guardar gasto", type="primary")

        if guardar_gasto:
            concepto = otro.strip() if tipo == "Otros" else tipo
            if not concepto:
                st.error("Escribe el concepto del gasto.")
            elif valor <= 0:
                st.error("Ingresa un valor mayor que cero.")
            else:
                base = leer_csv_seguro(ARCHIVO_GASTOS, ["fecha", "concepto", "valor"])
                nuevo = pd.DataFrame({
                    "fecha": [pd.Timestamp(fecha_gasto).strftime("%Y-%m-%d")],
                    "concepto": [concepto],
                    "valor": [int(valor)],
                })
                base = pd.concat([base, nuevo], ignore_index=True)
                guardar_csv(base, ARCHIVO_GASTOS, ["fecha", "concepto", "valor"])
                st.success(f"Gasto guardado: {concepto}, {dinero(valor)}.")
                st.rerun()


# -----------------------------------------------------------------------------
# Administración y edición
# -----------------------------------------------------------------------------
def administrar_datos():
    st.subheader("Administrar información")
    seccion = st.radio(
        "Selecciona la información que deseas administrar",
        ["Producción", "Gastos", "Inventario de vacas", "Configuración"],
        horizontal=True,
    )

    if seccion == "Producción":
        base = leer_csv_seguro(ARCHIVO_PRODUCCION, ["fecha", "litros"])
        base["fecha"] = pd.to_datetime(base["fecha"], errors="coerce")
        base["litros"] = pd.to_numeric(base["litros"], errors="coerce").astype("float64")
        editado = st.data_editor(
            base,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_produccion",
            column_config={
                "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                "litros": st.column_config.NumberColumn("Litros", min_value=0.0, step=0.5),
            },
        )
        if st.button("Guardar cambios de producción", type="primary"):
            validar = editado.copy()
            validar["fecha"] = pd.to_datetime(validar["fecha"], errors="coerce")
            validar["litros"] = pd.to_numeric(validar["litros"], errors="coerce")
            validar = validar.dropna(how="all")
            if validar[["fecha", "litros"]].isna().any().any() or (validar["litros"] < 0).any():
                st.error("Revisa las fechas y los litros. No pueden estar vacíos ni ser negativos.")
            else:
                validar["fecha"] = validar["fecha"].dt.strftime("%Y-%m-%d")
                guardar_csv(validar, ARCHIVO_PRODUCCION, ["fecha", "litros"])
                st.success("Producción actualizada.")
                st.rerun()

    elif seccion == "Gastos":
        base = leer_csv_seguro(ARCHIVO_GASTOS, ["fecha", "concepto", "valor"])
        base["fecha"] = pd.to_datetime(base["fecha"], errors="coerce")
        base["concepto"] = base["concepto"].fillna("").astype(str)
        base["valor"] = pd.to_numeric(base["valor"], errors="coerce").astype("float64")
        editado = st.data_editor(
            base,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_gastos",
            column_config={
                "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
                "concepto": st.column_config.TextColumn("Concepto"),
                "valor": st.column_config.NumberColumn("Valor (COP)", min_value=0, step=1000),
            },
        )
        if st.button("Guardar cambios de gastos", type="primary"):
            validar = editado.copy()
            validar["fecha"] = pd.to_datetime(validar["fecha"], errors="coerce")
            validar["concepto"] = validar["concepto"].fillna("").astype(str).str.strip()
            validar["valor"] = pd.to_numeric(validar["valor"], errors="coerce")
            validar = validar.dropna(how="all")
            validar = validar[~(
                validar["fecha"].isna()
                & validar["concepto"].eq("")
                & validar["valor"].isna()
            )]
            if (
                validar[["fecha", "valor"]].isna().any().any()
                or validar["concepto"].eq("").any()
                or (validar["valor"] < 0).any()
            ):
                st.error("Revisa fecha, concepto y valor. No pueden estar vacíos ni ser negativos.")
            else:
                validar["fecha"] = validar["fecha"].dt.strftime("%Y-%m-%d")
                guardar_csv(validar, ARCHIVO_GASTOS, ["fecha", "concepto", "valor"])
                st.success("Gastos actualizados.")
                st.rerun()

    elif seccion == "Inventario de vacas":
        st.write("Registra únicamente los animales reales de la finca.")
        base = cargar_vacas()
        editado = st.data_editor(
            base,
            num_rows="dynamic",
            use_container_width=True,
            key="editor_vacas",
            column_config={
                "codigo": st.column_config.TextColumn("Código"),
                "nombre": st.column_config.TextColumn("Nombre"),
                "estado": st.column_config.SelectboxColumn(
                    "Estado", options=["Producción", "Seca", "Novilla", "Ternera"], required=False
                ),
                "fecha_ingreso": st.column_config.DateColumn("Fecha de ingreso", format="DD/MM/YYYY"),
                "observaciones": st.column_config.TextColumn("Observaciones"),
            },
        )
        if st.button("Guardar inventario", type="primary"):
            validar = editado.copy().dropna(how="all")
            validar["codigo"] = validar["codigo"].fillna("").astype(str).str.strip()
            validar["nombre"] = validar["nombre"].fillna("").astype(str).str.strip()
            validar["estado"] = validar["estado"].fillna("").astype(str).str.strip()
            validar["observaciones"] = validar["observaciones"].fillna("").astype(str)
            validar["fecha_ingreso"] = pd.to_datetime(validar["fecha_ingreso"], errors="coerce")
            if validar["codigo"].eq("").any():
                st.error("Cada animal debe tener un código.")
            elif validar["codigo"].duplicated().any():
                st.error("Hay códigos repetidos. Cada animal debe tener uno diferente.")
            else:
                validar["fecha_ingreso"] = validar["fecha_ingreso"].dt.strftime("%Y-%m-%d")
                DATA.mkdir(parents=True, exist_ok=True)
                validar.to_csv(ARCHIVO_VACAS, index=False, encoding="utf-8-sig")
                st.success("Inventario actualizado.")
                st.rerun()

    else:
        config = cargar_configuracion()
        with st.form("form_configuracion"):
            vacas = st.number_input(
                "Vacas actualmente en producción", min_value=0, max_value=10000,
                value=int(config["vacas_en_produccion"]), step=1,
            )
            gasolina = st.number_input(
                "Gasolina mensual de referencia (COP)", min_value=0,
                value=int(config["gasolina_mensual"]), step=10000,
            )
            precio = st.number_input(
                "Precio estimado de venta por litro (COP)", min_value=0,
                value=int(config["precio_litro"]), step=50,
            )
            guardar = st.form_submit_button("Guardar configuración", type="primary")
        if guardar:
            guardar_configuracion({
                "vacas_en_produccion": int(vacas),
                "gasolina_mensual": int(gasolina),
                "precio_litro": int(precio),
            })
            st.success("Configuración actualizada.")
            st.rerun()

    st.divider()
    st.subheader("Copias de seguridad")
    col1, col2, col3 = st.columns(3)
    with col1:
        datos = ARCHIVO_PRODUCCION.read_bytes() if ARCHIVO_PRODUCCION.exists() else b"fecha,litros\n"
        st.download_button("Descargar producción", datos, "produccion_diaria_backup.csv", mime="text/csv")
    with col2:
        datos = ARCHIVO_GASTOS.read_bytes() if ARCHIVO_GASTOS.exists() else b"fecha,concepto,valor\n"
        st.download_button("Descargar gastos", datos, "gastos_backup.csv", mime="text/csv")
    with col3:
        datos = ARCHIVO_VACAS.read_bytes() if ARCHIVO_VACAS.exists() else b"codigo,nombre,estado,fecha_ingreso,observaciones\n"
        st.download_button("Descargar inventario", datos, "vacas_backup.csv", mime="text/csv")


# -----------------------------------------------------------------------------
# Panel principal
# -----------------------------------------------------------------------------
def mostrar_resumen(produccion, mensual, vacas_en_produccion, inicio, fin):
    st.subheader("Resumen del periodo")
    if produccion.empty:
        st.info("Todavía no hay producción registrada para el periodo seleccionado.")
        return

    total_litros = float(produccion["litros"].sum())
    ingresos = float(mensual["ingresos_estimados"].sum()) if not mensual.empty else 0.0
    costos = float(mensual["costos_totales"].sum()) if not mensual.empty else 0.0
    resultado = ingresos - costos
    margen = (resultado / ingresos * 100) if ingresos else 0.0

    st.caption(
        f"{len(produccion)} días con registro | {inicio:%d/%m/%Y} al {fin:%d/%m/%Y}"
    )
    cols = st.columns(4)
    cols[0].metric("Leche registrada", f"{total_litros:,.1f} L")
    cols[1].metric("Ingresos estimados", dinero(ingresos))
    cols[2].metric("Costos incluidos", dinero(costos))
    cols[3].metric("Resultado estimado", dinero(resultado), f"Margen: {margen:.1f}%")

    izquierda, derecha = st.columns([1.3, 1])
    with izquierda:
        figura = px.line(
            produccion, x="fecha", y="litros", markers=True,
            title="Producción diaria de leche",
            labels={"fecha": "Fecha", "litros": "Litros"},
        )
        figura.update_layout(hovermode="x unified")
        st.plotly_chart(figura, use_container_width=True)
    with derecha:
        st.plotly_chart(
            px.bar(
                mensual, x="mes", y="litros", title="Producción por mes",
                labels={"mes": "Mes", "litros": "Litros"},
            ),
            use_container_width=True,
        )
    st.caption(f"Vacas configuradas en producción: {vacas_en_produccion}.")


def mostrar_produccion(produccion):
    st.subheader("Análisis de producción")
    if produccion.empty:
        st.info("No hay datos de producción en este periodo.")
        return
    stats = calcular_estadisticas(produccion["litros"])
    cols = st.columns(5)
    etiquetas = [
        ("Promedio diario", stats["promedio"]),
        ("Mediana", stats["mediana"]),
        ("Desviación estándar", stats["desviacion"]),
        ("Mínimo", stats["minimo"]),
        ("Máximo", stats["maximo"]),
    ]
    for col, (etiqueta, valor) in zip(cols, etiquetas):
        col.metric(etiqueta, f"{valor:,.1f} L")

    izquierda, derecha = st.columns(2)
    por_dia = (
        produccion.groupby("dia_semana")["litros"].mean()
        .reindex(DIAS_SEMANA).dropna().reset_index()
    )
    izquierda.plotly_chart(
        px.bar(
            por_dia, x="dia_semana", y="litros",
            title="Promedio por día de la semana",
            labels={"dia_semana": "Día", "litros": "Promedio de litros"},
        ),
        use_container_width=True,
    )
    derecha.plotly_chart(
        px.histogram(
            produccion, x="litros", nbins=10,
            title="Distribución de la producción diaria",
            labels={"litros": "Litros por día", "count": "Frecuencia"},
        ),
        use_container_width=True,
    )
    st.dataframe(produccion[["fecha", "litros", "dia_semana"]], use_container_width=True, hide_index=True)


def mostrar_finanzas(mensual, gastos):
    st.subheader("Análisis financiero")
    if mensual.empty:
        st.info("Registra producción para generar el resumen financiero.")
        return

    izquierda, derecha = st.columns(2)
    izquierda.plotly_chart(
        px.bar(
            mensual, x="mes", y=["ingresos_estimados", "costos_totales"],
            barmode="group", title="Ingresos estimados y costos",
            labels={"value": "COP", "mes": "Mes", "variable": "Concepto"},
        ),
        use_container_width=True,
    )
    componentes = {
        "Alimento comprado": float(mensual["alimento_comprado"].sum()),
        "Gasolina": float(mensual["gasolina"].sum()),
        "Medicamentos registrados": float(mensual["medicamentos"].sum()),
        "Otros gastos": float(mensual["otros_gastos"].sum()),
    }
    componentes = {nombre: valor for nombre, valor in componentes.items() if valor > 0}
    if componentes:
        derecha.plotly_chart(
            px.pie(
                names=list(componentes), values=list(componentes.values()),
                title="Composición de los costos", hole=0.35,
            ),
            use_container_width=True,
        )
    else:
        derecha.info("No hay gastos registrados en el periodo.")

    st.subheader("Detalle mensual")
    columnas = [
        "mes", "litros", "ingresos_estimados", "alimento_comprado",
        "gasolina", "medicamentos", "otros_gastos", "costos_totales", "resultado_estimado",
    ]
    nombres = {
        "mes": "Mes", "litros": "Litros", "ingresos_estimados": "Ingresos estimados",
        "alimento_comprado": "Alimento comprado", "gasolina": "Gasolina",
        "medicamentos": "Medicamentos registrados", "otros_gastos": "Otros gastos",
        "costos_totales": "Costos incluidos", "resultado_estimado": "Resultado estimado",
    }
    detalle = mensual[columnas].rename(columns=nombres).copy()
    st.dataframe(detalle.round(0), use_container_width=True, hide_index=True)

    st.caption(
        "Los ingresos se calculan con el precio por litro configurado. El alimento representa "
        "compras registradas, no necesariamente consumo del mismo mes. La gasolina usa el valor "
        "mensual de referencia si no se registra un pago real en ese mes. Los medicamentos solo "
        "se cuentan cuando se registra una compra."
    )
    if not gastos.empty:
        st.subheader("Gastos registrados")
        st.dataframe(gastos[["fecha", "concepto", "valor"]], use_container_width=True, hide_index=True)


def main():
    st.title("Proyecto Lechero")
    st.caption("Registro y análisis de producción y gastos de la finca")

    config = cargar_configuracion()
    produccion_completa = cargar_produccion(str(ARCHIVO_PRODUCCION))
    gastos_completos = cargar_gastos(str(ARCHIVO_GASTOS))

    st.sidebar.header("Periodo de análisis")
    if produccion_completa.empty:
        hoy = pd.Timestamp.today().date()
        rango = st.sidebar.date_input("Fechas", value=(hoy, hoy), format="DD/MM/YYYY")
    else:
        fecha_min = produccion_completa["fecha"].min().date()
        fecha_max = produccion_completa["fecha"].max().date()
        rango = st.sidebar.date_input(
            "Fechas", value=(fecha_min, fecha_max),
            min_value=fecha_min, max_value=fecha_max, format="DD/MM/YYYY",
        )

    if not isinstance(rango, (tuple, list)) or len(rango) != 2:
        st.info("Selecciona una fecha inicial y una fecha final.")
        inicio = fin = pd.Timestamp.today()
    else:
        inicio, fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])

    if inicio > fin:
        st.error("La fecha inicial no puede ser posterior a la fecha final.")
        return

    produccion = produccion_completa[
        (produccion_completa["fecha"] >= inicio)
        & (produccion_completa["fecha"] <= fin)
    ].copy()
    gastos = gastos_completos[
        (gastos_completos["fecha"] >= inicio)
        & (gastos_completos["fecha"] <= fin)
    ].copy()

    mensual = resumen_mensual(
        produccion,
        gastos,
        int(config["precio_litro"]),
        int(config["gasolina_mensual"]),
        inicio,
        fin,
    )

    tab_resumen, tab_produccion, tab_finanzas, tab_registrar, tab_administrar = st.tabs(
        ["Resumen", "Producción", "Finanzas", "Registrar", "Administrar"]
    )
    with tab_resumen:
        mostrar_resumen(
            produccion, mensual, int(config["vacas_en_produccion"]), inicio, fin
        )
    with tab_produccion:
        mostrar_produccion(produccion)
    with tab_finanzas:
        mostrar_finanzas(mensual, gastos)
    with tab_registrar:
        registrar_datos()
    with tab_administrar:
        administrar_datos()


if __name__ == "__main__":
    main()
