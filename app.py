"""Dashboard de exploración y análisis económico de una finca lechera.

La aplicación separa los gastos registrados de los escenarios simulados.
Los datos se leen desde produccion_diaria.csv y gastos.csv.
"""
from pathlib import Path
import json
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" if (BASE / "data").exists() else BASE
ARCHIVO_PRODUCCION = DATA / "produccion_diaria.csv"
ARCHIVO_GASTOS = DATA / "gastos.csv"
ARCHIVO_VACAS = DATA / "vacas.csv"
ARCHIVO_CONFIG = DATA / "config_finca.json"
PRECIO_REFERENCIA = 2200
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

st.set_page_config(page_title="Proyecto Lechero", page_icon="🐄", layout="wide")


# ------------------------- Carga y validación -------------------------
def normalizar_texto(valor):
    """Normaliza texto para comparar conceptos sin depender de tildes o mayúsculas."""
    texto = str(valor).strip().lower()
    return "".join(
        caracter for caracter in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(caracter)
    )


@st.cache_data

def cargar_produccion(ruta, vacas_en_produccion=20):
    """Carga, valida y organiza la producción diaria."""
    df = pd.read_csv(ruta)
    requeridas = {"fecha", "litros"}
    if not requeridas.issubset(df.columns):
        raise ValueError("produccion_diaria.csv debe contener las columnas fecha y litros.")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["litros"] = pd.to_numeric(df["litros"], errors="coerce")
    df = df.dropna(subset=["fecha", "litros"])
    df = df[df["litros"] >= 0]
    # Si hay más de un registro por fecha, se suman: pueden ser varias entregas del día.
    df = df.groupby("fecha", as_index=False)["litros"].sum()
    df = df.sort_values("fecha").reset_index(drop=True)
    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["dia_semana"] = df["fecha"].dt.dayofweek.map(lambda numero: DIAS[numero])
    df["litros_por_vaca"] = df["litros"] / max(vacas_en_produccion, 1)
    return df


@st.cache_data

def cargar_gastos(ruta):
    """Carga los gastos reales registrados; si no hay archivo, devuelve tabla vacía."""
    if not Path(ruta).exists():
        return pd.DataFrame(columns=["fecha", "concepto", "valor", "mes", "concepto_normalizado"])
    df = pd.read_csv(ruta)
    requeridas = {"fecha", "concepto", "valor"}
    if not requeridas.issubset(df.columns):
        raise ValueError("gastos.csv debe contener las columnas fecha, concepto y valor.")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df["concepto"] = df["concepto"].fillna("Sin concepto").astype(str).str.strip()
    df = df.dropna(subset=["fecha", "valor"])
    df = df[df["valor"] >= 0].copy()
    df["mes"] = df["fecha"].dt.to_period("M").astype(str)
    df["concepto_normalizado"] = df["concepto"].map(normalizar_texto)
    return df


def dinero(valor):
    """Formatea valores monetarios en pesos colombianos."""
    return f"${valor:,.0f}".replace(",", ".")


def estadisticas(valores):
    """Calcula estadísticas descriptivas con NumPy."""
    datos = np.asarray(valores, dtype=float)
    if datos.size == 0:
        return {}
    media = float(np.mean(datos))
    desviacion = float(np.std(datos, ddof=1)) if datos.size > 1 else 0.0
    return {
        "Días registrados": int(datos.size),
        "Promedio diario": media,
        "Mediana": float(np.median(datos)),
        "Desviación estándar": desviacion,
        "Mínimo": float(np.min(datos)),
        "Máximo": float(np.max(datos)),
    }


def dias_del_periodo_en_mes(mes, inicio, fin):
    """Cuenta días calendario seleccionados dentro de un mes."""
    periodo = pd.Period(mes, freq="M")
    primer_dia = max(inicio, periodo.start_time.normalize())
    ultimo_dia = min(fin, periodo.end_time.normalize())
    if primer_dia > ultimo_dia:
        return 0
    return (ultimo_dia - primer_dia).days + 1


def resumen_mensual(produccion, gastos, precio_litro, gasolina_mensual,
                    ajuste_alimento, mano_obra_mensual, inicio, fin):
    """Construye resultados reales y escenarios sin mezclar sus conceptos."""
    mensual = (produccion.groupby("mes", as_index=False)
               .agg(dias_registrados=("litros", "count"), litros=("litros", "sum")))
    mensual["ingresos_estimados"] = mensual["litros"] * precio_litro

    conceptos = gastos.groupby(["mes", "concepto_normalizado"])["valor"].sum()

    def gasto_concepto(mes, concepto):
        return float(conceptos.get((mes, concepto), 0.0))

    mensual["alimento_comprado"] = mensual["mes"].map(
        lambda mes: gasto_concepto(mes, "alimento"))
    mensual["medicamentos"] = mensual["mes"].map(
        lambda mes: gasto_concepto(mes, "medicamentos"))
    mensual["gasolina_registrada"] = mensual["mes"].map(
        lambda mes: gasto_concepto(mes, "gasolina"))

    # Si el CSV tiene gasolina registrada en el mes, se usa ese valor.
    # Si no, se aplica el valor fijo mensual configurado, proporcional al rango seleccionado.
    fracciones = {}
    for mes in mensual["mes"]:
        dias_seleccionados = dias_del_periodo_en_mes(mes, inicio, fin)
        dias_mes = pd.Period(mes).days_in_month
        fracciones[mes] = dias_seleccionados / dias_mes
    mensual["fraccion_mes_seleccionada"] = mensual["mes"].map(fracciones)
    mensual["gasolina"] = np.where(
        mensual["gasolina_registrada"] > 0,
        mensual["gasolina_registrada"],
        gasolina_mensual * mensual["fraccion_mes_seleccionada"]
    )

    gastos_conocidos = {"alimento", "medicamentos", "gasolina"}
    otros_por_mes = (gastos[~gastos["concepto_normalizado"].isin(gastos_conocidos)]
                     .groupby("mes")["valor"].sum())
    mensual["otros_gastos"] = mensual["mes"].map(otros_por_mes).fillna(0.0)

    mensual["costos_registrados"] = (
        mensual["alimento_comprado"] + mensual["medicamentos"]
        + mensual["gasolina"] + mensual["otros_gastos"]
    )
    mensual["utilidad_registrada"] = mensual["ingresos_estimados"] - mensual["costos_registrados"]

    # Escenarios: no se presentan como gastos que ya ocurrieron.
    mensual["ajuste_alimento_simulado"] = mensual["alimento_comprado"] * ajuste_alimento / 100
    mensual["mano_obra_simulada"] = mano_obra_mensual * mensual["fraccion_mes_seleccionada"]
    mensual["costos_escenario"] = (
        mensual["costos_registrados"] + mensual["ajuste_alimento_simulado"]
        + mensual["mano_obra_simulada"]
    )
    mensual["utilidad_escenario"] = mensual["ingresos_estimados"] - mensual["costos_escenario"]
    return mensual


def marcar_atipicos(valores):
    """Identifica valores atípicos mediante mediana y desviación absoluta mediana."""
    datos = np.asarray(valores, dtype=float)
    mediana = np.median(datos)
    mad = np.median(np.abs(datos - mediana))
    if mad == 0:
        return np.zeros(len(datos), dtype=bool)
    puntuacion = 0.6745 * (datos - mediana) / mad
    return np.abs(puntuacion) > 3.5


# ---------------------- Edición y guardado de datos ----------------------
def guardar_csv(df, ruta, columnas):
    """Guarda una tabla editable en CSV y limpia la caché de lectura."""
    salida = df.copy()
    for columna in columnas:
        if columna not in salida.columns:
            salida[columna] = ""
    salida = salida[columnas]
    salida.to_csv(ruta, index=False, encoding="utf-8-sig")
    st.cache_data.clear()


def cargar_configuracion():
    """Carga parámetros básicos persistidos en un archivo JSON."""
    valores = {"vacas_en_produccion": 20, "gasolina_mensual": 400000, "precio_litro": 2200}
    if ARCHIVO_CONFIG.exists():
        try:
            with open(ARCHIVO_CONFIG, "r", encoding="utf-8") as archivo:
                valores.update(json.load(archivo))
        except (OSError, json.JSONDecodeError):
            pass
    return valores


def guardar_configuracion(valores):
    """Guarda la configuración de la finca."""
    DATA.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVO_CONFIG, "w", encoding="utf-8") as archivo:
        json.dump(valores, archivo, ensure_ascii=False, indent=2)


def cargar_vacas():
    """Carga el registro individual de animales con tipos compatibles con el editor."""
    columnas = ["codigo", "nombre", "estado", "fecha_ingreso", "observaciones"]
    if ARCHIVO_VACAS.exists():
        try:
            base = pd.read_csv(ARCHIVO_VACAS).reindex(columns=columnas)
        except (pd.errors.ParserError, OSError):
            base = pd.DataFrame(columns=columnas)
    else:
        base = pd.DataFrame(columns=columnas)

    # DateColumn requiere fechas reales, no cadenas leídas directamente del CSV.
    base["fecha_ingreso"] = pd.to_datetime(base["fecha_ingreso"], errors="coerce")
    for columna in ["codigo", "nombre", "estado", "observaciones"]:
        base[columna] = base[columna].astype("string")
    return base


def registrar_datos_rapidos():
    """Formularios sencillos para registrar la leche y los gastos sin editar tablas."""
    st.subheader("Agregar información de la finca")
    st.caption("Registra aquí la producción de leche y los gastos del día. Los datos se guardan en los archivos CSV de la aplicación.")

    tab_leche, tab_gastos = st.tabs(["🥛 Registrar leche", "💰 Registrar gasto"])

    with tab_leche:
        st.write("**Producción del día**")
        st.caption("Escribe por separado los litros del ordeño de la mañana y de la tarde. La aplicación guardará la suma como producción diaria.")
        with st.form("form_registro_leche", clear_on_submit=True):
            fecha_leche = st.date_input("Fecha", value=pd.Timestamp.today().date(), key="fecha_registro_leche")
            col_manana, col_tarde = st.columns(2)
            with col_manana:
                litros_manana = st.number_input("Litros de la mañana", min_value=0.0, max_value=10000.0, value=0.0, step=0.5)
            with col_tarde:
                litros_tarde = st.number_input("Litros de la tarde", min_value=0.0, max_value=10000.0, value=0.0, step=0.5)
            total_dia = litros_manana + litros_tarde
            st.metric("Total de leche del día", f"{total_dia:,.1f} litros")
            enviar_leche = st.form_submit_button("Guardar producción de leche", type="primary")

        if enviar_leche:
            if total_dia <= 0:
                st.error("Ingresa una cantidad mayor que cero para guardar la producción.")
            else:
                DATA.mkdir(parents=True, exist_ok=True)
                if ARCHIVO_PRODUCCION.exists():
                    base = pd.read_csv(ARCHIVO_PRODUCCION)
                else:
                    base = pd.DataFrame(columns=["fecha", "litros"])
                base = base.reindex(columns=["fecha", "litros"])
                base["fecha"] = pd.to_datetime(base["fecha"], errors="coerce")
                base["litros"] = pd.to_numeric(base["litros"], errors="coerce")
                base = base.dropna(subset=["fecha", "litros"])
                # Un registro por fecha: si ya existe, se reemplaza por el total actualizado del día.
                fecha_objetivo = pd.Timestamp(fecha_leche)
                base = base[base["fecha"].dt.date != fecha_leche]
                nuevo = pd.DataFrame({"fecha": [fecha_objetivo], "litros": [total_dia]})
                base = pd.concat([base, nuevo], ignore_index=True).sort_values("fecha")
                base["fecha"] = base["fecha"].dt.strftime("%Y-%m-%d")
                base.to_csv(ARCHIVO_PRODUCCION, index=False, encoding="utf-8-sig")
                st.cache_data.clear()
                st.session_state["mensaje_registro"] = f"Producción guardada: {total_dia:,.1f} litros para el {fecha_leche:%d/%m/%Y}."
                st.rerun()

    with tab_gastos:
        st.write("**Registrar una compra o gasto real**")
        with st.form("form_registro_gasto", clear_on_submit=True):
            fecha_gasto = st.date_input("Fecha del gasto", value=pd.Timestamp.today().date(), key="fecha_registro_gasto")
            concepto_opcion = st.selectbox("Tipo de gasto", ["Alimento", "Gasolina", "Medicamentos", "Otros"])
            concepto_otro = ""
            if concepto_opcion == "Otros":
                concepto_otro = st.text_input("¿Cuál gasto?")
            valor_gasto = st.number_input("Valor pagado (COP)", min_value=0, max_value=100000000, value=0, step=1000)
            enviar_gasto = st.form_submit_button("Guardar gasto", type="primary")

        if enviar_gasto:
            concepto_final = concepto_otro.strip() if concepto_opcion == "Otros" else concepto_opcion
            if not concepto_final:
                st.error("Escribe el nombre del gasto.")
            elif valor_gasto <= 0:
                st.error("Ingresa un valor mayor que cero.")
            else:
                DATA.mkdir(parents=True, exist_ok=True)
                if ARCHIVO_GASTOS.exists():
                    base = pd.read_csv(ARCHIVO_GASTOS)
                else:
                    base = pd.DataFrame(columns=["fecha", "concepto", "valor"])
                base = base.reindex(columns=["fecha", "concepto", "valor"])
                nuevo = pd.DataFrame({
                    "fecha": [pd.Timestamp(fecha_gasto).strftime("%Y-%m-%d")],
                    "concepto": [concepto_final],
                    "valor": [int(valor_gasto)],
                })
                pd.concat([base, nuevo], ignore_index=True).to_csv(
                    ARCHIVO_GASTOS, index=False, encoding="utf-8-sig"
                )
                st.cache_data.clear()
                st.session_state["mensaje_registro"] = f"Gasto guardado: {concepto_final} por {dinero(valor_gasto)}."
                st.rerun()


def administrar_datos():
    """Permite agregar, editar y eliminar registros y guardarlos en archivos."""
    st.subheader("Administrar información de la finca")
    st.caption("Edita las celdas directamente. Usa las filas nuevas para agregar y el icono de papelera para eliminar. Guarda cada tabla antes de salir.")

    seccion = st.radio("¿Qué deseas administrar?", ["Producción diaria", "Gastos", "Inventario de vacas", "Configuración"], horizontal=True)

    if seccion == "Producción diaria":
        st.write("**Registros de leche**")
        base = pd.read_csv(ARCHIVO_PRODUCCION) if ARCHIVO_PRODUCCION.exists() else pd.DataFrame({
            "fecha": pd.Series(dtype="datetime64[ns]"),
            "litros": pd.Series(dtype="float64"),
        })
        base = base.reindex(columns=["fecha", "litros"])
        # El CSV contiene fechas como texto; DateColumn necesita datetime.
        base["fecha"] = pd.to_datetime(base["fecha"], errors="coerce")
        base["litros"] = pd.to_numeric(base["litros"], errors="coerce").astype("float64")
        editado = st.data_editor(base, num_rows="dynamic", use_container_width=True, key="editor_produccion", column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "litros": st.column_config.NumberColumn("Litros", min_value=0.0, step=1.0)
        })
        if st.button("Guardar producción", type="primary"):
            validar = editado.copy()
            validar["fecha"] = pd.to_datetime(validar["fecha"], errors="coerce")
            validar["litros"] = pd.to_numeric(validar["litros"], errors="coerce")
            if validar[["fecha", "litros"]].isna().any().any() or (validar["litros"] < 0).any():
                st.error("Revisa las fechas y los litros: no pueden estar vacíos ni ser negativos.")
            else:
                guardar_csv(validar.assign(fecha=validar["fecha"].dt.strftime("%Y-%m-%d")), ARCHIVO_PRODUCCION, ["fecha", "litros"])
                st.success("Producción guardada.")
                st.rerun()

    elif seccion == "Gastos":
        st.write("**Compras y gastos reales**")
        base = pd.read_csv(ARCHIVO_GASTOS) if ARCHIVO_GASTOS.exists() else pd.DataFrame({
            "fecha": pd.Series(dtype="datetime64[ns]"),
            "concepto": pd.Series(dtype="string"),
            "valor": pd.Series(dtype="float64"),
        })
        base = base.reindex(columns=["fecha", "concepto", "valor"])
        # Convierte los tipos antes de pasarlos a st.data_editor.
        base["fecha"] = pd.to_datetime(base["fecha"], errors="coerce")
        base["concepto"] = base["concepto"].astype("string")
        base["valor"] = pd.to_numeric(base["valor"], errors="coerce").astype("float64")
        editado = st.data_editor(base, num_rows="dynamic", use_container_width=True, key="editor_gastos", column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            # Texto libre: acepta también conceptos adicionales que use la finca.
            "concepto": st.column_config.TextColumn("Concepto", help="Ej.: Alimento, Gasolina, Medicamentos u Otros"),
            "valor": st.column_config.NumberColumn("Valor (COP)", min_value=0, step=1000)
        })
        if st.button("Guardar gastos", type="primary"):
            validar = editado.copy()
            validar["fecha"] = pd.to_datetime(validar["fecha"], errors="coerce")
            validar["concepto"] = validar["concepto"].astype("string").str.strip()
            validar["valor"] = pd.to_numeric(validar["valor"], errors="coerce")
            # Ignora filas completamente vacías que pueden quedar al añadir una fila nueva.
            validar = validar.dropna(how="all")
            validar = validar[~(validar["fecha"].isna() & validar["concepto"].fillna("").eq("") & validar["valor"].isna())]
            if validar[["fecha", "concepto", "valor"]].isna().any().any() or validar["concepto"].fillna("").eq("").any() or (validar["valor"] < 0).any():
                st.error("Revisa fecha, concepto y valor. No pueden estar vacíos ni ser negativos.")
            else:
                guardar_csv(validar.assign(fecha=validar["fecha"].dt.strftime("%Y-%m-%d")), ARCHIVO_GASTOS, ["fecha", "concepto", "valor"])
                st.success("Gastos guardados.")
                st.rerun()

    elif seccion == "Inventario de vacas":
        st.write("**Registro individual (no es necesario completar datos que no conozcas)**")
        st.caption("Puedes agregar animales, cambiar su estado o eliminar registros. Los estados disponibles permiten distinguir vacas en producción, secas y animales jóvenes.")
        base = cargar_vacas()
        estados_base = ["Producción", "Seca", "Novilla", "Ternera"]
        estados_existentes = [str(v) for v in base["estado"].dropna().tolist() if str(v).strip()]
        estados = list(dict.fromkeys(estados_base + estados_existentes))
        base["estado"] = pd.Categorical(base["estado"].fillna(""), categories=estados + ([""] if "" not in estados else []))
        editado = st.data_editor(base, num_rows="dynamic", use_container_width=True, key="editor_vacas", column_config={
            "codigo": st.column_config.TextColumn("Código", help="Código interno, por ejemplo V01"),
            "nombre": st.column_config.TextColumn("Nombre"),
            "estado": st.column_config.SelectboxColumn("Estado", options=estados, required=True),
            "fecha_ingreso": st.column_config.DateColumn("Fecha de ingreso", format="DD/MM/YYYY"),
            "observaciones": st.column_config.TextColumn("Observaciones")
        })
        if st.button("Guardar inventario de vacas", type="primary"):
            validar = editado.copy()
            validar = validar.dropna(how="all")
            validar["codigo"] = validar["codigo"].astype("string").str.strip()
            validar["fecha_ingreso"] = pd.to_datetime(validar["fecha_ingreso"], errors="coerce")
            if validar["codigo"].fillna("").eq("").any():
                st.error("Cada animal debe tener un código único.")
            elif validar["codigo"].duplicated().any():
                st.error("Hay códigos repetidos. Cada vaca debe tener un código diferente.")
            else:
                validar["fecha_ingreso"] = validar["fecha_ingreso"].dt.strftime("%Y-%m-%d")
                validar.to_csv(ARCHIVO_VACAS, index=False, encoding="utf-8-sig")
                st.success("Inventario guardado.")
                st.rerun()

    else:
        st.write("**Parámetros generales**")
        config = cargar_configuracion()
        with st.form("form_config_finca"):
            vacas = st.number_input("Vacas actualmente en producción", min_value=0, max_value=10000, value=int(config["vacas_en_produccion"]), step=1)
            gasolina = st.number_input("Gasolina fija mensual (COP)", min_value=0, value=int(config["gasolina_mensual"]), step=10000)
            precio = st.number_input("Precio de venta por litro (COP)", min_value=0, value=int(config["precio_litro"]), step=50)
            if st.form_submit_button("Guardar configuración", type="primary"):
                guardar_configuracion({"vacas_en_produccion": int(vacas), "gasolina_mensual": int(gasolina), "precio_litro": int(precio)})
                st.success("Configuración guardada.")
                st.rerun()

    st.divider()
    st.download_button("Descargar copia de seguridad de producción", ARCHIVO_PRODUCCION.read_bytes() if ARCHIVO_PRODUCCION.exists() else b"", "produccion_diaria_backup.csv", disabled=not ARCHIVO_PRODUCCION.exists())
    st.download_button("Descargar copia de seguridad de gastos", ARCHIVO_GASTOS.read_bytes() if ARCHIVO_GASTOS.exists() else b"", "gastos_backup.csv", disabled=not ARCHIVO_GASTOS.exists())
    if ARCHIVO_VACAS.exists():
        st.download_button("Descargar inventario de vacas", ARCHIVO_VACAS.read_bytes(), "vacas_backup.csv")


# ------------------------------ Interfaz ------------------------------
def main():
    """Presenta el análisis interactivo de producción y finanzas."""
    st.title("🐄 Proyecto Lechero")
    st.caption("Exploración de datos y análisis económico de una finca lechera de Boyacá")
    mensaje_registro = st.session_state.pop("mensaje_registro", None)
    if mensaje_registro:
        st.success(mensaje_registro)
    configuracion = cargar_configuracion()
    vacas_en_produccion = int(configuracion["vacas_en_produccion"])

    if not ARCHIVO_PRODUCCION.exists():
        st.error(f"No se encontró el archivo: {ARCHIVO_PRODUCCION.name}")
        st.info("Ubica produccion_diaria.csv junto a app.py o dentro de la carpeta data.")
        st.stop()

    try:
        produccion = cargar_produccion(str(ARCHIVO_PRODUCCION), vacas_en_produccion)
        gastos = cargar_gastos(str(ARCHIVO_GASTOS))
    except (ValueError, pd.errors.ParserError) as error:
        st.error(f"No se pudieron cargar los datos: {error}")
        st.stop()

    if produccion.empty:
        st.warning("El archivo de producción no contiene registros válidos.")
        st.stop()

    # Barra lateral: filtros y parámetros transparentes.
    st.sidebar.header("Filtros de análisis")
    fecha_min = produccion["fecha"].min().date()
    fecha_max = produccion["fecha"].max().date()
    rango = st.sidebar.date_input(
        "Periodo de consulta", value=(fecha_min, fecha_max),
        min_value=fecha_min, max_value=fecha_max, format="DD/MM/YYYY"
    )
    if not isinstance(rango, (tuple, list)) or len(rango) != 2:
        st.info("Selecciona una fecha inicial y una fecha final.")
        st.stop()
    inicio, fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
    if inicio > fin:
        st.warning("La fecha inicial no puede ser posterior a la fecha final.")
        st.stop()

    st.sidebar.divider()
    st.sidebar.header("Parámetros del escenario")
    precio = st.sidebar.number_input(
        "Precio usado por litro (COP)", min_value=0, max_value=100000,
        value=int(configuracion["precio_litro"]), step=50,
        help="Se usa para estimar ingresos. Si el precio cambió durante el periodo, el resultado es aproximado."
    )
    gasolina_mensual = st.sidebar.number_input(
        "Gasolina fija mensual (COP)", min_value=0, max_value=10000000,
        value=int(configuracion["gasolina_mensual"]), step=10000,
        help="Ingresa el valor real mensual. Si hay gasolina registrada en el CSV para un mes, se prioriza ese registro."
    )
    ajuste_alimento = st.sidebar.slider(
        "Variación simulada del costo de alimento (%)", -50, 100, 0, step=5
    )
    mano_obra = st.sidebar.number_input(
        "Mano de obra simulada (COP/mes)", min_value=0, max_value=10000000,
        value=0, step=50000,
        help="Por defecto es cero porque el trabajo lo realiza la familia. Solo se incluye si deseas simular cuánto costaría contratar a alguien."
    )

    prod = produccion[(produccion["fecha"] >= inicio) & (produccion["fecha"] <= fin)].copy()
    gas = gastos[(gastos["fecha"] >= inicio) & (gastos["fecha"] <= fin)].copy()
    if prod.empty:
        st.warning("No hay producción registrada en el periodo seleccionado.")
        st.stop()

    prod["ingresos_estimados"] = prod["litros"] * precio
    prod["promedio_movil"] = prod["litros"].rolling(7, min_periods=1).mean()
    prod["atipico"] = marcar_atipicos(prod["litros"])
    mensual = resumen_mensual(
        prod, gas, precio, gasolina_mensual, ajuste_alimento,
        mano_obra, inicio, fin
    )

    # Resumen principal.
    total_litros = float(prod["litros"].sum())
    ingresos = float(mensual["ingresos_estimados"].sum())
    costos = float(mensual["costos_registrados"].sum())
    utilidad = ingresos - costos
    margen = (100 * utilidad / ingresos) if ingresos else 0.0

    st.subheader("Resumen del periodo")
    st.caption(
        f"{len(prod)} días con registro · {inicio:%d/%m/%Y} al {fin:%d/%m/%Y} · "
        "Los ingresos se estiman con el precio seleccionado."
    )
    metricas = st.columns(4)
    metricas[0].metric("Leche registrada", f"{total_litros:,.1f} L")
    metricas[1].metric("Ingresos estimados", dinero(ingresos))
    metricas[2].metric("Gastos registrados", dinero(costos))
    metricas[3].metric("Resultado estimado", dinero(utilidad), f"Margen: {margen:.1f}%")

    st.info(
        f"Inventario configurado: {vacas_en_produccion} vacas en producción. "
        "Criterio contable: el alimento se muestra como compra registrada (salida de dinero), "
        "no como consumo exacto del mes. Los medicamentos solo se cuentan cuando aparecen "
        "en gastos.csv. La mano de obra familiar no se descuenta como gasto real."
    )

    tab_resumen, tab_prod, tab_fin, tab_registrar, tab_datos, tab_admin = st.tabs(
        ["🏠 Resumen", "🥛 Producción", "💰 Finanzas", "➕ Registrar", "📋 Datos", "✏️ Administrar"]
    )

    with tab_resumen:
        izquierda, derecha = st.columns([1.35, 1])
        with izquierda:
            figura = px.line(
                prod, x="fecha", y="litros", markers=True,
                title="Producción diaria de leche",
                labels={"fecha": "Fecha", "litros": "Litros"}
            )
            figura.add_scatter(
                x=prod["fecha"], y=prod["promedio_movil"],
                mode="lines", name="Promedio móvil (7 días)"
            )
            puntos = prod[prod["atipico"]]
            if not puntos.empty:
                figura.add_scatter(
                    x=puntos["fecha"], y=puntos["litros"], mode="markers",
                    name="Valor atípico", marker={"size": 10, "symbol": "diamond"}
                )
            figura.update_layout(legend_title_text="", hovermode="x unified")
            st.plotly_chart(figura, use_container_width=True)
        with derecha:
            por_mes = mensual[["mes", "litros"]]
            st.plotly_chart(
                px.bar(por_mes, x="mes", y="litros", text_auto=".0f",
                       title="Producción acumulada por mes",
                       labels={"mes": "Mes", "litros": "Litros"}),
                use_container_width=True
            )
        st.subheader("Lectura rápida de los datos")
        stats = estadisticas(prod["litros"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Promedio diario", f"{stats['Promedio diario']:,.1f} L")
        c2.metric("Día de mayor producción", f"{stats['Máximo']:,.1f} L")
        c3.metric("Día de menor producción", f"{stats['Mínimo']:,.1f} L")

    with tab_prod:
        st.subheader("Comportamiento de la producción")
        stats = estadisticas(prod["litros"])
        columnas = st.columns(5)
        indicadores = [
            ("Promedio", f"{stats['Promedio diario']:,.1f} L"),
            ("Mediana", f"{stats['Mediana']:,.1f} L"),
            ("Desviación", f"{stats['Desviación estándar']:,.1f} L"),
            ("Mínimo", f"{stats['Mínimo']:,.1f} L"),
            ("Máximo", f"{stats['Máximo']:,.1f} L"),
        ]
        for columna, (nombre, valor) in zip(columnas, indicadores):
            columna.metric(nombre, valor)

        izquierda, derecha = st.columns(2)
        por_dia = (prod.groupby("dia_semana")["litros"].mean()
                   .reindex(DIAS).dropna().reset_index())
        izquierda.plotly_chart(
            px.bar(por_dia, x="dia_semana", y="litros",
                   title="Promedio de producción por día de la semana",
                   labels={"dia_semana": "Día", "litros": "Promedio de litros"}),
            use_container_width=True
        )
        derecha.plotly_chart(
            px.histogram(prod, x="litros", nbins=10,
                         title="Distribución de la producción diaria",
                         labels={"litros": "Litros por día", "count": "Frecuencia"}),
            use_container_width=True
        )
        st.caption(
            f"Coeficiente de variación: "
            f"{(100 * stats['Desviación estándar'] / stats['Promedio diario']) if stats['Promedio diario'] else 0:.1f}%. "
            "Un valor mayor indica más variación relativa entre los días registrados."
        )

    with tab_fin:
        st.subheader("Ingresos, compras y resultado económico")
        st.caption("Los ingresos son estimados con el precio seleccionado; los gastos provienen del CSV y del valor fijo de gasolina configurado.")
        izquierda, derecha = st.columns(2)
        izquierda.plotly_chart(
            px.bar(mensual, x="mes", y=["ingresos_estimados", "costos_registrados"],
                   barmode="group", title="Ingresos estimados frente a gastos registrados",
                   labels={"value": "COP", "mes": "Mes", "variable": "Concepto"}),
            use_container_width=True
        )
        componentes = {
            "Alimento comprado": float(mensual["alimento_comprado"].sum()),
            "Gasolina": float(mensual["gasolina"].sum()),
            "Medicamentos": float(mensual["medicamentos"].sum()),
            "Otros gastos": float(mensual["otros_gastos"].sum()),
        }
        componentes = {nombre: valor for nombre, valor in componentes.items() if valor > 0}
        if componentes:
            derecha.plotly_chart(
                px.pie(names=list(componentes), values=list(componentes.values()),
                       title="Composición de los gastos registrados", hole=0.35),
                use_container_width=True
            )
        else:
            derecha.info("Todavía no hay gastos registrados en el periodo seleccionado.")

        st.subheader("Simulador: ¿qué pasaría si…?")
        escenario_costos = float(mensual["costos_escenario"].sum())
        escenario_utilidad = ingresos - escenario_costos
        sim1, sim2, sim3 = st.columns(3)
        sim1.metric("Costo del escenario", dinero(escenario_costos))
        sim2.metric("Resultado del escenario", dinero(escenario_utilidad),
                    f"Cambio: {dinero(escenario_utilidad - utilidad)}")
        sim3.metric("Mano de obra simulada", dinero(float(mensual["mano_obra_simulada"].sum())))
        st.caption(
            f"El escenario aplica un cambio de {ajuste_alimento}% sobre las compras de alimento "
            "y agrega mano de obra solo si se configura. No modifica los gastos históricos."
        )
        precios = np.arange(max(0, int(precio) - 500), int(precio) + 501, 50)
        if len(precios) > 1:
            sensibilidad = pd.DataFrame({
                "precio_litro": precios,
                "resultado": total_litros * precios - costos,
            })
            grafico = px.line(
                sensibilidad, x="precio_litro", y="resultado", markers=True,
                title="Sensibilidad del resultado frente al precio de la leche",
                labels={"precio_litro": "Precio por litro (COP)", "resultado": "Resultado (COP)"}
            )
            grafico.add_vline(x=precio, line_dash="dash", annotation_text="Precio seleccionado")
            st.plotly_chart(grafico, use_container_width=True)

        with st.expander("Ver detalle financiero mensual"):
            columnas_tabla = [
                "mes", "litros", "ingresos_estimados", "alimento_comprado",
                "gasolina", "medicamentos", "otros_gastos", "costos_registrados",
                "utilidad_registrada", "costos_escenario", "utilidad_escenario"
            ]
            st.dataframe(mensual[columnas_tabla].round(0), use_container_width=True, hide_index=True)

    with tab_registrar:
        registrar_datos_rapidos()

    with tab_datos:
        st.subheader("Registros utilizados")
        st.write("Producción diaria")
        st.dataframe(prod[["fecha", "litros", "dia_semana"]], use_container_width=True, hide_index=True)
        st.write("Gastos registrados")
        if gas.empty:
            st.info("No hay gastos registrados en el periodo seleccionado.")
        else:
            st.dataframe(gas[["fecha", "concepto", "valor"]], use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar producción filtrada (CSV)",
            prod[["fecha", "litros", "dia_semana"]].to_csv(index=False).encode("utf-8-sig"),
            file_name="produccion_filtrada.csv", mime="text/csv"
        )
        if not gas.empty:
            st.download_button(
                "Descargar gastos filtrados (CSV)",
                gas[["fecha", "concepto", "valor"]].to_csv(index=False).encode("utf-8-sig"),
                file_name="gastos_filtrados.csv", mime="text/csv"
            )

    with tab_admin:
        administrar_datos()

    with st.expander("Acerca de los datos y sus limitaciones"):
        st.write(
            "Este proyecto utiliza registros propios de la finca. La cantidad de días "
            "analizados corresponde únicamente a los registros recuperados; no se generan "
            "filas artificiales para completar un tamaño de muestra. Los resultados describen "
            "el periodo disponible y no deben generalizarse a fechas sin información."
        )
        st.write(
            "El costo de alimento representa compras registradas, no necesariamente el alimento "
            "consumido en el mismo mes. Para calcular consumo sería necesario registrar inventario "
            "inicial, entradas, salidas y existencia final."
        )


if __name__ == "__main__":
    main()
