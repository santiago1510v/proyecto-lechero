"""Dashboard de producción lechera.

Lee los registros diarios de entrega de leche y las compras de alimento
de una finca de Boyacá y los presenta de forma interactiva.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" if (BASE / "data").exists() else BASE
VACAS = 20
PRECIO_LITRO = 2200
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

st.set_page_config(page_title="Producción lechera", page_icon="🐄", layout="wide")


@st.cache_data
def cargar_produccion():
    """Lee el CSV de producción diaria y agrega columnas calculadas."""
    df = pd.read_csv(DATA / "produccion_diaria.csv", parse_dates=["fecha"])
    df = df.drop_duplicates(subset="fecha").dropna(subset=["litros"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["mes"] = df["fecha"].dt.strftime("%Y-%m")
    df["dia_semana"] = df["fecha"].dt.dayofweek.map(lambda i: DIAS[i])
    df["ingresos"] = df["litros"] * PRECIO_LITRO
    df["litros_por_vaca"] = (df["litros"] / VACAS).round(1)
    return df


@st.cache_data
def cargar_gastos():
    """Lee el CSV de gastos (compras de alimento)."""
    df = pd.read_csv(DATA / "gastos.csv", parse_dates=["fecha"])
    df["mes"] = df["fecha"].dt.strftime("%Y-%m")
    return df


def estadisticas(litros):
    """Calcula estadísticas descriptivas de los litros con NumPy."""
    valores = np.asarray(litros, dtype=float)
    media = np.mean(valores)
    desviacion = np.std(valores, ddof=1) if len(valores) > 1 else 0.0
    return {
        "Promedio (L/día)": media,
        "Mediana (L/día)": np.median(valores),
        "Desviación estándar": desviacion,
        "Mínimo (L)": np.min(valores),
        "Máximo (L)": np.max(valores),
        "Variación (%)": 100 * desviacion / media if media else 0.0,
    }


def marcar_atipicos(litros):
    """Devuelve True en los días atípicos (z-score robusto: mediana y MAD)."""
    valores = np.asarray(litros, dtype=float)
    mediana = np.median(valores)
    mad = np.median(np.abs(valores - mediana))
    if mad == 0:
        return np.zeros(len(valores), dtype=bool)
    z = 0.6745 * (valores - mediana) / mad
    return np.abs(z) > 3.5


def resumen_mensual(prod, gastos):
    """Une producción y gastos por mes y calcula ingresos, costos y utilidad."""
    mensual = prod.groupby("mes").agg(
        dias=("litros", "count"),
        litros=("litros", "sum"),
        promedio_dia=("litros", "mean"),
    ).reset_index()
    costos = gastos.groupby("mes")["valor"].sum()
    mensual["ingresos"] = mensual["litros"] * PRECIO_LITRO
    mensual["costos"] = mensual["mes"].map(costos).fillna(0)
    mensual["utilidad"] = mensual["ingresos"] - mensual["costos"]
    return mensual


def main():
    """Construye la página: filtros, estadísticas, gráficas y tablas."""
    prod = cargar_produccion()
    gastos = cargar_gastos()

    st.title("🐄 Producción lechera de una finca en Boyacá")
    st.caption("Datos reales de entrega de leche y compras de alimento.")

    # ---------- Widgets (barra lateral) ----------
    st.sidebar.header("Filtros")
    desde = prod["fecha"].min().date()
    hasta = prod["fecha"].max().date()
    rango = st.sidebar.date_input("Rango de fechas", value=(desde, hasta),
                                  min_value=desde, max_value=hasta)
    if len(rango) != 2:
        st.info("Selecciona la fecha inicial y la final.")
        st.stop()

    opciones = {"Litros": "litros", "Ingresos (COP)": "ingresos",
                "Litros por vaca": "litros_por_vaca"}
    nombre = st.sidebar.selectbox("Variable a graficar", list(opciones))
    columna = opciones[nombre]
    ventana = st.sidebar.slider("Días del promedio móvil", 3, 14, 7)
    ver_atipicos = st.sidebar.checkbox("Marcar días atípicos", value=True)

    # ---------- Filtrado por fechas ----------
    inicio, fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
    dff = prod[(prod["fecha"] >= inicio) & (prod["fecha"] <= fin)].copy()
    gff = gastos[(gastos["fecha"] >= inicio) & (gastos["fecha"] <= fin)]
    if dff.empty:
        st.warning("No hay datos en ese rango.")
        st.stop()

    # ---------- Estadísticas con NumPy ----------
    stats = estadisticas(dff["litros"])
    cajas = st.columns(len(stats))
    for caja, (etiqueta, valor) in zip(cajas, stats.items()):
        caja.metric(etiqueta, f"{valor:,.1f}")
    st.caption(f"{len(dff)} días con registro, entre {dff['fecha'].min():%d/%m/%Y} "
               f"y {dff['fecha'].max():%d/%m/%Y}.")

    # ---------- Gráficas ----------
    dff["promedio_movil"] = dff[columna].rolling(ventana, min_periods=1).mean()
    dff["atipico"] = marcar_atipicos(dff["litros"])

    fig1 = px.line(dff, x="fecha", y=[columna, "promedio_movil"],
                   title=f"{nombre} por día y promedio móvil de {ventana} días",
                   labels={"value": nombre, "variable": "", "fecha": "Fecha"})
    atipicos = dff[dff["atipico"]]
    if ver_atipicos and not atipicos.empty:
        fig1.add_scatter(x=atipicos["fecha"], y=atipicos[columna], mode="markers",
                         name="Día atípico", marker={"size": 11, "color": "red"})
    st.plotly_chart(fig1)

    g1, g2 = st.columns(2)
    por_dia = (dff.groupby("dia_semana")["litros"].mean()
                  .reindex(DIAS).dropna().round(1).reset_index())
    g1.plotly_chart(px.bar(por_dia, x="dia_semana", y="litros",
                           title="Promedio de litros por día de la semana",
                           labels={"dia_semana": "Día", "litros": "Litros"}))
    g2.plotly_chart(px.histogram(dff, x="litros", nbins=10,
                                 title="Distribución de los litros diarios",
                                 labels={"litros": "Litros por día"}))

    mensual = resumen_mensual(dff, gff)
    st.plotly_chart(px.bar(mensual, x="mes", y=["ingresos", "costos"],
                           barmode="group", title="Ingresos y costos por mes (COP)",
                           labels={"value": "Pesos", "variable": "", "mes": "Mes"}))

    # ---------- Tablas ----------
    with st.expander("📋 Resumen por mes"):
        st.caption("Los meses incompletos tienen menos días; compáralos con el promedio por día. "
                   "Los costos son solo las compras de alimento registradas.")
        st.dataframe(mensual, hide_index=True)
    with st.expander("📋 Producción diaria"):
        st.dataframe(dff[["fecha", "litros", "dia_semana"]], hide_index=True)
        st.download_button("Descargar CSV", dff.to_csv(index=False).encode("utf-8"),
                           "produccion_filtrada.csv", "text/csv")


main()
