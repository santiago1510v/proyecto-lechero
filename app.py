"""Dashboard de gestión financiera de una finca lechera de Boyacá.

Lee los registros diarios de entrega de leche y los gastos de la finca y
muestra producción, ingresos, costos y utilidad, con un simulador de
escenarios para ayudar a tomar decisiones.
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
# Costos mensuales que estima el productor (COP). Si en gastos.csv hay un
# registro real con ese concepto para un mes, se usa el real.
COSTOS_MENSUALES_ESTIMADOS = {"Gasolina": 400000, "Medicamentos": 300000}
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado",
        "Domingo"]

st.set_page_config(page_title="Finanzas lecheras", page_icon="🐄",
                   layout="wide")


@st.cache_data
def cargar_produccion():
    """Lee el CSV de producción diaria y agrega columnas calculadas."""
    df = pd.read_csv(DATA / "produccion_diaria.csv", parse_dates=["fecha"])
    df = df.drop_duplicates(subset="fecha").dropna(subset=["litros"])
    df = df.sort_values("fecha").reset_index(drop=True)
    df["mes"] = df["fecha"].dt.strftime("%Y-%m")
    df["dia_semana"] = df["fecha"].dt.dayofweek.map(lambda i: DIAS[i])
    df["litros_por_vaca"] = (df["litros"] / VACAS).round(1)
    return df


@st.cache_data
def cargar_gastos():
    """Lee el CSV de gastos (fecha, concepto, valor)."""
    df = pd.read_csv(DATA / "gastos.csv", parse_dates=["fecha"])
    df["mes"] = df["fecha"].dt.strftime("%Y-%m")
    return df


def dinero(valor):
    """Da formato de pesos a un número."""
    return f"${valor:,.0f}"


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


def resumen_mensual(prod, gastos, precio, ajuste_alimento=0, mano_obra=0):
    """Calcula ingresos, costos y utilidad por mes.

    Los costos estimados (gasolina, medicamentos) y la mano de obra
    simulada se reparten según los días con registro de cada mes.
    """
    mensual = prod.groupby("mes").agg(
        dias=("litros", "count"), litros=("litros", "sum")).reset_index()
    dias_mes = [pd.Period(mes).days_in_month for mes in mensual["mes"]]
    fraccion = mensual["dias"] / dias_mes

    alimento = gastos[gastos["concepto"] == "Alimento"].groupby("mes")["valor"]
    mensual["ingresos"] = mensual["litros"] * precio
    mensual["alimento"] = (mensual["mes"].map(alimento.sum()).fillna(0)
                           * (1 + ajuste_alimento / 100))

    otros = 0
    for concepto, valor_mensual in COSTOS_MENSUALES_ESTIMADOS.items():
        real = gastos[gastos["concepto"] == concepto].groupby("mes")["valor"]
        estimado = valor_mensual * fraccion
        otros = otros + mensual["mes"].map(real.sum()).fillna(estimado)

    conocidos = ["Alimento"] + list(COSTOS_MENSUALES_ESTIMADOS)
    extra = gastos[~gastos["concepto"].isin(conocidos)].groupby("mes")["valor"]
    mensual["otros"] = otros + mensual["mes"].map(extra.sum()).fillna(0)
    mensual["mano_obra"] = mano_obra * fraccion
    mensual["costos"] = (mensual["alimento"] + mensual["otros"]
                         + mensual["mano_obra"])
    mensual["utilidad"] = mensual["ingresos"] - mensual["costos"]
    return mensual


def tarjetas_financieras(total):
    """Arma la lista de indicadores financieros del periodo (nombre, valor)."""
    ingresos = total["ingresos"]
    margen = 100 * total["utilidad"] / ingresos if ingresos else 0.0
    peso_alimento = 100 * total["alimento"] / ingresos if ingresos else 0.0
    costo_litro = total["costos"] / total["litros"]
    return [
        ("Ingresos", dinero(ingresos)),
        ("Costos", dinero(total["costos"])),
        ("Utilidad", dinero(total["utilidad"])),
        ("Margen de utilidad", f"{margen:.1f} %"),
        ("Costo por litro", dinero(costo_litro)),
        ("Alimento sobre ingresos", f"{peso_alimento:.1f} %"),
        ("Utilidad por vaca", dinero(total["utilidad"] / VACAS)),
    ]


def main():
    """Construye la página: filtros, escenario, producción y finanzas."""
    prod = cargar_produccion()
    gastos = cargar_gastos()

    st.title("🐄 Gestión financiera de una finca lechera en Boyacá")
    st.caption("Datos reales de entrega de leche y gastos de la finca. "
               "Gasolina y medicamentos son estimaciones del productor.")

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

    st.sidebar.header("Escenario financiero")
    precio = st.sidebar.number_input("Precio del litro (COP)", min_value=1000,
                                     max_value=4000, value=PRECIO_LITRO,
                                     step=50)
    ajuste = st.sidebar.slider("Cambio en el costo del alimento (%)",
                               -30, 50, 0)
    mano_obra = st.sidebar.number_input("Mano de obra si se pagara (COP/mes)",
                                        min_value=0, value=0, step=100000)

    # ---------- Filtrado por fechas ----------
    inicio, fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1])
    dff = prod[(prod["fecha"] >= inicio) & (prod["fecha"] <= fin)].copy()
    gff = gastos[(gastos["fecha"] >= inicio) & (gastos["fecha"] <= fin)]
    if dff.empty:
        st.warning("No hay datos en ese rango.")
        st.stop()
    dff["ingresos"] = dff["litros"] * precio

    tab_prod, tab_fin = st.tabs(["📈 Producción", "💰 Finanzas"])

    # ---------- Pestaña de producción ----------
    with tab_prod:
        stats = estadisticas(dff["litros"])
        cajas = st.columns(len(stats))
        for caja, (etiqueta, valor) in zip(cajas, stats.items()):
            caja.metric(etiqueta, f"{valor:,.1f}")
        st.caption(f"{len(dff)} días con registro, entre "
                   f"{dff['fecha'].min():%d/%m/%Y} y "
                   f"{dff['fecha'].max():%d/%m/%Y}.")

        dff["promedio_movil"] = dff[columna].rolling(
            ventana, min_periods=1).mean()
        dff["atipico"] = marcar_atipicos(dff["litros"])
        fig1 = px.line(dff, x="fecha", y=[columna, "promedio_movil"],
                       title=f"{nombre} por día y promedio móvil de "
                             f"{ventana} días",
                       labels={"value": nombre, "variable": "",
                               "fecha": "Fecha"})
        atipicos = dff[dff["atipico"]]
        if ver_atipicos and not atipicos.empty:
            fig1.add_scatter(x=atipicos["fecha"], y=atipicos[columna],
                             mode="markers", name="Día atípico",
                             marker={"size": 11, "color": "red"})
        st.plotly_chart(fig1)

        g1, g2 = st.columns(2)
        por_dia = (dff.groupby("dia_semana")["litros"].mean()
                      .reindex(DIAS).dropna().round(1).reset_index())
        g1.plotly_chart(px.bar(
            por_dia, x="dia_semana", y="litros",
            title="Promedio de litros por día de la semana",
            labels={"dia_semana": "Día", "litros": "Litros"}))
        g2.plotly_chart(px.histogram(
            dff, x="litros", nbins=10,
            title="Distribución de los litros diarios",
            labels={"litros": "Litros por día"}))

    # ---------- Pestaña de finanzas ----------
    mensual = resumen_mensual(dff, gff, precio, ajuste, mano_obra)
    total = mensual[["litros", "ingresos", "alimento", "otros", "mano_obra",
                     "costos", "utilidad"]].sum()
    with tab_fin:
        tarjetas = tarjetas_financieras(total)
        for fila in (tarjetas[:4], tarjetas[4:]):
            for caja, (etiqueta, valor) in zip(st.columns(len(fila)), fila):
                caja.metric(etiqueta, valor)
        st.caption("Costos = alimento registrado + gasolina y medicamentos "
                   "estimados (repartidos por días) + mano de obra solo si "
                   "la simulas en la barra lateral.")

        f1, f2 = st.columns(2)
        f1.plotly_chart(px.bar(
            mensual, x="mes", y=["ingresos", "costos"], barmode="group",
            title="Ingresos y costos por mes (COP)",
            labels={"value": "Pesos", "variable": "", "mes": "Mes"}))
        partes = {"Alimento": total["alimento"],
                  "Gasolina y medicamentos": total["otros"],
                  "Mano de obra": total["mano_obra"]}
        partes = {k: v for k, v in partes.items() if v > 0}
        f2.plotly_chart(px.pie(names=list(partes), values=list(partes.values()),
                               title="¿En qué se van los costos?"))

        precios = np.arange(1800, 2701, 100)
        utilidad_precio = total["litros"] * precios - total["costos"]
        sensibilidad = pd.DataFrame({"precio": precios,
                                     "utilidad": utilidad_precio})
        fig_s = px.line(sensibilidad, x="precio", y="utilidad", markers=True,
                        title="Utilidad del periodo según el precio del litro",
                        labels={"precio": "Precio del litro (COP)",
                                "utilidad": "Utilidad (COP)"})
        fig_s.add_vline(x=precio, line_dash="dash")
        st.plotly_chart(fig_s)

    # ---------- Tablas ----------
    with st.expander("📋 Resumen financiero por mes"):
        st.caption("Los meses incompletos tienen menos días. Las compras de "
                   "alimento son por tandas, así que un mes puede salir "
                   "mejor o peor según cuándo se compró.")
        st.dataframe(mensual.round(0), hide_index=True)
    with st.expander("📋 Producción diaria"):
        st.dataframe(dff[["fecha", "litros", "dia_semana"]], hide_index=True)
        st.download_button("Descargar CSV",
                           dff.to_csv(index=False).encode("utf-8"),
                           "produccion_filtrada.csv", "text/csv")


main()
