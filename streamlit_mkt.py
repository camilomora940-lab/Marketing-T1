import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
from PIL import Image
import plotly.graph_objects as go
# Configuración profesional para la UdeC
st.set_page_config(page_title="Marketing - Estimación de Demanda", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('df_final_companias_agrupadas.csv')
    # Limpieza rápida para el slider
    df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int)
    # Asegurarnos de tener la columna de compañía principal si no existe con ese nombre exacto
    # Aquí asumimos que usas 'production_companies' o una procesada
    return df

df = load_data()
logo=Image.open("logo.png")
# --- SIDEBAR: CONTROLES DE LA PRESENTACIÓN ---
st.sidebar.image(logo, use_container_width=True) # Opcional: logo DII
st.sidebar.header("Panel de Control")

# 1. Filtro de Tiempo
years = st.sidebar.slider("Periodo de Análisis", 1920, 2016, (2000, 2016))

# 2. Filtro de Géneros
generos = ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama', 'Horror', 'Romance', 'Science_Fiction', 'Thriller']
sel_generos = st.sidebar.multiselect("Géneros en Pantalla", generos, default=generos)

# 3. Filtro de Presupuesto
max_b = st.sidebar.number_input("Presupuesto Máx (USD)", value=int(df['budget'].max()), step=10000000)

# Aplicar Filtros
mask = (df['release_year'].between(years[0], years[1])) & (df['budget'] <= max_b)
if sel_generos:
    # Filtra filas donde al menos uno de los géneros seleccionados sea 1
    mask = mask & (df[sel_generos].sum(axis=1) > 0)

df_filtered = df[mask].copy()

# --- CUERPO PRINCIPAL ---
st.title("Trabajo 1 - Estimación de Demanda en la Industria Cinematográfica")
st.markdown(f"**Muestra actual:** {df_filtered.shape[0]} películas filtradas.")

tabs = st.tabs(["Visualización de Datos", "Modelo Econométrico", "Análisis de ROI"])

with tabs[0]:
    st.header("Análisis de Revenue y Distribuciones")
    
    # Replicando la lógica de tu celda de "Top Revenue"
    top_10 = df_filtered.sort_values('revenue', ascending=False).head(10)
    
    fig_top = px.scatter(
        df_filtered, x="budget", y="revenue", 
        color_discrete_sequence=['gray'], opacity=0.3,
        title="Películas con Mayor Revenue vs Presupuesto (Escala Normal)"
    )
    # Resaltar el top 10
    fig_top = px.scatter(df_filtered, x="budget", y="revenue", opacity=0.3,
                         hover_name="title", title="Top 10 Revenue vs Resto del Dataset")
    fig_top.update_traces(marker=dict(color='gray')) # Todo a gris primero
    # Añadir los colores del Top 10 individualmente
    for i, row in top_10.iterrows():
        fig_top.add_trace(go.Scatter(x=[row['budget']], y=[row['revenue']], 
                                     mode='markers', marker=dict(size=12),
                                     name=row['title']))
    st.plotly_chart(fig_top, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribución de Presupuesto")
        fig_hist_b = px.histogram(df_filtered, x="budget", nbins=30, color_discrete_sequence=['#003366'])
        st.plotly_chart(fig_hist_b, use_container_width=True)
    
    with col2:
        st.subheader("Distribución de Ingresos")
        fig_hist_r = px.histogram(df_filtered, x="revenue", nbins=30, color_discrete_sequence=['#E5A823'])
        st.plotly_chart(fig_hist_r, use_container_width=True)

# --- TAB 2: TRANSFORMACIÓN LOGARÍTMICA ---
with tabs[1]:
    st.header("Justificación de Transformación Logarítmica")
    st.write("Para corregir el sesgo y cumplir los supuestos de la regresión lineal, transformamos las variables.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Antes (Raw Data)")
        fig_raw = px.scatter(df_filtered, x="budget", y="revenue", trendline="ols")
        st.plotly_chart(fig_raw, use_container_width=True)
        
    with c2:
        st.subheader("Después (Log-Log)")
        fig_log = px.scatter(df_filtered, x="log_budget", y="log_revenue", trendline="ols",
                             color_discrete_sequence=['#003366'])
        st.plotly_chart(fig_log, use_container_width=True)
    
    with c2:
        st.subheader("Después (Log-Log)")
        fig_log = px.scatter(df_filtered, x="log_budget", y="log_revenue", trendline="ols",
                             color_discrete_sequence=['#003366'])
        st.plotly_chart(fig_log, use_container_width=True)

with tabs[1]:
    st.header("Resultados de la Regresión (Modelo 3)")
    
    # Definición de la fórmula según tu modelo 3
    # Nota: Ajusta 'compania_principal_agrupadas2' al nombre real en tu CSV
    formula3 = 'log_revenue ~ log_budget + log_vote_count + vote_average + is_english + C(release_month)'
    
    try:
        modelo3 = smf.ols(formula=formula3, data=df_filtered).fit()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("R-Squared", f"{modelo3.rsquared:.3f}")
            st.write("**Coeficientes Críticos:**")
            st.write(f"Budget: `{modelo3.params['log_budget']:.4f}`")
            st.write(f"Votos: `{modelo3.params['log_vote_count']:.4f}`")
            
            # Alerta de significancia
            p_val = modelo3.pvalues['log_budget']
            if p_val < 0.05:
                st.success(f"Presupuesto es significativo (p={p_val:.3f})")
            else:
                st.warning("Presupuesto no es significativo con estos filtros.")

        with c2:
            st.text("Resumen de Coeficientes (Statsmodels)")
            st.write(modelo3.summary().tables[1])
            
    except Exception as e:
        st.error(f"Error al correr el modelo: Selecciona más datos. (Detalle: {e})")

with tabs[2]:
    st.header("Cálculo de ROI y Elasticidad Marginal")
    
    if 'modelo3' in locals():
        e_budget = modelo3.params['log_budget']
        e_votos = modelo3.params['log_vote_count']
        
        st.info(f"""
        ### Interpretación de Elasticidades:
        * **Elasticidad Presupuesto ({e_budget:.2f}):** Por cada 1% de aumento en presupuesto, el ingreso sube un **{e_budget:.2f}%**.
        * **Elasticidad Votos ({e_votos:.2f}):** Por cada 1% de aumento en votos (engagement), el ingreso sube un **{e_votos:.2f}%**.
        """)
        
        # Simulador de ROI
        st.subheader("Simulador de ROI Marginal")
        inversion_extra = st.slider("Aumento de Presupuesto (%)", 0, 100, 10)
        
        revenue_promedio = df_filtered['revenue'].mean()
        budget_promedio = df_filtered['budget'].mean()
        
        gasto_extra = budget_promedio * (inversion_extra / 100)
        ingreso_extra = revenue_promedio * ((inversion_extra * e_budget) / 100)
        roi_marginal = (ingreso_extra / gasto_extra) if gasto_extra > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Gasto Extra Est.", f"${gasto_extra/1e6:.2f}M")
        c2.metric("Ingreso Extra Est.", f"${ingreso_extra/1e6:.2f}M")
        c3.metric("ROI del Aumento", f"{roi_marginal:.2f}x")
        
        st.write(f"**Conclusión:** Por cada $1 adicional invertido en producción, se recuperan **${roi_marginal:.2f}**.")

st.markdown("---")
st.caption("Trabajo 1 - Marketing 2026 | Universidad de Concepción")