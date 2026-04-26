from io import StringIO

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
generos = ['Action', 'Adventure', 'Animation', 'Comedy', 'Drama', 'Horror', 'Romance', 'Science_Fiction', 'Thriller','Fantasy', 'Mystery', 'Documentary', 'Family', 'War', 'Music', 'History', 'Western', 'TV_Movie', 'Foreign','Crime','Otro']
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

tabs = st.tabs(["Visualización de Datos", "Modelo Regresión lineal", "Análisis de ROI"])

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
                         hover_name="title", title="Top 10 Recaudación vs Resto del Dataset")
    fig_top.update_traces(marker=dict(color='gray')) # Todo a gris primero
    # Añadir los colores del Top 10 individualmente
    for i, row in top_10.iterrows():
        fig_top.add_trace(go.Scatter(x=[row['budget']], y=[row['revenue']], 
                                     mode='markers', marker=dict(size=12),
                                     name=row['title']))
    st.plotly_chart(fig_top, use_container_width=True)
    st.header("Análisis de Rentabilidad por Género")
    generos_lista = [
    'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
    'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 
    'Horror', 'Music', 'Mystery', 'Romance', 'Science_Fiction', 
    'Thriller', 'War', 'Western', 'Foreign', 'TV_Movie', 'Otro']
    promedios = []
    for g in generos_lista:
        if g in df_filtered.columns:
            sub_df = df_filtered[df_filtered[g] == 1]
            if not sub_df.empty:
                mean_rev = sub_df['revenue'].mean()
                count = len(sub_df)
                promedios.append({'Genero': g, 'Revenue_Promedio': mean_rev, 'Cantidad': count})
    df_promedios = pd.DataFrame(promedios).sort_values('Revenue_Promedio', ascending=True)
    fig_generos = px.bar(
    df_promedios, 
    x='Revenue_Promedio', 
    y='Genero',
    orientation='h',
    title="<b>Revenue Promedio por Género (Incluyendo Nichos)</b>",
    labels={'Revenue_Promedio': 'Revenue Promedio (USD)', 'Genero': 'Género'},
    color='Revenue_Promedio',
    color_continuous_scale='Blues',
    text_auto='.2s',
    hover_data={'Cantidad': True, 'Revenue_Promedio': ':$,.0f'}
)
    fig_generos.update_layout(
    height=900, # Aumentamos un poco más el alto para las 3 nuevas filas
    margin=dict(l=150, r=50), 
    xaxis_title="Revenue Promedio (USD)",
    yaxis_title="",
    coloraxis_showscale=False,
    # Si la diferencia entre Adventure y TV_Movie es demasiada, 
    # puedes descomentar la siguiente línea para usar escala logarítmica:
    # log_x=True 
)
    fig_generos.update_traces(textposition='outside', cliponaxis=False)
    st.plotly_chart(fig_generos, use_container_width=True, key="grafico_generos_completo")
    #ahora otro gráfico con lo mismo pero logaritmico para ver mejor las diferencias entre los géneros con menor revenue
    fig_generos_log = px.bar(df_promedios, x='Revenue_Promedio', y='Genero', orientation='h',
                             title="<b>Revenue Promedio por Género (Escala Logarítmica)</b>",
                             labels={'Revenue_Promedio': 'Revenue Promedio (USD)', 'Genero': 'Género'},
                             color='Revenue_Promedio', color_continuous_scale='Blues',
                                text_auto='.2s', log_x=True, hover_data={'Cantidad': True, 'Revenue_Promedio': ':$,.0f'})
    fig_generos_log.update_layout(
        height=900,
        margin=dict(l=150, r=50),
        xaxis_title="Revenue Promedio (USD, Escala Logarítmica)",
        yaxis_title="",
        coloraxis_showscale=False
    )
    fig_generos_log.update_traces(textposition='outside', cliponaxis=False)
    st.plotly_chart(fig_generos_log, use_container_width=True, key="grafico_generos_log")

    st.header("Análisis de las Top 10 Productoras")
    col_prod = 'compania_principal_agrupadas2'
    if col_prod in df_filtered.columns:
        df_top_prod = df_filtered.groupby(col_prod).agg({'revenue': 'mean','title': 'count'}).reset_index()
        df_top_prod.columns = ['Productora', 'Revenue_Promedio', 'Cantidad_Peliculas']
        df_top_10_prod = df_top_prod.sort_values('Revenue_Promedio', ascending=False).head(10)
        df_top_10_prod = df_top_10_prod.sort_values('Revenue_Promedio', ascending=True)
        fig_top_prod = px.bar(
            df_top_10_prod, x='Revenue_Promedio', y='Productora',orientation='h',title="<b>Top 10 Productoras con Mayor Revenue Promedio</b>",
            labels={'Revenue_Promedio': 'Revenue Promedio (USD)', 'Productora': 'Compañía'},
            color='Revenue_Promedio',
            color_continuous_scale='GnBu', # Escala de verdes/azules
            text_auto='.3s', # Muestra el valor abreviado (ej: 1.2B o 450M)
            hover_data={'Cantidad_Peliculas': True}
        )
        fig_top_prod.update_layout(
            height=600,
            margin=dict(l=200),
            xaxis_title="Revenue Promedio (USD)",
            yaxis_title="",
            coloraxis_showscale=False
        )
        fig_top_prod.update_traces(textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_top_prod, use_container_width=True, key="grafico_top_10_prod")
        top_empresa = df_top_10_prod.iloc[-1]['Productora']
        top_valor = df_top_10_prod.iloc[-1]['Revenue_Promedio']
        st.info(f"💡 **Insight de Mercado:** La productora **{top_empresa}** lidera el ranking con una recaudación promedio de **${top_valor/1e6:.1f}M** por película en el periodo seleccionado.")
    else:
        st.warning("No se encontró la columna de productoras agrupadas.")

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
    st.markdown("---")
    st.subheader("Estimación del Modelo Seleccionado")
    genre_cols = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 
              'Documentary', 'Otro', 'Family', 'Fantasy', 'Foreign', 
              'History', 'Horror', 'Music', 'Mystery', 'Romance', 
              'Science_Fiction', 'TV_Movie', 'Thriller', 'War', 'Western']

    genre_terms = ' + '.join(genre_cols)
    formula3 = f'log_revenue ~  log_budget + log_vote_count + vote_average  + is_english + C(release_month) + release_year + {genre_terms}'
    
    try:
        modelo3 = smf.ols(formula=formula3, data=df_filtered).fit()
        
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.metric("R-Squared (Ajustado)", f"{modelo3.rsquared_adj:.3f}")
            st.metric("Número de Observaciones", f"{int(modelo3.nobs)}")
            st.metric("F-Estadístico", f"{modelo3.fvalue:.2f}")
            st.write("**Elasticidades Encontradas:**")
            st.info(f"Budget: **{modelo3.params['log_budget']:.4f}**")
            st.info(f"Votos: **{modelo3.params['log_vote_count']:.4f}**")
            
            if modelo3.pvalues['log_budget'] < 0.05:
                st.success("Presupuesto Estadísticamente Significativo")
            else:
                st.warning("Presupuesto No Significativo")

        with m_col2:
            st.write("**Tabla de Coeficientes:**")
            # SOLUCIÓN StringIO para evitar el error de [Errno 2]
            tabla_html = modelo3.summary().tables[1].as_html()
            df_coef = pd.read_html(StringIO(tabla_html), header=0, index_col=0)[0]
            st.dataframe(df_coef, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error en la regresión: {e}. Intenta ajustar los filtros para tener más datos.")
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
