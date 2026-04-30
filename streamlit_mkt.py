from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.formula.api as smf
from PIL import Image
import plotly.graph_objects as go
import requests

def get_movie_poster(movie_title, year=None):
    api_key = "8f89112bc2feaa6f8f93dcf025a44917"
    # Si hay año, lo añadimos a la query de búsqueda de la API
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    if year:
        search_url += f"&year={year}"
    
    try:
        response = requests.get(search_url).json()
        if response['results']:
            # La API devuelve los resultados ordenados por relevancia/popularidad
            poster_path = response['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
        return "https://via.placeholder.com/500x750?text=No+Poster"
    except:
        return "https://via.placeholder.com/500x750?text=Error+API"
# Configuración profesional para la UdeC
st.set_page_config(page_title="Grupo 15 Marketing - Estimación de Demanda", layout="wide")

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
st.sidebar.header("Filtros")

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

tab_analisis, tab_modelo, tab_simulador = st.tabs(["Analisis de Datos", "Modelo de regresión", "Simulador Estrategico"])
with tab_analisis:
    st.header("Supuestos que utilizamos para el análisis")
    st.markdown("""- **Supuesto 1: Deflación de Ingresos:** Para comparar películas de diferentes años, ajustamos los ingresos a dólares constantes usando el IPC. Esto nos permite analizar la demanda real sin distorsiones por inflación.""")
    st.markdown("""- **Supuesto 2: Independencia de Observaciones:** Asumimos que cada película es una observación independiente,asumiendo que el éxito de una secuela o de una película de la misma franquicia se captura a través de su propio presupuesto y volumen de votos.""")
    st.markdown("---")
    st.header("Análisis de los datos")
    st.markdown("---")
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
    st.subheader("Análisis de Recaudación por Género")
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
    title="<b>Revenue Promedio por Género </b>",
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

    st.subheader("Participación de las productoras en el mercado")
    #Grafico de torta para mostrar la participación de las productoras en el mercado numero de peliculas de cada una)
    if 'compania_principal_agrupadas2' in df_filtered.columns:
        df_prod_count = df_filtered['compania_principal_agrupadas2'].value_counts().reset_index()
        df_prod_count.columns = ['Productora', 'Cantidad_Peliculas']
        total_pelis = df_prod_count['Cantidad_Peliculas'].sum()
        df_prod_count['Porcentaje'] = (df_prod_count['Cantidad_Peliculas'] / total_pelis) * 100
        umbral=1.5
        mask_minoritarias=(df_prod_count['Porcentaje'] < umbral) & (df_prod_count['Productora'] != '0_Otros')
        suma_minoritarias = df_prod_count.loc[mask_minoritarias, 'Cantidad_Peliculas'].sum()
        actual_otros=0
        if '0_Otros' in df_prod_count['Productora'].values:
            actual_otros = df_prod_count.loc[df_prod_count['Productora'] == '0_Otros', 'Cantidad_Peliculas'].values[0]
        df_pie_final=df_prod_count[~mask_minoritarias & (df_prod_count['Productora'] != '0_Otros')].copy()
        fila_otros_total = pd.DataFrame([{
        'Productora': '0_Otros', 
        'Cantidad_Peliculas': actual_otros + suma_minoritarias
    }])
        df_pie_final = pd.concat([df_pie_final, fila_otros_total], ignore_index=True)
        
        fig_pie = px.pie(
        df_pie_final, 
        values='Cantidad_Peliculas', 
        names='Productora', 
        title="<b>Participación de Productoras en el Mercado</b>",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True, key="grafico_participacion_productoras")
    st.markdown("---")
    st.subheader("Análisis de las Top 10 Productoras")
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
        
    st.subheader("Películas Estrella por Productora")
    st.markdown("Estas son las películas con mayor recaudación de cada una de las Top 10 productoras.")
    # 1. Obtenemos los nombres de las Top 10 productoras (reutilizando el cálculo anterior)
    top_10_nombres = df_top_10_prod['Productora'].tolist()
    mejores_peliculas = []
    genre_cols_2= ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama',
              'Documentary', 'Otro', 'Family', 'Fantasy', 'Foreign', 
              'History', 'Horror', 'Music', 'Mystery', 'Romance', 
              'Science_Fiction', 'TV_Movie', 'Thriller', 'War', 'Western']
    mapear_mes={1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    for prod in top_10_nombres:
        peli_top = df_filtered[df_filtered[col_prod] == prod].sort_values('revenue', ascending=False).iloc[0]
        generos_asociados = [g for g in genre_cols_2 if g in peli_top and peli_top[g] == 1]
        generos_texto= ", ".join(generos_asociados) if generos_asociados else "Sin Género Específico"
        mejores_peliculas.append({'Productora': prod,
            'Pelicula': peli_top['title'],
            'Revenue': peli_top['revenue'],
            'Presupuesto': peli_top['budget'],
            'año': peli_top['release_year'],
            'mes': mapear_mes.get(peli_top['release_month'], 'Desconocido'),   # está en numeros del 1 al 12, si quieres mostrar el nombre del mes, puedes usar un diccionario para mapearlo
            'Generos': generos_texto})
    df_estrellas=pd.DataFrame(mejores_peliculas).sort_values('Revenue', ascending=False).head(10)
    cols_por_fila = 5
    colms = st.columns(cols_por_fila)
    for i, (index, row) in enumerate(df_estrellas.iterrows()):
        with colms[i% cols_por_fila]:
            st.subheader(f"{row['Productora']}")
            st.write(f"**Película:** {row['Pelicula']}")
            st.write(f"**Géneros:** {row['Generos']}")
            st.write(f"**Revenue:** ${row['Revenue']/1e6:.2f}M")
            st.write(f"**Presupuesto:** ${row['Presupuesto']/1e6:.2f}M")
            #año y mes de estreno
            st.write(f"**Año de Estreno:** {row['año']}")
            st.write(f"**Mes de Estreno:** {row['mes']}")
            #  # Si tienes la columna del mes, cámbiala aquí
            poster = get_movie_poster(row['Pelicula'],row['año'])
            st.image(poster, caption=row['Pelicula'])
            if i < 5:
                st.write("")
    with st.expander("Ver lista completa de películas líderes"):
        st.dataframe(df_estrellas[['Productora', 'Pelicula','Generos', 'Revenue']].sort_values('Revenue', ascending=False), use_container_width=True)
# analisis de estacionalidad
    st.subheader("Estacionalidad del mercado")
    df_mes = df_filtered.groupby('release_month').agg({'revenue': 'mean', 'title': 'count'}).reset_index()
    df_mes.columns = ['Mes', 'Revenue_Promedio', 'Cantidad_Peliculas']
    fig_mes = px.bar(df_mes, x='Mes', y='Revenue_Promedio', title="<b>Revenue Promedio por Mes de Estreno</b>",
                     labels={'Revenue_Promedio': 'Revenue Promedio (USD)', 'Mes': 'Mes del Año'},
                     color='Revenue_Promedio', color_continuous_scale='OrRd', text_auto='.2s', hover_data={'Cantidad_Peliculas': True})
    fig_mes.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(range(1, 13)), ticktext=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']),
        xaxis_title="Mes del Año")
    st.plotly_chart(fig_mes, use_container_width=True, key="grafico_estacionalidad")

# --- TAB 2: TRANSFORMACIÓN LOGARÍTMICA ---
with tab_modelo:
    st.header("Transformación Logarítmica")
    st.write("Para corregir el sesgo y cumplir los supuestos de la regresión lineal, transformamos las variables.")
    var_to_show = st.selectbox("Seleccione Variable a Comparar", 
                            options=['budget', 'revenue', 'vote_count'],
                            format_func=lambda x: "Presupuesto" if x=='budget' else ("Ingresos" if x=='revenue' else "Votos"))
    col_raw, col_log = st.columns(2)
    with col_raw:
        st.subheader("Distribución Original (Sesgada)")
        fig_raw = px.histogram(df_filtered, x=var_to_show, 
                           nbins=50, 
                           title=f"Original: {var_to_show}",
                           color_discrete_sequence=['#EF4444'], # Rojo para indicar sesgo
                           opacity=0.7)
        st.plotly_chart(fig_raw, use_container_width=True)
        st.caption("Nota el 'muro' a la izquierda: la mayoría son películas de bajo presupuesto/ingreso.")
    with col_log:
        st.subheader("Distribución Logarítmica (Normalizada)")
        log_col_name = f"log_{var_to_show}" if f"log_{var_to_show}" in df_filtered.columns else var_to_show
        fig_log = px.histogram(df_filtered, x=log_col_name, 
                           nbins=50, 
                           title=f"Log-Transformada: {log_col_name}",
                           color_discrete_sequence=['#10B981'], # Verde para indicar normalidad
                           opacity=0.7)
        st.plotly_chart(fig_log, use_container_width=True)
        st.caption("Tras el logaritmo, los datos se distribuyen de forma campana (normal), ideal para el modelo OLS.")
        st.markdown("---")
    with st.container():
        st.subheader("Matriz de correlación (variables cuantitativas)")
        numeric_cols = ['log_budget', 'log_revenue', 'log_vote_count', 'vote_average']
        corr_matrix = df_filtered[numeric_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Correlación entre Variables Cuantitativas")
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("---")
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
        
        if 'modelo3' in locals():
            st.subheader("Gráfico de Valores Reales vs Predichos (Log-Log)")
            df_filtered['pred_log_revenue'] = modelo3.fittedvalues
            fig_pred = px.scatter(df_filtered, x='log_revenue', y='pred_log_revenue',
                                  title="Valores Reales vs Predichos (Log-Log)",
                                  labels={'log_revenue': 'Log(Revenue Real)', 'pred_log_revenue': 'Log(Revenue Predicho)'},
                                  color_discrete_sequence=['#990000'])
            fig_pred.add_shape(type='line', x0=df_filtered['log_revenue'].min(), y0=df_filtered['log_revenue'].min(),
                               x1=df_filtered['log_revenue'].max(), y1=df_filtered['log_revenue'].max(),
                               line=dict(color='blue', dash='dash'))
            st.plotly_chart(fig_pred, use_container_width=True)

        
with tab_simulador:
    st.header("Simulador de Ingresos Estimados")
    st.markdown("Ingresa los parámetros de tu proyecto para obtener una predicción basada en el modelo de regresión.")

    if 'modelo3' in locals():
        col_input, col_result = st.columns([1, 1])

        with col_input:
            st.subheader("Configuración del Proyecto")
            
            # 1. Presupuesto
            presupuesto_input = st.number_input("Presupuesto de Producción (USD)", 
                                               min_value=100000, 
                                               value=50000000, 
                                               step=500000)
            
            # 2. Selección de Géneros (Multiselect para que el cliente elija el mix)
            # Nota: El modelo usará el impacto de estos géneros si están incluidos en la fórmula
            generos_sel = st.multiselect("Géneros de la película", genre_cols_2, default=['Action'])
            
            # 3. Idioma
            idioma_opcion = st.radio("¿Idioma original Inglés?", ["Sí", "No"], horizontal=True)
            is_english_input = 1 if idioma_opcion == "Sí" else 0
            
            # 4. Mes de estreno
            mes_input = st.slider("Mes de estreno planificado", 1, 12, 6)
            
            # 5. Año de estreno (Opcional: si tu modelo no tiene la variable 'year', 
            # úsalo para contextualizar el análisis de mercado)
            anio_input = st.number_input("Año de estreno", min_value=2026, max_value=2040, value=2027)
            
            # 6. Variables de control (Votos y Calificación promedio de la muestra para simular)
            st.caption("Variables de control (basadas en promedios históricos)")
            votos_sim = st.number_input("Votos esperados (Engagement)", value=int(df_filtered['vote_count'].mean()))
            rating_sim = st.slider("Calificación esperada (Rating)", 1.0, 10.0, 6.5)

        # --- LÓGICA DE PREDICCIÓN ---
        # Transformamos a logaritmo lo que el modelo requiere
        log_b_input = np.log(presupuesto_input)
        log_v_input = np.log(votos_sim)
        
        # Crear DataFrame para el modelo con los nombres de variables EXACTOS de tu fórmula
        df_input_sim = pd.DataFrame({
            'log_budget': [log_b_input],
            'log_vote_count': [log_v_input],
            'vote_average': [rating_sim],
            'is_english': [is_english_input],
            'release_month': [float(mes_input)],
            'release_year': [float(anio_input)],
        })
        #añadir los generos seleccionados al df_input_sim
        for g in genre_cols_2:
            df_input_sim[g] = 1 if g in generos_sel else 0

        try:
            # Predicción en escala logarítmica
            pred_log_revenue = modelo3.predict(df_input_sim)[0]
            # Convertir de vuelta a dólares reales
            revenue_estimado = np.exp(pred_log_revenue)
            utilidad = revenue_estimado - presupuesto_input
            roi_ratio = revenue_estimado / presupuesto_input
            
            with col_result:
                st.subheader("Resultados de la Simulación")
                
                st.metric("Recaudación Estimada", f"${revenue_estimado/1e6:.2f}M")
                
                # Color del delta según si hay ganancia o pérdida
                st.metric("Utilidad Neta Est.", f"${utilidad/1e6:.2f}M", 
                          delta=f"{roi_ratio:.2f}x ROI", 
                          delta_color="normal" if utilidad > 0 else "inverse")
                
                st.write("---")
                # Análisis Estratégico
                st.markdown("### Recomendaciones")
                
                if revenue_estimado > presupuesto_input * 2:
                    st.success("**Potencial Blockbuster:** El modelo estima que los ingresos duplicarán la inversión.")
                elif revenue_estimado > presupuesto_input:
                    st.warning("**Punto de Equilibrio:** El proyecto es rentable pero con márgenes ajustados.")
                else:
                    st.error("**Riesgo Elevado:** La recaudación estimada no cubre los costos de producción.")

                # Tip sobre el mes de estreno
                meses_pico = [6, 7, 12] # Meses de verano y navidad
                if mes_input in meses_pico:
                    st.info(f"📅 **Ventaja Estacional:** Estrenar en el mes {mes_input} aprovecha periodos de alta demanda histórica.")
                
                # Tip sobre idiomas
                if is_english_input == 0:
                    st.caption("Nota: Las películas en idiomas distintos al inglés suelen tener una recaudación base menor en el mercado global según los datos.")

        except Exception as e:
            st.error(f"Error en la simulación: {e}")
    else:
        st.warning("Debes ejecutar la regresión en la pestaña anterior para activar el simulador.")
    with st.expander("Detalles de las formulas"):
        st.markdown("#### Modelo de regresión log-log")
        st.write("El modelo predice el logaritmo del ingreso para capturar elasticidades:")
        st.latex(r"""
                    \begin{aligned}
                    \ln(\text{Revenue}) = \beta_0 & + \beta_1 \ln(\text{Budget}) + \beta_2 \ln(\text{Votes}) + \beta_3 (\text{Rating}) \\
                    & + \beta_4 (\text{Is English}) + \beta_5 (\text{Year}) + \sum_{m=1}^{12} \gamma_m (\text{Month}_m) \\
                    & + \sum_{g \in \text{Generos}} \delta_g (\text{Género}_g) + \epsilon
                    \end{aligned}
                    """)
        st.write("Para volver a valores monetarios(USD), se aplica la función exponencial al resultado del modelo:")
        st.latex(r"\text{Revenue} = e^{\widehat{\ln(\text{Revenue})}}")
        st.markdown("#### Cálculo del Return on Investment (ROI):")
        st.write("Representa la eficiencia del capital invertido:")
        st.latex(r"\text{ROI} = \frac{\text{Revenue Estimado}}{\text{Presupuesto Invertido}}")
        st.info("Un ROI mayor a 1 indica que el proyecto es rentable, mientras que un ROI menor a 1 sugiere una posible pérdida.")


st.markdown("---")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.caption("Trabajo I - Marketing - DII UdeC 2026")
with col_f2:
    st.markdown("<div style='text-align: right; color: gray; font-size: 0.8em;'>"
                "Grupo 15 : Rocio Arriagada, Diego Fernandez, Martin Lagos, Camila Leiva, Camilo Mora, Hernán Saavedra </div>", 
                unsafe_allow_html=True)
