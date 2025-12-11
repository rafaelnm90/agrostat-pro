import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.express as px

# --- CONFIGURAÇÃO DE LOGS ---
EXIBIR_LOGS = True

def log_message(mensagem):
    if EXIBIR_LOGS:
        print(mensagem)

# --- FUNÇÃO GERADORA DE LETRAS (SEGURA) ---
def get_letra_segura(n):
    """
    Gera sequência segura: a, b...z, a1, b1...z1, a2...
    Evita confusão com intersecções estatísticas (como 'ab').
    """
    ciclo = n // 26
    letra_idx = n % 26
    letra = chr(97 + letra_idx) 
    
    if ciclo == 0:
        return letra
    else:
        return f"{letra}{ciclo}"

# --- FUNÇÕES DE INTERPRETAÇÃO ---
def explicar_anova(p_val, shapiro_p, bartlett_p, metodo_usado):
    texto_sig = ""
    cor_sig = ""
    if p_val < 0.05:
        texto_sig = f"✅ **Conclusão da ANOVA:** O valor-p ({p_val:.4f}) é inferior a 0.05. **Há diferença significativa** entre os tratamentos."
        cor_sig = "success"
    else:
        texto_sig = f"⚠️ **Conclusão da ANOVA:** O valor-p ({p_val:.4f}) é superior a 0.05. Não há evidência de diferença estatística."
        cor_sig = "warning"
        
    texto_shapiro = "segue uma distribuição Normal" if shapiro_p > 0.05 else "não segue estritamente a Normalidade"
    texto_bartlett = "são Homogêneas" if bartlett_p > 0.05 else "são Heterogêneas"
    
    nota_metodo = ""
    if "Sequencial Ajustado" in metodo_usado:
        nota_metodo = " (Nota: Devido à complexidade dos dados, foi utilizado o método Sequencial Ajustado para garantir estabilidade)."
    
    resumo_pressupostos = f"A análise dos resíduos mostra que a distribuição dos dados **{texto_shapiro}** e as variâncias entre os grupos **{texto_bartlett}**.{nota_metodo}"
    return texto_sig, cor_sig, resumo_pressupostos

def explicar_ranking(df_resultado, nome_teste):
    df_sorted = df_resultado.sort_values('Media', ascending=False)
    lider_trat = df_sorted.index[0]
    lider_media = df_sorted.iloc[0]['Media']
    
    col_letra = 'Letras' if 'Letras' in df_sorted.columns else 'Grupo'
    letra_lider = df_sorted.iloc[0][col_letra]
    
    empates = []
    for trat in df_sorted.index[1:]:
        letra_trat = df_sorted.loc[trat, col_letra]
        eh_igual = False
        if nome_teste == "Scott-Knott":
            if letra_trat == letra_lider: eh_igual = True
        else:
            set_lider = set(letra_lider)
            set_trat = set(letra_trat)
            if not set_lider.isdisjoint(set_trat): eh_igual = True
        
        if eh_igual: empates.append(trat)
            
    texto = f"📊 **Análise de Desempenho ({nome_teste}):**\n\n"
    texto += f"🥇 **Líder Numérico:** O tratamento **{lider_trat}** obteve a maior média absoluta ({lider_media:.2f}).\n\n"
    
    if empates:
        qtd_mostra = 5
        if len(empates) > qtd_mostra:
            lista_empates = ", ".join(empates[:qtd_mostra]) + f" e outros {len(empates)-qtd_mostra}"
        else:
            lista_empates = ", ".join(empates)
            
        texto += (
            f"🤝 **Empate Estatístico:** Atenção! O líder **{lider_trat}** NÃO difere estatisticamente de: **{lista_empates}**.\n"
            f"Todos estes formam o grupo superior."
        )
    else:
        texto += f"🏆 **Superioridade Absoluta:** O tratamento **{lider_trat}** diferiu estatisticamente de todos os demais."

    return texto

# --- ALGORITMOS MATEMÁTICOS ---
def scott_knott(means, mse, df_resid, reps):
    results = pd.DataFrame({'Media': means}).sort_values('Media', ascending=False)
    medias_ordenadas = results['Media'].values
    indices = results.index
    
    def cluster_medias(meds, ind):
        n = len(meds)
        if n < 2: return {ind[0]: 1}
        melhor_b0, corte_idx = -1, -1
        grand_mean = np.mean(meds)
        for i in range(1, n):
            g1, g2 = meds[:i], meds[i:]
            b0 = i * (np.mean(g1) - grand_mean)**2 + (n-i) * (np.mean(g2) - grand_mean)**2
            if b0 > melhor_b0: melhor_b0, corte_idx = b0, i
        sigma2 = mse / reps
        lamb = (np.pi / (2 * (np.pi - 2))) * (melhor_b0 / sigma2)
        critico = stats.chi2.ppf(0.95, df=n/(np.pi-2)) 
        if lamb > critico:
            dict_left = cluster_medias(meds[:corte_idx], ind[:corte_idx])
            dict_right = cluster_medias(meds[corte_idx:], ind[corte_idx:])
            max_grp = max(dict_left.values())
            for k in dict_right: dict_right[k] += max_grp
            return {**dict_left, **dict_right}
        else: return {x: 1 for x in ind}

    grupos_dict = cluster_medias(medias_ordenadas, indices)
    results['Grupo_Num'] = results.index.map(grupos_dict)
    
    unique_grps = sorted(results['Grupo_Num'].unique())
    mapa_letras = {num: get_letra_segura(i) for i, num in enumerate(unique_grps)}
    
    results['Grupo'] = results['Grupo_Num'].map(mapa_letras)
    return results[['Media', 'Grupo']]

def gerar_letras_tukey(tukey_res, medias):
    df_letras = pd.DataFrame({'Media': medias}).sort_values('Media', ascending=False)
    trats = df_letras.index.tolist()
    adj = {t: set() for t in trats} 
    tukey_data = pd.DataFrame(data=tukey_res._results_table.data[1:], columns=tukey_res._results_table.data[0])
    for _, row in tukey_data.iterrows():
        if not row['reject']:
            g1, g2 = row['group1'], row['group2']
            if g1 in adj and g2 in adj: adj[g1].add(g2); adj[g2].add(g1)

    cliques = []
    def bron_kerbosch(R, P, X):
        if not P and not X: cliques.append(R); return
        if not P: return
        try: u = list(P | X)[0]; vizinhos_u = adj[u]
        except: vizinhos_u = set()
        for v in list(P - vizinhos_u):
            bron_kerbosch(R | {v}, P & adj[v], X & adj[v])
            P.remove(v); X.add(v)
    bron_kerbosch(set(), set(trats), set())
    if not cliques: cliques = [{t} for t in trats]

    cliques_info = []
    for c in cliques:
        media_clique = df_letras.loc[list(c), 'Media'].mean()
        cliques_info.append({'membros': c, 'media': media_clique})
    cliques_info.sort(key=lambda x: x['media'], reverse=True)
    
    mapa_letras = {t: [] for t in trats}
    for i, clique in enumerate(cliques_info):
        letra = get_letra_segura(i)
        for membro in clique['membros']:
            if letra not in mapa_letras[membro]: mapa_letras[membro].append(letra)
            
    mapa_final = {}
    for t in mapa_letras:
        mapa_final[t] = "".join(sorted(mapa_letras[t]))
        
    df_letras['Letras'] = df_letras.index.map(mapa_final)
    return df_letras

# --- INTERFACE PRINCIPAL ---
st.set_page_config(page_title="AgroStat Pro", page_icon="🌱", layout="wide")
st.title("🌱 AgroStat Pro: Análise Estatística Inteligente")

def rodar_analise(df, col_trat, col_resp, delineamento, col_bloco=None):
    log_message(f"🚀 Iniciando: {col_resp}")
    res = {}
    res['descritiva'] = df.groupby(col_trat)[col_resp].agg(['mean', 'std', 'count']).reset_index()
    
    try:
        log_message("Tentando ANOVA Tipo III...")
        if delineamento == 'DBC':
            if not col_bloco: return "⚠️ Selecione a coluna de Blocos!"
            formula = f"{col_resp} ~ C({col_trat}) + C({col_bloco})"
        else: formula = f"{col_resp} ~ C({col_trat})"
            
        modelo = ols(formula, data=df).fit()
        anova = sm.stats.anova_lm(modelo, typ=3)
        metodo_usado = "Tipo III (Marginal)"
    except ValueError:
        log_message("⚠️ Fallback para Tipo I Ajustado...")
        if delineamento == 'DBC': formula = f"{col_resp} ~ C({col_bloco}) + C({col_trat})" 
        else: formula = f"{col_resp} ~ C({col_trat})"
        modelo = ols(formula, data=df).fit()
        anova = sm.stats.anova_lm(modelo, typ=1) 
        metodo_usado = "Sequencial Ajustado (Fallback Robusto)"

    res['anova'], res['residuos'], res['modelo'] = anova, modelo.resid, modelo
    res['metodo'] = metodo_usado
    
    mse = modelo.mse_resid
    df_resid = modelo.df_resid
    reps_media = df.groupby(col_trat)[col_resp].count().mean() 
    res['cv'] = (np.sqrt(mse) / df[col_resp].mean()) * 100
    res['shapiro'] = stats.shapiro(modelo.resid)
    grupos = [g[col_resp].values for _, g in df.groupby(col_trat)]
    res['bartlett'] = stats.bartlett(*grupos)
    
    p_val = anova.loc[f"C({col_trat})", "PR(>F)"]
    if p_val < 0.05:
        medias = df.groupby(col_trat)[col_resp].mean()
        tukey = pairwise_tukeyhsd(endog=df[col_resp], groups=df[col_trat], alpha=0.05)
        res['tukey_raw'] = tukey
        res['tukey_df'] = gerar_letras_tukey(tukey, medias)
        res['scott_knott'] = scott_knott(medias, mse, df_resid, reps_media)
    else: res['tukey_df'], res['scott_knott'] = None, None
    return res

st.sidebar.header("📂 Entrada de Dados")
arquivo = st.sidebar.file_uploader("Upload CSV ou Excel", type=["xlsx", "csv"])

if arquivo:
    if arquivo.name.endswith('.csv'): df = pd.read_csv(arquivo)
    else: df = pd.read_excel(arquivo)
    st.sidebar.success("Dados carregados!")
    colunas = df.columns.tolist()
    st.sidebar.markdown("---")
    tipo_del = st.sidebar.radio("Delineamento:", ("DIC (Inteiramente Casualizado)", "DBC (Blocos Casualizados)"))
    delineamento = "DIC" if "DIC" in tipo_del else "DBC"
    col_trat = st.sidebar.selectbox("Tratamentos", colunas)
    col_resp = st.sidebar.selectbox("Variável Resposta", [c for c in colunas if c != col_trat])
    col_bloco = None
    if delineamento == "DBC": col_bloco = st.sidebar.selectbox("Blocos", [c for c in colunas if c not in [col_trat, col_resp]])

    if st.sidebar.button("🚀 Processar Estatística"):
        resultado = rodar_analise(df, col_trat, col_resp, delineamento, col_bloco)
        if isinstance(resultado, str): st.error(resultado)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Média Geral", f"{df[col_resp].mean():.2f}")
            c2.metric("CV (%)", f"{resultado['cv']:.2f}%")
            p_val = resultado['anova'].loc[f"C({col_trat})", "PR(>F)"]
            sig = p_val < 0.05
            c3.metric("ANOVA", "Significativo" if sig else "Não Significativo", delta_color="inverse" if sig else "normal")
            
            st.markdown("---")
            t1, t2, t3, t4 = st.tabs(["📋 ANOVA & Pressupostos", "📏 Teste Tukey", "📦 Teste Scott-Knott", "📈 Gráficos"])
            
            with t1:
                st.subheader("1. Quadro da ANOVA")
                st.caption(f"⚙️ Método de Cálculo: {resultado['metodo']}")
                st.dataframe(resultado['anova'].style.format("{:.4f}"))
                st.subheader("2. Testes de Pressupostos")
                c_a, c_b = st.columns(2)
                sp = resultado['shapiro'][1]
                bp = resultado['bartlett'][1]
                c_a.metric("Normalidade (Shapiro-Wilk)", f"P = {sp:.4f}", delta="Normal" if sp > 0.05 else "Não Normal", delta_color="normal" if sp > 0.05 else "inverse")
                c_b.metric("Homogeneidade (Bartlett)", f"P = {bp:.4f}", delta="Homogêneo" if bp > 0.05 else "Heterogêneo", delta_color="normal" if bp > 0.05 else "inverse")
                st.markdown("---")
                st.markdown("#### 🧠 Interpretação do Assistente")
                txt_sig, cor_sig, txt_press = explicar_anova(p_val, sp, bp, resultado['metodo'])
                if cor_sig == "success": st.success(txt_sig)
                else: st.warning(txt_sig)
                st.info(f"📉 **Sobre os Dados:** {txt_press}")

            with t2:
                if resultado['tukey_df'] is not None:
                    st.subheader("Tabela de Médias (Tukey 5%)")
                    st.dataframe(resultado['tukey_df'].style.format({"Media": "{:.2f}"}))
                    with st.expander("Ver estatísticas detalhadas (P-values)"):
                        st.text(resultado['tukey_raw'].summary())
                    st.markdown("---")
                    st.markdown("#### 🧠 Interpretação do Ranking")
                    st.info(explicar_ranking(resultado['tukey_df'], "Tukey"))
                else: st.warning("Teste não realizado (ANOVA não significativa).")
                
            with t3:
                if resultado['scott_knott'] is not None:
                    st.subheader("Agrupamento de Médias (Scott-Knott 5%)")
                    st.dataframe(resultado['scott_knott'].style.format({"Media": "{:.2f}"}))
                    st.markdown("---")
                    st.markdown("#### 🧠 Interpretação do Agrupamento")
                    st.info(explicar_ranking(resultado['scott_knott'], "Scott-Knott"))
                else: st.warning("Teste não realizado (ANOVA não significativa).")
                
            with t4:
                fig = px.box(df, x=col_trat, y=col_resp, title="Dispersão dos Dados")
                st.plotly_chart(fig, use_container_width=True)
                
                # GRÁFICO 1: TUKEY
                if resultado['tukey_df'] is not None:
                    dg = resultado['tukey_df'].reset_index().rename(columns={'index': col_trat})
                    fig2 = px.bar(dg, x=col_trat, y='Media', text='Letras', title="Ranking de Médias (Tukey)")
                    fig2.update_traces(textposition='outside')
                    st.plotly_chart(fig2, use_container_width=True)
                
                # GRÁFICO 2: SCOTT-KNOTT (NOVIDADE)
                if resultado['scott_knott'] is not None:
                    dg_sk = resultado['scott_knott'].reset_index().rename(columns={'index': col_trat})
                    # Note que usamos 'Grupo' no lugar de 'Letras'
                    fig3 = px.bar(dg_sk, x=col_trat, y='Media', text='Grupo', title="Ranking de Médias (Scott-Knott)")
                    fig3.update_traces(textposition='outside')
                    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("👈 Faça upload do arquivo para começar a análise.")