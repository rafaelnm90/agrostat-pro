import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import studentized_range
import statsmodels.api as sm
from statsmodels.formula.api import ols
import plotly.express as px

# --- CONFIGURAÇÃO DE LOGS ---
EXIBIR_LOGS = True

def log_message(mensagem):
    if EXIBIR_LOGS:
        print(mensagem)

# --- UTILITÁRIOS ---
def get_letra_segura(n):
    """Gera sequência a, b...z, a1, b1... de forma segura."""
    try:
        ciclo = int(n) // 26
        letra_idx = int(n) % 26
        letra = chr(97 + letra_idx) 
        if ciclo == 0: return letra
        else: return f"{letra}{ciclo}"
    except:
        return "?"

def formatar_tabela_anova(anova_df):
    """Padrão científico ABNT + Coluna Sig."""
    cols_map = {'sum_sq': 'SQ', 'df': 'GL', 'F': 'Fcalc', 'PR(>F)': 'P-valor'}
    df = anova_df.rename(columns=cols_map)
    df.insert(2, 'QM', df['SQ'] / df['GL'])
    
    if 'Intercept' in df.index: df = df.drop('Intercept')
        
    new_index = []
    for idx in df.index:
        nome = idx.replace('C(', '').replace(')', '').replace(':', ' x ')
        if 'Residual' in nome: nome = 'Resíduo'
        new_index.append(nome)
    df.index = new_index
    
    def verificar_sig(p):
        if pd.isna(p): return "" 
        if p < 0.001: return "***" 
        if p < 0.01: return "**"   
        if p < 0.05: return "*"    
        return "ns"                
    
    df['Sig.'] = df['P-valor'].apply(verificar_sig)
    return df

def classificar_cv(cv):
    """Classifica o CV segundo Pimentel-Gomes (2009)."""
    if cv < 10:
        return "🟢 Baixo (Alta Precisão)"
    elif cv < 20:
        return "🟡 Médio (Boa Precisão)"
    elif cv < 30:
        return "🟠 Alto (Baixa Precisão)"
    else:
        return "🔴 Muito Alto (Inadequado)"

# --- ALGORITMOS MATEMÁTICOS ---

def tukey_manual_preciso(medias, mse, df_resid, n_reps, n_trats):
    """
    Implementação Manual do Tukey HSD com precisão do R.
    """
    ep = np.sqrt(mse / n_reps)
    q_crit = studentized_range.ppf(0.95, n_trats, df_resid)
    hsd = q_crit * ep
    
    trats = medias.index.tolist()
    adj = {t: set() for t in trats}
    
    sorted_medias = medias.sort_values(ascending=False)
    vals = sorted_medias.values
    keys = sorted_medias.index
    
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            diff = abs(vals[i] - vals[j])
            if diff < hsd: 
                t1, t2 = keys[i], keys[j]
                adj[t1].add(t2)
                adj[t2].add(t1)
                
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
        media_clique = medias.loc[list(c)].mean()
        cliques_info.append({'membros': c, 'media': media_clique})
    
    cliques_info.sort(key=lambda x: x['media'], reverse=True)
    
    mapa_letras = {t: [] for t in trats}
    for i, clique in enumerate(cliques_info):
        letra = get_letra_segura(i)
        for membro in clique['membros']:
            if letra not in mapa_letras[membro]: 
                mapa_letras[membro].append(letra)
                
    mapa_final = {}
    for t in trats:
        mapa_final[t] = "".join(sorted(mapa_letras[t]))
        
    df_res = pd.DataFrame({'Media': medias, 'Letras': pd.Series(mapa_final)})
    return df_res.sort_values('Media', ascending=False)

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

def explaining_ranking(df_resultado, nome_teste):
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
    texto += f"🥇 **Líder Numérico:** **{lider_trat}** (Média: {lider_media:.2f}).\n"
    
    if empates:
        qtd_mostra = 5
        lista_empates = ", ".join(empates[:qtd_mostra]) + (f" e outros {len(empates)-qtd_mostra}" if len(empates) > qtd_mostra else "")
        texto += f"🤝 **Empate Estatístico:** O líder não difere de: **{lista_empates}**."
    else:
        texto += f"🏆 **Superioridade Absoluta:** O tratamento diferiu estatisticamente de todos os demais."
    return texto

def calcular_homogeneidade(df, col_trat, col_resp, col_local, col_bloco, delineamento):
    locais = df[col_local].unique()
    mses = {}
    
    for loc in locais:
        df_loc = df[df[col_local] == loc]
        if delineamento == 'DBC': formula = f"{col_resp} ~ C({col_trat}) + C({col_bloco})"
        else: formula = f"{col_resp} ~ C({col_trat})"
        try:
            modelo = ols(formula, data=df_loc).fit()
            mses[loc] = modelo.mse_resid
        except: pass 
            
    if not mses: return None, None, {}
    max_mse = max(mses.values())
    min_mse = min(mses.values())
    razao = max_mse / min_mse
    return razao, mses, {k: v for k, v in sorted(mses.items(), key=lambda item: item[1])}

def rodar_analise_individual(df, col_trat, col_resp, delineamento, col_bloco=None):
    res = {}
    if delineamento == 'DBC': formula = f"{col_resp} ~ C({col_trat}) + C({col_bloco})"
    else: formula = f"{col_resp} ~ C({col_trat})"
    
    try:
        modelo = ols(formula, data=df).fit()
        anova = sm.stats.anova_lm(modelo, typ=3)
    except:
        if delineamento == 'DBC': formula = f"{col_resp} ~ C({col_bloco}) + C({col_trat})"
        modelo = ols(formula, data=df).fit()
        anova = sm.stats.anova_lm(modelo, typ=1)
        
    res['anova'] = anova
    res['modelo'] = modelo
    res['mse'] = modelo.mse_resid
    res['df_resid'] = modelo.df_resid
    res['p_val'] = anova.loc[f"C({col_trat})", "PR(>F)"]
    res['shapiro'] = stats.shapiro(modelo.resid)
    grupos = [g[col_resp].values for _, g in df.groupby(col_trat)]
    res['bartlett'] = stats.bartlett(*grupos)
    
    return res

def rodar_analise_conjunta(df, col_trat, col_resp, col_local, delineamento, col_bloco=None):
    res = {}
    termos = f"C({col_trat}) + C({col_local}) + C({col_trat}):C({col_local})"
    if delineamento == 'DBC':
        termos += f" + C({col_bloco}):C({col_local})"
    formula = f"{col_resp} ~ {termos}"
    
    try:
        modelo = ols(formula, data=df).fit()
        anova = sm.stats.anova_lm(modelo, typ=3)
    except:
        modelo = ols(formula, data=df).fit()
        anova = sm.stats.anova_lm(modelo, typ=1)
        
    res['anova'] = anova
    res['modelo'] = modelo
    res['mse'] = modelo.mse_resid
    res['df_resid'] = modelo.df_resid
    
    try:
        res['p_trat'] = anova.loc[f"C({col_trat})", "PR(>F)"]
        res['p_interacao'] = anova.loc[f"C({col_trat}):C({col_local})", "PR(>F)"]
    except:
        res['p_trat'] = 0.0
        res['p_interacao'] = 0.0
        for idx in anova.index:
            if f"C({col_trat})" in idx and ":" not in idx: res['p_trat'] = anova.loc[idx, "PR(>F)"]
            if f"C({col_trat}):C({col_local})" in idx: res['p_interacao'] = anova.loc[idx, "PR(>F)"]
    return res

# --- INTERFACE PRINCIPAL ---
st.set_page_config(page_title="AgroStat Pro", page_icon="🌱", layout="wide")
st.title("🌱 AgroStat Pro: Análises Estatísticas")

# 1. SIDEBAR CONFIG
st.sidebar.header("📂 Configuração de Dados")
arquivo = st.sidebar.file_uploader("Upload CSV ou Excel", type=["xlsx", "csv"])

if arquivo:
    if arquivo.name.endswith('.csv'): df = pd.read_csv(arquivo)
    else: df = pd.read_excel(arquivo)
    colunas = df.columns.tolist()
    
    st.sidebar.success(f"Carregado: {len(df)} linhas")
    st.sidebar.markdown("---")
    
    tipo_del = st.sidebar.radio("Delineamento:", ("DIC", "DBC"))
    delineamento = "DIC" if "DIC" in tipo_del else "DBC"
    
    col_trat = st.sidebar.selectbox("Tratamentos (Genótipos)", colunas)
    
    OPCAO_PADRAO = "Local Único (Análise Individual)" 
    col_local = st.sidebar.selectbox("Local/Ambiente (Opcional)", [OPCAO_PADRAO] + [c for c in colunas if c != col_trat])
    
    col_bloco = None
    if delineamento == "DBC":
        col_bloco = st.sidebar.selectbox("Blocos", [c for c in colunas if c not in [col_trat, col_local]])

    cols_ocupadas = [col_trat, col_local]
    if col_bloco: cols_ocupadas.append(col_bloco)
    
    lista_resps = st.sidebar.multiselect("Variáveis Resposta (Selecione 1 ou mais)", [c for c in colunas if c not in cols_ocupadas])

    modo_analise = "INDIVIDUAL"
    if col_local != OPCAO_PADRAO:
        n_locais = len(df[col_local].unique())
        if n_locais > 1:
            modo_analise = "CONJUNTA"
            st.sidebar.info(f"🌍 Modo Conjunta Ativado! ({n_locais} locais detectados)")
        else:
            st.sidebar.warning("⚠️ Coluna de Local selecionada, mas há apenas 1 local. Rodando modo Individual.")

    if st.sidebar.button("🚀 Processar Estatística"):
        if not lista_resps:
            st.error("⚠️ Por favor, selecione pelo menos uma Variável Resposta.")
        else:
            st.markdown(f"### 📋 Resultados: {len(lista_resps)} variáveis processadas")
            
            for i, col_resp in enumerate(lista_resps):
                with st.expander(f"📊 Variável: {col_resp} (Clique para ver detalhes)", expanded=(i==0)):
                    st.markdown(f"### Análise de: **{col_resp}**")
                    
                    if modo_analise == "INDIVIDUAL":
                        res = rodar_analise_individual(df, col_trat, col_resp, delineamento, col_bloco)
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Média Geral", f"{df[col_resp].mean():.2f}")
                        
                        cv_val = (np.sqrt(res['mse'])/df[col_resp].mean())*100
                        class_cv = classificar_cv(cv_val)
                        c2.metric("CV (%)", f"{cv_val:.2f}%", delta=class_cv, delta_color="off", help="Classificação conforme Pimentel-Gomes (2009)")
                        
                        sig = res['p_val'] < 0.05
                        c3.metric("ANOVA", "Significativo" if sig else "ns", delta_color="inverse" if sig else "normal")
                        
                        t1, t2, t3, t4 = st.tabs(["📋 ANOVA", "📏 Teste Tukey", "📦 Teste Scott-Knott", "📈 Gráficos"])
                        
                        with t1:
                            anova_formatada = formatar_tabela_anova(res['anova'])
                            st.dataframe(anova_formatada.style.format({
                                "SQ": "{:.4f}", "GL": "{:.0f}", "QM": "{:.4f}", "Fcalc": "{:.4f}", "P-valor": "{:.4f}"
                            }))
                            st.caption("_Legenda: *** (P<0.001); ** (P<0.01); * (P<0.05); ns (Não Significativo)_")
                            st.info(f"Shapiro-Wilk: P={res['shapiro'][1]:.4f} | Bartlett: P={res['bartlett'][1]:.4f}")
                        
                        if sig:
                            reps = df.groupby(col_trat)[col_resp].count().mean()
                            medias = df.groupby(col_trat)[col_resp].mean()
                            n_trats = len(medias)
                            
                            with t2:
                                df_tukey = tukey_manual_preciso(medias, res['mse'], res['df_resid'], reps, n_trats)
                                st.dataframe(df_tukey.style.format({"Media": "{:.2f}"}))
                                st.caption(explaining_ranking(df_tukey, "Tukey"))
                            
                            with t3:
                                df_sk = scott_knott(medias, res['mse'], res['df_resid'], reps)
                                st.dataframe(df_sk.style.format({"Media": "{:.2f}"}))
                                st.caption(explaining_ranking(df_sk, "Scott-Knott"))
                            
                            # --- NOVIDADE v6.3: Gráficos de Tukey E Scott-Knott ---
                            with t4:
                                # Gráfico Tukey
                                st.markdown("##### Ranking Tukey")
                                df_plot_tukey = df_tukey.reset_index().rename(columns={'index': col_trat})
                                fig_tukey = px.bar(df_plot_tukey, x=col_trat, y='Media', text='Letras', title=f"Ranking {col_resp} (Tukey)")
                                st.plotly_chart(fig_tukey, use_container_width=True)
                                
                                st.markdown("---")
                                
                                # Gráfico Scott-Knott
                                st.markdown("##### Ranking Scott-Knott")
                                df_plot_sk = df_sk.reset_index().rename(columns={'index': col_trat})
                                # Nota: Scott-Knott usa coluna 'Grupo', Tukey usa 'Letras'
                                fig_sk = px.bar(df_plot_sk, x=col_trat, y='Media', text='Grupo', title=f"Ranking {col_resp} (Scott-Knott)")
                                fig_sk.update_traces(marker_color='#2E86C1') # Cor diferente para diferenciar
                                st.plotly_chart(fig_sk, use_container_width=True)
                        else:
                            st.warning("ANOVA não significativa. Médias não diferem.")

                    else: # CONJUNTA
                        razao, mses, mses_detalhe = calcular_homogeneidade(df, col_trat, col_resp, col_local, col_bloco, delineamento)
                        res_conj = rodar_analise_conjunta(df, col_trat, col_resp, col_local, delineamento, col_bloco)
                        
                        st.subheader("Métricas Gerais")
                        c_m1, c_m2, c_m3 = st.columns(3)
                        media_geral = df[col_resp].mean()
                        c_m1.metric("Média Geral", f"{media_geral:.2f}")
                        
                        cv_conjunto = (np.sqrt(res_conj['mse']) / media_geral) * 100
                        class_cv_conj = classificar_cv(cv_conjunto)
                        c_m2.metric("CV Conjunto (%)", f"{cv_conjunto:.2f}%", delta=class_cv_conj, delta_color="off", help="Classificação conforme Pimentel-Gomes (2009).")
                        
                        if razao:
                            c_m3.metric("Razão Máx/Mín MSE", f"{razao:,.2f}", help="Razão entre o Maior e o Menor Erro Residual.")
                            if razao < 7: st.success("Variâncias Homogêneas (OK)")
                            else: st.error("Variâncias Heterogêneas (>7)")
                        
                        st.markdown("---")
                        st.subheader("Quadro da ANOVA Conjunta")
                        anova_conj_formatada = formatar_tabela_anova(res_conj['anova'])
                        st.dataframe(anova_conj_formatada.style.format({
                            "SQ": "{:.4f}", "GL": "{:.0f}", "QM": "{:.4f}", "Fcalc": "{:.4f}", "P-valor": "{:.4f}"
                        }))
                        st.caption("_Legenda: *** (P<0.001); ** (P<0.01); * (P<0.05); ns (Não Significativo)_")
                        
                        p_int = res_conj['p_interacao']
                        tem_interacao = p_int < 0.05
                        
                        if tem_interacao: st.error(f"⚠️ **Interação Significativa (P={p_int:.4f})**")
                        else: st.success(f"✅ **Interação Não Significativa (P={p_int:.4f})**")
                            
                        locais_unicos = sorted(df[col_local].unique())
                        abas = st.tabs(["📊 Média Geral"] + [f"📍 {loc}" for loc in locais_unicos] + ["📈 Gráfico Interação"])
                        
                        with abas[0]:
                            medias_geral = df.groupby(col_trat)[col_resp].mean()
                            reps_geral = df.groupby(col_trat)[col_resp].count().mean() 
                            df_sk_geral = scott_knott(medias_geral, res_conj['mse'], res_conj['df_resid'], reps_geral)
                            st.dataframe(df_sk_geral.style.format({"Media": "{:.2f}"}))
                            
                            # Correção de nome de coluna para o gráfico Conjunto também
                            df_plot_geral = df_sk_geral.reset_index().rename(columns={'index': col_trat})
                            fig_g = px.bar(df_plot_geral, x=col_trat, y='Media', text='Grupo', title=f"Média Geral {col_resp}")
                            st.plotly_chart(fig_g, use_container_width=True)

                        for k, loc in enumerate(locais_unicos):
                            with abas[k+1]:
                                df_loc = df[df[col_local] == loc]
                                res_loc = rodar_analise_individual(df_loc, col_trat, col_resp, delineamento, col_bloco)
                                if res_loc['p_val'] < 0.05:
                                    medias_loc = df_loc.groupby(col_trat)[col_resp].mean()
                                    reps_loc = df_loc.groupby(col_trat)[col_resp].count().mean()
                                    
                                    n_trats_loc = len(medias_loc)
                                    df_tukey_loc = tukey_manual_preciso(medias_loc, res_loc['mse'], res_loc['df_resid'], reps_loc, n_trats_loc)
                                    
                                    st.dataframe(df_tukey_loc.style.format({"Media": "{:.2f}"}))
                                    
                                    df_plot_loc = df_tukey_loc.reset_index().rename(columns={'index': col_trat})
                                    fig_l = px.bar(df_plot_loc, x=col_trat, y='Media', text='Letras', title=f"Ranking {col_resp} em {loc}")
                                    st.plotly_chart(fig_l, use_container_width=True)
                                else:
                                    st.warning(f"Sem diferença significativa em {loc}.")

                        with abas[-1]:
                            df_inter = df.groupby([col_trat, col_local])[col_resp].mean().reset_index()
                            fig_int = px.line(df_inter, x=col_local, y=col_resp, color=col_trat, markers=True, title=f"Interação GxE: {col_resp}")
                            st.plotly_chart(fig_int, use_container_width=True)

else:
    st.info("👈 Faça upload do arquivo para começar.")
