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

# --- INICIALIZAÇÃO DO ESTADO (MEMÓRIA) ---
if 'transformacoes' not in st.session_state:
    st.session_state['transformacoes'] = {} 
if 'processando' not in st.session_state:
    st.session_state['processando'] = False

def get_transformacao_atual(col_nome):
    return st.session_state['transformacoes'].get(col_nome, "Nenhuma")

def set_transformacao(col_nome, tipo):
    st.session_state['transformacoes'][col_nome] = tipo
    key_np = f"show_np_{col_nome}"
    if key_np in st.session_state:
        st.session_state[key_np] = False

def reset_analise():
    st.session_state['processando'] = False

def log_message(mensagem):
    if EXIBIR_LOGS:
        print(mensagem)

# --- UTILITÁRIOS E FORMATAÇÃO (NOVO v6.38) ---
def get_letra_segura(n):
    try:
        ciclo = int(n) // 26
        letra_idx = int(n) % 26
        letra = chr(97 + letra_idx) 
        if ciclo == 0: return letra
        else: return f"{letra}{ciclo}"
    except:
        return "?"

def formatar_numero(valor, decimais=2):
    """
    Formatação Híbrida:
    - Se o valor for muito pequeno (< 0.001), usa notação científica.
    - Caso contrário, usa casas decimais fixas.
    """
    try:
        v = float(valor)
        if pd.isna(v): return "-"
        if v == 0: return f"{0:.{decimais}f}"
        
        # Se for menor que 0.001 e não for zero, usa notação científica
        if abs(v) < 0.001:
            return f"{v:.2e}" # Ex: 3.45e-05
        else:
            return f"{v:.{decimais}f}"
    except:
        return str(valor)

def formatar_tabela_anova(anova_df):
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
    
    # Aplica a formatação híbrida nas colunas numéricas para exibição
    cols_numericas = ['SQ', 'QM', 'Fcalc', 'P-valor']
    for col in cols_numericas:
        if col in df.columns:
            # Usa 4 casas decimais como base para a tabela
            df[col] = df[col].apply(lambda x: formatar_numero(x, decimais=4))
            
    return df

def classificar_cv(cv):
    if cv < 10: return "🟢 Baixo (Alta Precisão)"
    elif cv < 20: return "🟡 Médio (Boa Precisão)"
    elif cv < 30: return "🟠 Alto (Baixa Precisão)"
    else: return "🔴 Muito Alto (Inadequado)"

# --- FUNÇÕES DE MÉTRICAS E INTERPRETAÇÃO ---
def calcular_metricas_extras(anova_df, modelo, col_trat):
    """Calcula métricas e define classes para verificação de alertas."""
    metrics = {
        'rmse': 0.0, 'r2': 0.0, 'acuracia': 0.0, 'h2': 0.0,
        'r2_class': "", 'ac_class': "N/A", 'h2_class': "N/A"
    }
    
    try:
        metrics['rmse'] = np.sqrt(modelo.mse_resid)
        metrics['r2'] = modelo.rsquared
        
        if metrics['r2'] >= 0.50: metrics['r2_class'] = "OK"
        else: metrics['r2_class'] = "🔴"

        # Tenta buscar Fcalc numérico na ANOVA original (antes da formatação de string)
        # Como passamos a anova formatada (strings) para essa função, precisamos recalcular ou tratar
        # Idealmente, recalcular F simples aqui para garantir
        mse = modelo.mse_resid
        
        # Recaptura o F do modelo diretamente para evitar erro de string
        f_calc = 0
        # Estratégia de fallback: pegar o maior F da tabela se não achar o tratamento exato
        # Mas vamos tentar achar o tratamento
        for idx in anova_df.index:
            if col_trat in idx and ":" not in idx:
                # Tenta converter de volta se for string ou pega do modelo
                try:
                    val = anova_df.loc[idx, "Fcalc"]
                    f_calc = float(val) if val != "-" else 0
                except:
                    f_calc = 0
                break
        
        if f_calc <= 1:
            metrics['acuracia'] = 0.0
            metrics['h2'] = 0.0
            metrics['ac_class'] = "🔴"
            metrics['h2_class'] = "🔴"
        else:
            metrics['acuracia'] = np.sqrt(1 - (1/f_calc))
            metrics['h2'] = 1 - (1/f_calc)
            
            if metrics['acuracia'] >= 0.50: metrics['ac_class'] = "OK"
            else: metrics['ac_class'] = "🔴"
            
            if metrics['h2'] >= 0.50: metrics['h2_class'] = "OK"
            else: metrics['h2_class'] = "🔴"
            
    except:
        metrics['ac_class'] = "Erro"
        metrics['h2_class'] = "Erro"
        
    return metrics

def gerar_relatorio_metricas(anova_df, modelo, col_trat, media_real, p_valor, razao_mse=None):
    """Gera texto explicativo em lista."""
    rmse = np.sqrt(modelo.mse_resid)
    r2 = modelo.rsquared
    
    # 1. ANOVA STATUS
    if p_valor < 0.05:
        sig_txt = "🟢 Significativo (Há diferença estatística entre tratamentos)."
    else:
        sig_txt = "🔴 Não Significativo (Médias estatisticamente iguais)."

    # 2. R2
    if r2 >= 0.90: r2_txt = "🟢 O modelo é excelente, explicando quase toda a variação."
    elif r2 >= 0.70: r2_txt = "🟢 O modelo tem bom ajuste aos dados."
    elif r2 >= 0.50: r2_txt = "🟡 Ajuste regular. Há muita variação não explicada."
    else: r2_txt = "🔴 Baixo ajuste. O modelo explica pouco o fenômeno (⚠️ Atenção)."

    # 3. CV
    cv_val = (rmse / media_real) * 100
    if cv_val < 10: cv_txt = "🟢 Baixo (Alta Precisão Experimental)."
    elif cv_val < 20: cv_txt = "🟡 Médio (Boa Precisão)."
    elif cv_val < 30: cv_txt = "🟠 Alto (Baixa Precisão)."
    else: cv_txt = "🔴 Muito Alto (Dados muito dispersos) (⚠️ Atenção)."

    # 4. ACURÁCIA & H2
    try:
        f_calc = 0
        for idx in anova_df.index:
            if col_trat in idx and ":" not in idx:
                try:
                    val = anova_df.loc[idx, "Fcalc"]
                    f_calc = float(val) if val != "-" else 0
                except: f_calc = 0
                break
        
        if f_calc <= 1:
            acuracia = 0.0
            herdabilidade = 0.0
            ac_txt = "🔴 Crítico: Variação genética não detectada (F < 1). Seleção ineficaz (⚠️ Atenção)."
            h2_txt = "🔴 A variância ambiental superou a genética (⚠️ Atenção)."
        else:
            acuracia = np.sqrt(1 - (1/f_calc))
            herdabilidade = 1 - (1/f_calc)
            
            if acuracia >= 0.90: ac_txt = "🟢 Muito Alta: Excelente confiabilidade para selecionar genótipos."
            elif acuracia >= 0.70: ac_txt = "🟢 Alta: Boa segurança na seleção."
            elif acuracia >= 0.50: ac_txt = "🟡 Moderada: Seleção requer cautela."
            else: ac_txt = "🔴 Baixa: Pouca confiança para selecionar (⚠️ Atenção)."
            
            if herdabilidade >= 0.80: h2_txt = "🟢 Alta magnitude (forte controle genético)."
            elif herdabilidade >= 0.50: h2_txt = "🟡 Média magnitude."
            else: h2_txt = "🔴 Baixa magnitude (forte influência ambiental) (⚠️ Atenção)."
            
    except:
        acuracia, herdabilidade = 0, 0
        ac_txt = "⚠️ Não Estimável: Parâmetros estatísticos insuficientes."
        h2_txt = "⚠️ Não Estimável: Parâmetros estatísticos insuficientes."

    # CONSTRUÇÃO DO TEXTO COM FORMATAÇÃO HÍBRIDA
    txt_media = formatar_numero(media_real)
    txt_cv = formatar_numero(cv_val)
    txt_ac = formatar_numero(acuracia)
    txt_h2 = formatar_numero(herdabilidade)
    txt_r2 = formatar_numero(r2)
    txt_rmse = formatar_numero(rmse)
    txt_p = formatar_numero(p_valor, decimais=4)

    texto = ""
    texto += f"- 📊 **Média Geral:** `{txt_media}` — Valor central dos dados.\n"
    texto += f"- ⚡ **CV (%):** `{txt_cv}%` — {cv_txt}\n"
    texto += f"- 🎯 **Acurácia Seletiva:** `{txt_ac}` — {ac_txt}\n"
    texto += f"- 🧬 **Herdabilidade ($h^2$):** `{txt_h2}` — {h2_txt}\n"
    texto += f"- 📉 **Coeficiente de Determinação (R²):** `{txt_r2}` — {r2_txt}\n"
    texto += f"- 📏 **Raiz do Erro Quadrático Médio (RMSE):** `{txt_rmse}` — Erro médio absoluto na unidade da variável.\n"
    
    if razao_mse:
        razao_txt = "🟢 Homogêneo (Confiável)" if razao_mse < 7 else "🔴 Heterogêneo (⚠️ Atenção)"
        txt_razao = formatar_numero(razao_mse)
        texto += f"- ⚖️ **Razão de Erro Quadrático Médio (MSE):** `{txt_razao}` — {razao_txt}\n"

    texto += f"- 🔍 **ANOVA:** `P={txt_p}` — {sig_txt}\n"

    return texto

# --- DIAGNÓSTICO E TABELAS ---
def gerar_tabela_diagnostico(p_shapiro, p_bartlett=None):
    if p_shapiro < 0.05:
        cond_sw, conc_sw = "$P < 0.05$", "Rejeita $H_0$. **NÃO Normal** ⚠️"
    else:
        cond_sw, conc_sw = "$P \ge 0.05$", "Não Rejeita $H_0$. **Normal** ✅"
    
    txt_shap = formatar_numero(p_shapiro, 4)
    tabela = f"| Teste | P-valor | Condição | Conclusão |\n| :--- | :--- | :--- | :--- |\n"
    tabela += f"| **Shapiro-Wilk** | ${txt_shap}$ | {cond_sw} | {conc_sw} |\n"
    
    if p_bartlett is not None:
        if p_bartlett < 0.05:
            cond_bt, conc_bt = "$P < 0.05$", "Rejeita $H_0$. **NÃO Homogêneo** ⚠️"
        else:
            cond_bt, conc_bt = "$P \ge 0.05$", "Não Rejeita $H_0$. **Homogêneo** ✅"
        txt_bart = formatar_numero(p_bartlett, 4)
        tabela += f"| **Bartlett** | ${txt_bart}$ | {cond_bt} | {conc_bt} |\n"
    
    return tabela

def aplicar_transformacao(df, col_resp, tipo_transformacao):
    """Aplica transformação e retorna df novo e nome da coluna nova."""
    nova_col = col_resp
    df_copy = df.copy()
    
    if tipo_transformacao == "Log10":
        nova_col = f"{col_resp}_Log"
        df_copy[nova_col] = np.log10(df_copy[col_resp].where(df_copy[col_resp] > 0, 1e-10))
    elif tipo_transformacao == "Raiz Quadrada (SQRT)":
        nova_col = f"{col_resp}_Sqrt"
        df_copy[nova_col] = np.sqrt(df_copy[col_resp].where(df_copy[col_resp] >= 0, 0))
        
    return df_copy, nova_col

# --- MOTORES ESTATÍSTICOS ---

def calcular_nao_parametrico(df, col_trat, col_resp, delineamento, col_bloco=None):
    try:
        df_clean = df.dropna(subset=[col_resp])
        
        if delineamento == 'DIC':
            grupos = [g[col_resp].values for _, g in df_clean.groupby(col_trat)]
            if len(grupos) < 2: return "Erro", None
            stat, p = stats.kruskal(*grupos)
            return "Kruskal-Wallis", p
        
        elif delineamento == 'DBC':
            try:
                pivot = df_clean.pivot_table(index=col_bloco, columns=col_trat, values=col_resp)
                if pivot.isnull().values.any(): return "Inviável (Dados Faltantes)", None
                stat, p = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])
                return "Friedman", p
            except Exception as e: return f"Erro ({str(e)})", None
    except: return "Erro", None
    return "N/A", None

def tukey_manual_preciso(medias, mse, df_resid, n_reps, n_trats):
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
            P.remove(v)
            X.add(v)
            
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
    res['shapiro'] = stats.shapiro(modelo.resid)
    grupos = [g[col_resp].values for _, g in df.groupby(col_trat)]
    res['bartlett'] = stats.bartlett(*grupos)
    
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
arquivo = st.sidebar.file_uploader("Upload CSV ou Excel", type=["xlsx", "csv"], on_change=reset_analise)

if arquivo:
    if arquivo.name.endswith('.csv'): df = pd.read_csv(arquivo)
    else: df = pd.read_excel(arquivo)
    colunas = df.columns.tolist()
    
    st.sidebar.success(f"Carregado: {len(df)} linhas")
    st.sidebar.markdown("---")
    
    # ATENÇÃO: TODOS OS INPUTS AGORA TEM O CALLBACK DE RESET
    tipo_del = st.sidebar.radio("Delineamento:", ("DIC", "DBC"), on_change=reset_analise)
    delineamento = "DIC" if "DIC" in tipo_del else "DBC"
    
    col_trat = st.sidebar.selectbox("Tratamentos (Genótipos)", colunas, on_change=reset_analise)
    
    OPCAO_PADRAO = "Local Único (Análise Individual)" 
    col_local = st.sidebar.selectbox("Local/Ambiente (Opcional)", [OPCAO_PADRAO] + [c for c in colunas if c != col_trat], on_change=reset_analise)
    
    col_bloco = None
    if delineamento == "DBC":
        col_bloco = st.sidebar.selectbox("Blocos", [c for c in colunas if c not in [col_trat, col_local]], on_change=reset_analise)

    cols_ocupadas = [col_trat, col_local]
    if col_bloco: cols_ocupadas.append(col_bloco)
    
    lista_resps = st.sidebar.multiselect("Variáveis Resposta (Selecione 1 ou mais)", [c for c in colunas if c not in cols_ocupadas], on_change=reset_analise)

    modo_analise = "INDIVIDUAL"
    if col_local != OPCAO_PADRAO:
        n_locais = len(df[col_local].unique())
        if n_locais > 1:
            modo_analise = "CONJUNTA"
            st.sidebar.info(f"🌍 Modo Conjunta Ativado! ({n_locais} locais detectados)")
        else:
            st.sidebar.warning("⚠️ Coluna de Local selecionada, mas há apenas 1 local. Rodando modo Individual.")

    # --- BOTÃO PRINCIPAL ---
    if st.sidebar.button("🚀 Processar Estatística"):
        st.session_state['processando'] = True

    if st.session_state['processando']:
        if not lista_resps:
            st.error("⚠️ Por favor, selecione pelo menos uma Variável Resposta.")
        else:
            st.markdown(f"### 📋 Resultados: {len(lista_resps)} variáveis processadas")
            
            for i, col_resp_original in enumerate(lista_resps):
                with st.expander(f"📊 Variável: {col_resp_original}", expanded=(i==0)):
                    
                    # TRANSFORMAÇÃO INDIVIDUAL
                    transf_atual = get_transformacao_atual(col_resp_original)
                    df_proc, col_resp = aplicar_transformacao(df.copy(), col_resp_original, transf_atual)
                    
                    if transf_atual != "Nenhuma":
                        st.info(f"🔄 **Transformação Ativa:** {transf_atual} (Coluna: {col_resp})")
                    
                    st.markdown(f"### Análise de: **{col_resp}**")
                    
                    # --- EXECUÇÃO DA ANÁLISE ---
                    p_shap, p_bart = 1.0, 1.0 
                    res_analysis = {}
                    
                    if modo_analise == "INDIVIDUAL":
                        res = rodar_analise_individual(df_proc, col_trat, col_resp, delineamento, col_bloco)
                        res_analysis = res
                        p_shap, p_bart = res['shapiro'][1], res['bartlett'][1]
                        
                        # ANOVA formatada primeiro para pegar dados
                        anova_tab = formatar_tabela_anova(res['anova'])
                        
                        st.markdown("#### 📝 Métricas Estatísticas")
                        txt_metrics = gerar_relatorio_metricas(anova_tab, res['modelo'], col_trat, df_proc[col_resp].mean(), res['p_val'])
                        st.markdown(txt_metrics)
                        
                        # --- CAIXAS DE ALERTA (INDIVIDUAL) ---
                        extras = calcular_metricas_extras(anova_tab, res['modelo'], col_trat)
                        cv_val = (np.sqrt(res['mse'])/df_proc[col_resp].mean())*100
                        
                        if cv_val > 20: st.error(f"🚨 CV Crítico: {cv_val:.2f}% (>20%). Dados muito dispersos.")
                        if "🔴" in extras['ac_class']: st.error("🚨 Acurácia Baixa: Seleção genética pouco confiável.")
                        if "🔴" in extras['h2_class']: st.error("🚨 Herdabilidade Baixa: Forte influência ambiental.")
                        if "🔴" in extras['r2_class']: st.error("🚨 R² Baixo: O modelo não explica bem os dados.")
                        
                        if res['p_val'] >= 0.05:
                            st.error("🚨 ANOVA Não Significativa: Não há diferença estatística entre os tratamentos.")
                        
                        sig = res['p_val'] < 0.05
                        
                        t1, t2, t3, t4 = st.tabs(["📋 ANOVA & Diagnóstico", "📦 Teste de Tukey", "📦 Teste de Scott-Knott", "📈 Gráficos"])
                        
                        with t1:
                            st.markdown("### 📊 Análise de Variância (ANOVA)")
                            st.dataframe(anova_tab)
                            st.caption("_Legenda: *** (P<0.001); ** (P<0.01); * (P<0.05); ns (Não Significativo)_")
                            st.markdown("---")
                            st.markdown("#### 🩺 Diagnóstico dos Pressupostos")
                            st.markdown(gerar_tabela_diagnostico(p_shap, p_bart))
                            if p_shap < 0.05 or (p_bart is not None and p_bart < 0.05):
                                st.warning("⚠️ Violação de Pressupostos detectada! Verifique as opções no rodapé.")
                                
                        if sig:
                            reps = df_proc.groupby(col_trat)[col_resp].count().mean()
                            medias = df_proc.groupby(col_trat)[col_resp].mean()
                            n_trats = len(medias)
                            with t2:
                                df_tukey = tukey_manual_preciso(medias, res['mse'], res['df_resid'], reps, n_trats)
                                st.dataframe(df_tukey.style.format({"Media": "{:.2f}"}))
                                st.caption(explaining_ranking(df_tukey, "Tukey"))
                            with t3:
                                df_sk = scott_knott(medias, res['mse'], res['df_resid'], reps)
                                st.dataframe(df_sk.style.format({"Media": "{:.2f}"}))
                                st.caption(explaining_ranking(df_sk, "Scott-Knott"))
                            with t4:
                                f1 = px.bar(df_tukey.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Letras', title=f"Tukey: {col_resp}")
                                st.plotly_chart(f1, use_container_width=True)
                                st.markdown("---")
                                f2 = px.bar(df_sk.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupo', title=f"Scott-Knott: {col_resp}")
                                f2.update_traces(marker_color='#2E86C1')
                                st.plotly_chart(f2, use_container_width=True)
                        else:
                            st.warning("ANOVA não significativa.")

                    else: # CONJUNTA
                        res_conj = rodar_analise_conjunta(df_proc, col_trat, col_resp, col_local, delineamento, col_bloco)
                        res_analysis = res_conj
                        p_shap, p_bart = res_conj['shapiro'][1], res_conj['bartlett'][1]
                        razao, _, _ = calcular_homogeneidade(df_proc, col_trat, col_resp, col_local, col_bloco, delineamento)
                        
                        anova_tab = formatar_tabela_anova(res_conj['anova'])
                        
                        st.markdown("#### 📝 Métricas Estatísticas")
                        txt_metrics = gerar_relatorio_metricas(anova_tab, res_conj['modelo'], col_trat, df_proc[col_resp].mean(), res_conj['p_trat'], razao)
                        st.markdown(txt_metrics)
                        
                        # --- CAIXAS DE ALERTA (CONJUNTA) ---
                        extras = calcular_metricas_extras(anova_tab, res_conj['modelo'], col_trat)
                        cv_conj = (np.sqrt(res_conj['mse']) / df_proc[col_resp].mean()) * 100
                        
                        if cv_conj > 20: st.error(f"🚨 CV Crítico: {cv_conj:.2f}% (>20%). Dados muito dispersos.")
                        if "🔴" in extras['ac_class']: st.error("🚨 Acurácia Baixa.")
                        if "🔴" in extras['h2_class']: st.error("🚨 Herdabilidade Baixa.")
                        if "🔴" in extras['r2_class']: st.error("🚨 R² Baixo.")
                        if razao and razao > 7: st.error(f"🚨 Variâncias Heterogêneas (Razão MSE: {razao:.2f} > 7). Isso invalida a ANOVA conjunta, mesmo que o resultado seja significativo.")
                        
                        if res_conj['p_trat'] >= 0.05:
                            st.error("🚨 ANOVA Não Significativa: Não há diferença estatística entre os tratamentos.")

                        st.markdown("### 📊 Análise de Variância (ANOVA)")
                        st.dataframe(anova_tab)
                        st.caption("_Legenda: *** (P<0.001); ** (P<0.01); * (P<0.05); ns (Não Significativo)_")
                        
                        p_int = res_conj.get('p_interacao', 1.0)
                        if p_int < 0.05: st.error(f"⚠️ **Interação Significativa (P={p_int:.4f})**")
                        else: st.success(f"✅ **Interação Não Significativa (P={p_int:.4f})**")
                        
                        st.markdown("---")
                        st.markdown("#### 🩺 Diagnóstico (Conjunto)")
                        st.markdown(gerar_tabela_diagnostico(p_shap, p_bart))
                        if p_shap < 0.05 or (p_bart is not None and p_bart < 0.05): st.warning("⚠️ Violação de Pressupostos detectada! Verifique as opções no rodapé.")
                        if p_int < 0.05: st.info("Desdobramento disponível nas abas abaixo (omitido para brevidade visual nesta etapa).")
                        
                        locais_unicos = sorted(df_proc[col_local].unique())
                        abas = st.tabs(["📊 Média Geral"] + [f"📍 {loc}" for loc in locais_unicos] + ["📈 Gráfico Interação"])
                        
                        with abas[0]:
                            medias_geral = df_proc.groupby(col_trat)[col_resp].mean()
                            reps_geral = df_proc.groupby(col_trat)[col_resp].count().mean() 
                            df_sk_geral = scott_knott(medias_geral, res_conj['mse'], res_conj['df_resid'], reps_geral)
                            st.dataframe(df_sk_geral.style.format({"Media": "{:.2f}"}))
                            f_g = px.bar(df_sk_geral.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupo', title=f"Média Geral {col_resp}")
                            st.plotly_chart(f_g, use_container_width=True)
                            
                        for k, loc in enumerate(locais_unicos):
                            with abas[k+1]:
                                df_loc = df_proc[df_proc[col_local] == loc]
                                res_loc = rodar_analise_individual(df_loc, col_trat, col_resp, delineamento, col_bloco)
                                if res_loc['p_val'] < 0.05:
                                    medias_loc = df_loc.groupby(col_trat)[col_resp].mean()
                                    reps_loc = df_loc.groupby(col_trat)[col_resp].count().mean()
                                    n_trats_loc = len(medias_loc)
                                    
                                    df_tukey_loc = tukey_manual_preciso(medias_loc, res_loc['mse'], res_loc['df_resid'], reps_loc, n_trats_loc)
                                    df_sk_loc = scott_knott(medias_loc, res_loc['mse'], res_loc['df_resid'], reps_loc)
                                    
                                    sub_t1, sub_t2, sub_t3 = st.tabs(["📦 Teste de Tukey", "📦 Teste de Scott-Knott", "📈 Gráficos"])
                                    
                                    with sub_t1:
                                        st.dataframe(df_tukey_loc.style.format({"Media": "{:.2f}"}))
                                        st.caption(explaining_ranking(df_tukey_loc, "Tukey"))
                                        
                                    with sub_t2:
                                        st.dataframe(df_sk_loc.style.format({"Media": "{:.2f}"}))
                                        st.caption(explaining_ranking(df_sk_loc, "Scott-Knott"))
                                        
                                    with sub_t3:
                                        f_l = px.bar(df_tukey_loc.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Letras', title=f"Ranking {col_resp} em {loc} (Tukey)")
                                        st.plotly_chart(f_l, use_container_width=True)
                                        
                                        f_s = px.bar(df_sk_loc.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupo', title=f"Ranking {col_resp} em {loc} (Scott-Knott)")
                                        f_s.update_traces(marker_color='#2E86C1')
                                        st.plotly_chart(f_s, use_container_width=True)
                                else:
                                    st.warning(f"Sem diferença significativa em {loc}.")
                                    
                        with abas[-1]:
                            df_inter = df_proc.groupby([col_trat, col_local])[col_resp].mean().reset_index()
                            f_i = px.line(df_inter, x=col_local, y=col_resp, color=col_trat, markers=True, title=f"Interação GxE: {col_resp}")
                            st.plotly_chart(f_i, use_container_width=True)

                    # --- LÓGICA DE DECISÃO FINAL ---
                    if transf_atual != "Nenhuma" and p_shap >= 0.05 and (p_bart is None or p_bart >= 0.05):
                        st.markdown("---")
                        st.markdown("### 🛡️ Solução Final: Análise Paramétrica (ANOVA)")
                        st.success(f"✅ **Transformação Eficaz!** Com **{transf_atual}**, os dados atendem aos pressupostos. Esta ANOVA é válida.")
                        if st.button("Voltar ao Original", key=f"reset_success_{col_resp_original}"):
                            set_transformacao(col_resp_original, "Nenhuma")
                            st.rerun()

                    elif p_shap < 0.05 or (p_bart is not None and p_bart < 0.05):
                        st.markdown("---")
                        st.error("🚨 ALERTA ESTATÍSTICO GRAVE: ANOVA INVÁLIDA")
                        st.markdown("""
                        Como os dados não seguem a **Normalidade** e/ou **Homogeneidade**, a média e o desvio padrão perdem o sentido.
                        **NÃO USE A ANOVA (Teste F)** para tomar decisões, pois ela pode apresentar resultados falsos (falso positivo ou negativo).
                        
                        **O que fazer?**
                        1. Tente realizar a **Transformação dos Dados** nas opções abaixo.
                        2. Se o problema persistir, analise cada local individualmente usando testes Não-Paramétricos.
                        """)
                        
                        if transf_atual == "Nenhuma":
                            col_btn1, col_btn2 = st.columns([1, 4])
                            with col_btn1:
                                if st.button("🧪 Tentar Log10", key=f"btn_log_{col_resp_original}"):
                                    set_transformacao(col_resp_original, "Log10")
                                    st.rerun()
                            with col_btn2:
                                st.caption("Clique para aplicar transformação Logarítmica apenas nesta variável.")

                        elif transf_atual == "Log10":
                            st.warning(f"A transformação **Log10** não resolveu o problema.")
                            col_btn1, col_btn2 = st.columns([1, 4])
                            with col_btn1:
                                if st.button("🌱 Tentar Raiz Quadrada", key=f"btn_sqrt_{col_resp_original}"):
                                    set_transformacao(col_resp_original, "Raiz Quadrada (SQRT)")
                                    st.rerun()
                            if st.button("Voltar ao Original", key=f"reset_log_{col_resp_original}"):
                                set_transformacao(col_resp_original, "Nenhuma")
                                st.rerun()

                        elif transf_atual == "Raiz Quadrada (SQRT)":
                            st.warning(f"A transformação **Raiz Quadrada** também não resolveu.")
                            st.markdown("### 🛡️ Solução Final: Estatística Não-Paramétrica")
                            
                            key_np = f"show_np_{col_resp_original}"
                            if key_np not in st.session_state: st.session_state[key_np] = False
                            
                            if not st.session_state[key_np]:
                                if st.button("🛡️ Rodar Estatística Não-Paramétrica", key=f"btn_run_np_{col_resp_original}"):
                                    st.session_state[key_np] = True
                                    st.rerun()
                            else:
                                nome_np, p_np = calcular_nao_parametrico(df_proc, col_trat, col_resp, delineamento, col_bloco)
                                if p_np is not None:
                                    st.success(f"Resultado do Teste de **{nome_np}**:")
                                    sig_np = "Significativo (Diferença Real)" if p_np < 0.05 else "Não Significativo (Iguais)"
                                    st.metric(label="P-valor Não-Paramétrico", value=f"{p_np:.4f}", delta=sig_np, delta_color="inverse" if p_np < 0.05 else "normal")
                                else:
                                    st.error("Não foi possível calcular o teste não-paramétrico (verifique dados faltantes ou delineamento).")
                                
                                if st.button("Ocultar Resultado", key=f"btn_hide_np_{col_resp_original}"):
                                    st.session_state[key_np] = False
                                    st.rerun()
                            
                            if st.button("Voltar ao Original", key=f"reset_sqrt_{col_resp_original}"):
                                set_transformacao(col_resp_original, "Nenhuma")
                                st.rerun()

else:
    st.info("👈 Faça upload do arquivo para começar.")
