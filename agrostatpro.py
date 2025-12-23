# ==============================================================================
# 📂 BLOCO 01: Imports, Configuração de Logs e Estado (Memória)
# ==============================================================================
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
# ==============================================================================
# 🏁 FIM DO BLOCO 01
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 02: Limpeza, Formatação e Utilitários Básicos (CORRIGIDO)
# ==============================================================================
def limpar_e_converter_dados(df, col_resp):
    """
    Converte coluna para float, tratando vírgulas (PT-BR) e erros.
    """
    serie = df[col_resp].copy()
    
    # 1. Se for objeto/string, tenta trocar vírgula por ponto
    if serie.dtype == 'object':
        serie = serie.astype(str).str.replace(',', '.', regex=False)
    
    # 2. Converte para numérico (Coerce transforma erros em NaN)
    serie = pd.to_numeric(serie, errors='coerce')
    
    return serie

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
        
        if abs(v) < 0.001:
            return f"{v:.2e}" 
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
        # Converte para string
        nome = str(idx)
        
        # LIMPEZA PROFUNDA (Remove artefatos do Patsy/Statsmodels)
        # Remove 'C(' do início
        nome = nome.replace('C(', '')
        # Remove ', Sum)' que apareceu por causa da correção estatística
        nome = nome.replace(', Sum)', '')
        # Remove qualquer parêntese solto restante
        nome = nome.replace(')', '')
        
        # Formatação Visual
        nome = nome.replace(':', ' x ')
        
        # Tradução
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
    
    cols_numericas = ['SQ', 'QM', 'Fcalc', 'P-valor']
    for col in cols_numericas:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: formatar_numero(x, decimais=4))
            
    return df

def classificar_cv(cv):
    if cv < 10: return "🟢 Baixo (Alta Precisão)"
    elif cv < 20: return "🟡 Médio (Boa Precisão)"
    elif cv < 30: return "🟠 Alto (Baixa Precisão)"
    else: return "🔴 Muito Alto (Inadequado)"
# ==============================================================================
# 🏁 FIM DO BLOCO 02
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 03: Cálculo de Métricas e Relatórios de Texto (COM RETORNO DE F)
# ==============================================================================
def calcular_metricas_extras(anova_df, modelo, col_trat):
    """Calcula métricas, define classes e retorna o valor F bruto para diagnóstico."""
    metrics = {
        'rmse': 0.0, 'r2': 0.0, 'acuracia': 0.0, 'h2': 0.0,
        'r2_class': "", 'ac_class': "N/A", 'h2_class': "N/A",
        'f_valor_bruto': 0.0 # Novo campo para diagnóstico no App
    }
    
    try:
        metrics['rmse'] = np.sqrt(modelo.mse_resid)
        metrics['r2'] = modelo.rsquared
        
        if metrics['r2'] >= 0.50: metrics['r2_class'] = "OK"
        else: metrics['r2_class'] = "🔴"

        # Tenta buscar Fcalc numérico
        f_calc = 0
        for idx in anova_df.index:
            # Limpeza extra para garantir match da string
            idx_clean = str(idx).replace("C(", "").replace(")", "")
            
            # Verifica se é a linha do tratamento (e não interação ou resíduo)
            if col_trat in idx_clean and ":" not in idx_clean: 
                try:
                    val = anova_df.loc[idx, "Fcalc"]
                    f_calc = float(val) if val != "-" else 0
                except:
                    f_calc = 0
                break
        
        # Salva o valor bruto para usar no aviso do App
        metrics['f_valor_bruto'] = f_calc

        # Lógica da Herdabilidade/Acurácia
        if pd.isna(f_calc) or f_calc <= 1:
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

    # 4. ACURÁCIA & H2 (Reutiliza lógica robusta visualmente)
    try:
        f_calc = 0
        for idx in anova_df.index:
            idx_clean = str(idx).replace("C(", "").replace(")", "")
            if col_trat in idx_clean and ":" not in idx_clean:
                try:
                    val = anova_df.loc[idx, "Fcalc"]
                    f_calc = float(val) if val != "-" else 0
                except: f_calc = 0
                break
        
        if f_calc <= 1:
            acuracia = 0.0
            herdabilidade = 0.0
            ac_txt = "🔴 Crítico: Variação genética não detectada (F <= 1). Seleção ineficaz (⚠️ Atenção)."
            h2_txt = "🔴 Zero: A variância ambiental superou a genética (⚠️ Atenção)."
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
    texto += f"- 🧬 **Herdabilidade (h²):** `{txt_h2}` — {h2_txt}\n"
    texto += f"- 📉 **Coeficiente de Determinação (R²):** `{txt_r2}` — {r2_txt}\n"
    texto += f"- 📏 **Raiz do Erro Quadrático Médio (RMSE):** `{txt_rmse}` — Erro médio absoluto na unidade da variável.\n"
    
    if razao_mse:
        razao_txt = "🟢 Homogêneo (Confiável)" if razao_mse < 7 else "🔴 Heterogêneo (⚠️ Atenção)"
        txt_razao = formatar_numero(razao_mse)
        texto += f"- ⚖️ **Razão de Erro Quadrático Médio (MSE):** `{txt_razao}` — {razao_txt}\n"

    texto += f"- 🔍 **ANOVA (Genótipos):** `P={txt_p}` — {sig_txt}\n"

    return texto
# ==============================================================================
# 🏁 FIM DO BLOCO 03
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 04: Diagnóstico Visual e Transformações (COM CACHE ⚡)
# ==============================================================================
def gerar_tabela_diagnostico(p_shapiro, p_bartlett=None, p_levene=None):
    """Gera tabela Markdown pura (Leve)."""
    tabela = "| Teste Estatístico | P-valor | Resultado | Conclusão |\n"
    tabela += "| :--- | :---: | :--- | :--- |\n"
    
    # 1. Shapiro
    if pd.isna(p_shapiro):
        p_txt, res_txt, conc_txt = "-", "Não Calculado", "Ignorado ⚪"
    elif p_shapiro < 0.05:
        p_txt, res_txt, conc_txt = f"{p_shapiro:.4f}", "P < 0.05", "Rejeita H0 (**NÃO Normal**) ⚠️"
    else:
        p_txt, res_txt, conc_txt = f"{p_shapiro:.4f}", "P >= 0.05", "Aceita H0 (**Normal**) ✅"
    tabela += f"| **Shapiro-Wilk** | {p_txt} | {res_txt} | {conc_txt} |\n"
    
    # 2. Bartlett
    if p_bartlett is not None:
        if pd.isna(p_bartlett):
            p_txt, res_txt, conc_txt = "-", "Não Calculado", "Ignorado ⚪"
        elif p_bartlett < 0.05:
            p_txt, res_txt, conc_txt = f"{p_bartlett:.4f}", "P < 0.05", "Rejeita H0 (**Heterogêneo**) ⚠️"
        else:
            p_txt, res_txt, conc_txt = f"{p_bartlett:.4f}", "P >= 0.05", "Aceita H0 (**Homogêneo**) ✅"
        tabela += f"| **Bartlett** | {p_txt} | {res_txt} | {conc_txt} |\n"

    # 3. Levene
    if p_levene is not None:
        if pd.isna(p_levene):
            p_txt, res_txt, conc_txt = "-", "Não Calculado", "Ignorado ⚪"
        elif p_levene < 0.05:
            p_txt, res_txt, conc_txt = f"{p_levene:.4f}", "P < 0.05", "Rejeita H0 (**Heterogêneo**) ⚠️"
        else:
            p_txt, res_txt, conc_txt = f"{p_levene:.4f}", "P >= 0.05", "Aceita H0 (**Homogêneo**) ✅"
        tabela += f"| **Levene** | {p_txt} | {res_txt} | {conc_txt} |\n"
    
    return tabela

# OTIMIZAÇÃO: Cache para não reprocessar a transformação toda vez que a tela atualiza
@st.cache_data(show_spinner=False)
def aplicar_transformacao(df, col_resp, tipo_transformacao):
    """Aplica transformação matemática nos dados (Cacheada)."""
    df_copy = df.copy()
    
    # LIMPEZA CRÍTICA
    df_copy[col_resp] = limpar_e_converter_dados(df_copy, col_resp)
    
    nova_col = col_resp
    if tipo_transformacao == "Log10":
        nova_col = f"{col_resp}_Log"
        df_copy[nova_col] = np.log10(df_copy[col_resp].where(df_copy[col_resp] > 0, 1e-10))
    elif tipo_transformacao == "Raiz Quadrada (SQRT)":
        nova_col = f"{col_resp}_Sqrt"
        df_copy[nova_col] = np.sqrt(df_copy[col_resp].where(df_copy[col_resp] >= 0, 0))
        
    return df_copy, nova_col
# ==============================================================================
# 🏁 FIM DO BLOCO 04
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 05: Motores Estatísticos II (Testes de Médias - COM CACHE ⚡)
# ==============================================================================

# OTIMIZAÇÃO: O cálculo do Tukey é pesado. Salvamos o resultado em cache.
@st.cache_data(show_spinner=False)
def tukey_manual_preciso(medias, mse, df_resid, r, n_trats):
    """Calcula Tukey (HSD) e retorna DataFrame pronto com letras."""
    # 1. Cálculo do Delta (DMS)
    q_critico = studentized_range.ppf(1 - 0.05, n_trats, df_resid)
    dms = q_critico * np.sqrt(mse / r)
    
    # 2. Ordenação
    medias_ord = medias.sort_values(ascending=False)
    nomes = medias_ord.index.tolist()
    vals = medias_ord.values
    
    # 3. Agrupamento por Letras (Algoritmo Ganancioso Otimizado)
    letras = {}
    for nome in nomes: letras[nome] = ""
    
    # Lógica de varredura
    cobriu_tudo = False
    idx_start = 0
    letra_atual_idx = 0
    
    while idx_start < len(vals):
        # Cria um grupo a partir de idx_start
        grupo_atual = [idx_start]
        referencia = vals[idx_start]
        
        # Tenta estender o grupo
        for i in range(idx_start + 1, len(vals)):
            diff = referencia - vals[i] # Como está decrescente, ref >= val
            if diff < dms: # Não difere
                grupo_atual.append(i)
            # Nota: Tukey permite sobreposição, então não paramos no primeiro fail em lógicas complexas,
            # mas para visualização simples, o agrupamento contíguo funciona bem para "barras".
            
        # Atribui a letra para todos do grupo
        letra_char = get_letra_segura(letra_atual_idx)
        for idx in grupo_atual:
            nome_trat = nomes[idx]
            # Evita duplicação da mesma letra no mesmo tratamento
            if letra_char not in letras[nome_trat]:
                letras[nome_trat] += letra_char
        
        # Avança para o próximo não coberto ou incrementa start
        letra_atual_idx += 1
        idx_start += 1 
            
    # Formata Saída
    df_res = pd.DataFrame({'Media': vals, 'Letras': [letras[n] for n in nomes]}, index=nomes)
    return df_res

# OTIMIZAÇÃO: Scott-Knott é recursivo e pesado. Cache essencial.
@st.cache_data(show_spinner=False)
def scott_knott(medias, mse, df_resid, r):
    """Algoritmo de Scott-Knott de agrupamento (Clusterização)."""
    medias_ord = medias.sort_values(ascending=False)
    valores = medias_ord.values
    nomes = medias_ord.index
    
    resultados = {nome: '' for nome in nomes}
    grupos_counter = 0

    def sk_recursive(start_idx, end_idx):
        nonlocal grupos_counter
        n = end_idx - start_idx
        if n <= 1:
            l = get_letra_segura(grupos_counter)
            resultados[nomes[start_idx]] = l
            grupos_counter += 1
            return

        # Busca melhor ponto de corte (Max B0)
        best_bo = -1
        best_k = -1
        
        subset = valores[start_idx:end_idx]
        
        for i in range(1, n): # Pontos de corte possíveis
            g1 = subset[:i]
            g2 = subset[i:]
            b0 = i * (np.mean(g1) - np.mean(subset))**2 + (n-i) * (np.mean(g2) - np.mean(subset))**2
            if b0 > best_bo:
                best_bo = b0
                best_k = i
                
        # Teste de Significância (Lambda / Qui-quadrado)
        sigma2 = mse / r
        lambda_val = (np.pi / (2 * (np.pi - 2))) * (best_bo / sigma2)
        
        # Valor crítico aproximado (Qui-quadrado)
        p_val = 1 - stats.chi2.cdf(lambda_val, df=1) # Aproximação usual
        
        if p_val > 0.05: # Não rejeita H0 -> Homogêneo
            l = get_letra_segura(grupos_counter)
            for k in range(start_idx, end_idx):
                resultados[nomes[k]] = l
            grupos_counter += 1
        else: # Rejeita H0 -> Heterogêneo -> Divide e conquista
            sk_recursive(start_idx, start_idx + best_k)
            sk_recursive(start_idx + best_k, end_idx)

    sk_recursive(0, len(valores))
    
    # Ajuste para garantir ordem alfabética a, b, c nos grupos formados (UX)
    # Re-mapeia as letras baseadas na média do grupo
    df_temp = pd.DataFrame({'Media': valores, 'LetraRaw': [resultados[n] for n in nomes]}, index=nomes)
    media_grupos = df_temp.groupby('LetraRaw')['Media'].mean().sort_values(ascending=False)
    
    mapa_final = {}
    for i, (letra_velha, _) in enumerate(media_grupos.items()):
        mapa_final[letra_velha] = get_letra_segura(i)
        
    df_temp['Grupo'] = df_temp['LetraRaw'].map(mapa_final)
    return df_temp[['Media', 'Grupo']]

def explaining_ranking(df, method="Tukey"):
    return f"Nota: Médias seguidas pela mesma letra/grupo não diferem estatisticamente ({method} 5%)."
# ==============================================================================
# 🏁 FIM DO BLOCO 05
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 06: Motores Estatísticos II (Algoritmos Complexos: Tukey e Scott-Knott)
# ==============================================================================
def tukey_manual_preciso(medias, mse, df_resid, n_reps, n_trats):
    if EXIBIR_LOGS: print(f"🧪 Iniciando Tukey Manual | n_trats: {n_trats} | GL Resíduo: {df_resid}")

    ep = np.sqrt(mse / n_reps)
    q_crit = studentized_range.ppf(0.95, n_trats, df_resid)
    hsd = q_crit * ep
    
    if EXIBIR_LOGS: print(f"📉 DMS (HSD) calculado: {hsd:.4f}")

    trats = medias.index.tolist()
    adj = {t: set() for t in trats}
    
    sorted_medias = medias.sort_values(ascending=False)
    vals = sorted_medias.values
    keys = sorted_medias.index
    
    # Construção do grafo de não-diferença
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            diff = abs(vals[i] - vals[j])
            if diff < hsd: 
                t1, t2 = keys[i], keys[j]
                adj[t1].add(t2)
                adj[t2].add(t1)
                
    cliques = []
    # Algoritmo Bron-Kerbosch para encontrar cliques máximos
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
    
    # CORREÇÃO AQUI: Força o nome do índice (cabeçalho da coluna 1) a ser igual ao original
    df_res.index.name = medias.index.name
    
    if EXIBIR_LOGS: print("✅ Tukey finalizado com sucesso.")
    return df_res.sort_values('Media', ascending=False)

def scott_knott(means, mse, df_resid, reps, n_trats=None):
    """
    Algoritmo Scott-Knott.
    """
    if EXIBIR_LOGS: print(f"🌲 Iniciando Scott-Knott | Médias a agrupar: {len(means)}")
    
    results = pd.DataFrame({'Media': means}).sort_values('Media', ascending=False)
    medias_ordenadas = results['Media'].values
    indices = results.index
    
    def cluster_medias(meds, ind):
        n = len(meds)
        if n < 2: return {ind[0]: 1}
        melhor_b0, corte_idx = -1, -1
        grand_mean = np.mean(meds)
        
        # Busca o ponto de corte que maximiza a soma de quadrados entre grupos (B0)
        for i in range(1, n):
            g1, g2 = meds[:i], meds[i:]
            b0 = i * (np.mean(g1) - grand_mean)**2 + (n-i) * (np.mean(g2) - grand_mean)**2
            if b0 > melhor_b0: melhor_b0, corte_idx = b0, i
            
        sigma2 = mse / reps
        # Estatística de teste Lambda
        lamb = (np.pi / (2 * (np.pi - 2))) * (melhor_b0 / sigma2)
        # Valor crítico Qui-Quadrado assintótico
        critico = stats.chi2.ppf(0.95, df=n/(np.pi-2)) 
        
        if lamb > critico:
            if EXIBIR_LOGS: print(f"✂️ Corte detectado no índice {corte_idx} (Lambda={lamb:.2f} > Crit={critico:.2f})")
            dict_left = cluster_medias(meds[:corte_idx], ind[:corte_idx])
            dict_right = cluster_medias(meds[corte_idx:], ind[corte_idx:])
            max_grp = max(dict_left.values())
            for k in dict_right: dict_right[k] += max_grp
            return {**dict_left, **dict_right}
        else: 
            return {x: 1 for x in ind}

    grupos_dict = cluster_medias(medias_ordenadas, indices)
    results['Grupo_Num'] = results.index.map(grupos_dict)
    unique_grps = sorted(results['Grupo_Num'].unique())
    mapa_letras = {num: get_letra_segura(i) for i, num in enumerate(unique_grps)}
    results['Grupo'] = results['Grupo_Num'].map(mapa_letras)
    
    # GARANTIA EXTRA: Assegura que o nome do índice está preservado aqui também
    results.index.name = means.index.name
    
    if EXIBIR_LOGS: print("✅ Scott-Knott finalizado.")
    return results[['Media', 'Grupo']]
# ==============================================================================
# 🏁 FIM DO BLOCO 06
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 07: Motores Estatísticos III (Rodar Modelos OLS - COM CACHE ⚡)
# ==============================================================================
from patsy.contrasts import Sum

@st.cache_data(show_spinner=False)
def rodar_analise_individual(df, cols_trats, col_resp, delineamento, col_bloco=None):
    """
    Roda ANOVA Individual Dinâmica (Suporta Fatorial).
    Recebe: cols_trats (LISTA de colunas).
    """
    df_calc = df.copy()
    res = {}
    
    # 1. Montagem Dinâmica da Fórmula (fator1 * fator2 * ...)
    # O operador '*' no Patsy gera automaticamente Efeitos Principais + Interações
    termos_trats = [f"C({trat}, Sum)" for trat in cols_trats]
    formula_trats = " * ".join(termos_trats)
    
    if delineamento == 'DBC': 
        formula = f"{col_resp} ~ {formula_trats} + C({col_bloco}, Sum)"
    else: 
        formula = f"{col_resp} ~ {formula_trats}"
    
    # 2. Execução do Modelo
    try:
        modelo = ols(formula, data=df_calc).fit()
        anova = sm.stats.anova_lm(modelo, typ=3)
    except:
        # Fallback para typ=1 se typ=3 falhar (raro, mas possível em dados desbalanceados)
        modelo = ols(formula, data=df_calc).fit()
        anova = sm.stats.anova_lm(modelo, typ=1)
        
    res['anova'] = anova
    res['modelo'] = modelo
    res['mse'] = modelo.mse_resid
    res['df_resid'] = modelo.df_resid
    
    # 3. Extração do P-valor (Busca o P-valor da interação de maior ordem ou do fator único)
    try:
        if len(cols_trats) == 1:
            res['p_val'] = anova.loc[f"C({cols_trats[0]}, Sum)", "PR(>F)"]
        else:
            # Tenta pegar a interação total (ex: A:B:C)
            p_found = 1.0
            for idx in anova.index:
                if str(idx).count(":") == len(cols_trats) - 1: # Ex: 2 fatores tem 1 ':', 3 fatores tem 2 ':'
                      res['p_val'] = anova.loc[idx, "PR(>F)"]
                      break
    except:
        res['p_val'] = 1.0 # Fallback seguro

    # 4. Pressupostos
    res['shapiro'] = stats.shapiro(modelo.resid)
    
    # Para Bartlett/Levene em Fatorial, precisamos criar um grupo único combinado
    if len(cols_trats) > 1:
        grupo_combinado = df_calc[cols_trats].astype(str).agg(' + '.join, axis=1)
    else:
        grupo_combinado = df_calc[cols_trats[0]]
        
    grupos_vals = [g[col_resp].values for _, g in df_calc.assign(temp_group=grupo_combinado).groupby('temp_group')]
    
    res['bartlett'] = stats.bartlett(*grupos_vals)
    res['levene'] = stats.levene(*grupos_vals, center='median')
    
    return res

@st.cache_data(show_spinner=False)
def rodar_analise_conjunta(df, col_trat_combo, col_resp, col_local, delineamento, col_bloco=None):
    """
    Para conjunta, usaremos a estratégia da 'Coluna Sintética' (col_trat_combo) 
    para simplificar a interação Tripla (Local x FatorA x FatorB).
    """
    df_calc = df.copy()
    res = {}
    if delineamento == 'DBC':
        termos = f"C({col_trat_combo}, Sum) * C({col_local}, Sum) + C({col_bloco}, Sum):C({col_local}, Sum)"
    else:
        termos = f"C({col_trat_combo}, Sum) * C({col_local}, Sum)"
        
    formula = f"{col_resp} ~ {termos}"
    
    try:
        modelo = ols(formula, data=df_calc).fit()
        anova = sm.stats.anova_lm(modelo, typ=3)
    except:
        modelo = ols(formula, data=df_calc).fit()
        anova = sm.stats.anova_lm(modelo, typ=1)
        
    res['anova'] = anova
    res['modelo'] = modelo
    res['mse'] = modelo.mse_resid
    res['df_resid'] = modelo.df_resid
    res['shapiro'] = stats.shapiro(modelo.resid)
    grupos = [g[col_resp].values for _, g in df_calc.groupby(col_trat_combo)]
    res['bartlett'] = stats.bartlett(*grupos)
    res['levene'] = stats.levene(*grupos, center='median')
    
    res['p_trat'] = 1.0
    res['p_interacao'] = 1.0
    
    for idx in anova.index:
        nome = str(idx)
        if col_trat_combo in nome and col_local not in nome and ":" not in nome:
            res['p_trat'] = anova.loc[idx, "PR(>F)"]
        if col_trat_combo in nome and col_local in nome and ":" in nome:
            res['p_interacao'] = anova.loc[idx, "PR(>F)"]
            
    return res

# --- NOVA FUNÇÃO ADICIONADA: CÁLCULO DE HOMOGENEIDADE PARA CONJUNTA ---
@st.cache_data(show_spinner=False)
def calcular_homogeneidade(df, col_trat, col_resp, col_local, col_bloco, delineamento):
    """
    Calcula a razão entre o maior e o menor MSE dos locais individuais.
    Retorna: (Razão, Maior_MSE, Menor_MSE)
    """
    if EXIBIR_LOGS: print(f"⚖️ Verificando homogeneidade de variâncias entre locais...")
    
    locais = df[col_local].unique()
    mses = []
    
    for loc in locais:
        df_loc = df[df[col_local] == loc]
        # Roda a análise individual para pegar o MSE (QMRes)
        # Nota: Passamos [col_trat] como lista porque a função individual espera lista
        res = rodar_analise_individual(df_loc, [col_trat], col_resp, delineamento, col_bloco)
        mses.append(res['mse'])
        
    if not mses: return 0, 0, 0
    
    max_mse = max(mses)
    min_mse = min(mses)
    
    if min_mse == 0: return 999, max_mse, min_mse # Evita divisão por zero
    
    razao = max_mse / min_mse
    
    if EXIBIR_LOGS: print(f"⚖️ Razão MSE: {razao:.2f} (Max: {max_mse:.4f}, Min: {min_mse:.4f})")
    
    return razao, max_mse, min_mse
# ==============================================================================
# 🏁 FIM DO BLOCO 07
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 08: Interface - Setup e CSS
# ==============================================================================
st.set_page_config(page_title="AgroStat Pro", page_icon="🌱", layout="wide")

# --- FUNÇÃO CSS PARA ROLAGEM DE ABAS ---
def configurar_estilo_abas():
    log_message("🎨 Aplicando estilos CSS ROBUSTOS para rolagem de abas...")
    st.markdown("""
        <style>
            div[data-baseweb="tab-list"] {
                display: flex !important;
                flex-wrap: nowrap !important;
                overflow-x: auto !important;
                white-space: nowrap !important;
                width: 100% !important;
                padding-bottom: 8px !important;
            }
            div[data-baseweb="tab"] {
                flex: 0 0 auto !important;
                width: auto !important;
                min-width: 50px !important;
                margin-right: 5px !important;
            }
            div[data-baseweb="tab-list"]::-webkit-scrollbar {
                height: 12px !important;
            }
            div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
                background-color: #888 !important;
                border-radius: 6px !important;
                border: 2px solid #f1f1f1 !important;
            }
            div[data-baseweb="tab-list"]::-webkit-scrollbar-track {
                background: #f1f1f1 !important;
            }
        </style>
    """, unsafe_allow_html=True)

configurar_estilo_abas()

st.title("🌱 AgroStat Pro: Análises Estatísticas")
# ==============================================================================
# 🏁 FIM DO BLOCO 08
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 09: Interface - Sidebar (V21 - MENU DE NAVEGAÇÃO)
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/seed.png", width=60) # Ícone opcional
st.sidebar.title("AgroStat Pro")

# --- MENU PRINCIPAL ---
# Define qual "parte" do aplicativo será exibida
modo_app = st.sidebar.radio(
    "Navegação:",
    ("📊 Análise Estatística", "🎲 Planejamento (Sorteio)"),
    index=0
)

st.sidebar.markdown("---")

# ==============================================================================
# LÓGICA CONDICIONAL DA SIDEBAR
# ==============================================================================

# --- MODO 1: ANÁLISE ESTATÍSTICA (O que já existia) ---
if modo_app == "📊 Análise Estatística":
    st.sidebar.header("📂 Configuração de Análise")

    if 'processando' not in st.session_state:
        st.session_state['processando'] = False

    if 'ultimo_arquivo_id' not in st.session_state:
        st.session_state['ultimo_arquivo_id'] = None

    arquivo = st.sidebar.file_uploader("Upload CSV ou Excel", type=["xlsx", "csv"], key="uploader")

    if arquivo is not None:
        if arquivo.file_id != st.session_state['ultimo_arquivo_id']:
            st.cache_data.clear() 
            st.session_state['ultimo_arquivo_id'] = arquivo.file_id
            st.session_state['processando'] = False 
            st.rerun()

    if arquivo:
        try:
            if arquivo.name.endswith('.csv'): 
                df = pd.read_csv(arquivo)
            else: 
                df = pd.read_excel(arquivo)
        except Exception as e:
            st.sidebar.error(f"Erro ao ler arquivo: {e}")
            st.stop()
            
        colunas = df.columns.tolist()
        st.sidebar.success(f"Carregado: {len(df)} linhas")
        st.sidebar.markdown("---")
        
        # Inputs de Configuração
        tipo_del = st.sidebar.radio("Delineamento Experimental:", ("DIC (Inteiramente Casualizado)", "DBC (Blocos Casualizados)"), on_change=reset_analise)
        delineamento = "DIC" if "DIC" in tipo_del else "DBC"
        
        cols_trats = st.sidebar.multiselect("Fatores/Tratamentos (Selecione 1 ou mais)", colunas, on_change=reset_analise)
        
        # --- ALTERAÇÃO AQUI: Rótulo mais rígido ---
        OPCAO_PADRAO = "Local Único (Análise Individual)" 
        col_local = st.sidebar.selectbox("Coluna de Local/Ambiente", [OPCAO_PADRAO] + [c for c in colunas if c not in cols_trats], on_change=reset_analise)
        
        col_bloco = None
        cols_ocupadas = cols_trats + [col_local]
        
        if delineamento == "DBC":
            col_bloco = st.sidebar.selectbox("Blocos (Repetições)", [c for c in colunas if c not in cols_ocupadas], on_change=reset_analise)
            cols_ocupadas.append(col_bloco)
        else:
            # --- ALTERAÇÃO AQUI: Remoção do (Automático) e (Opcional) ---
            # Agora só mostra as colunas disponíveis, obrigando a seleção de uma
            col_rep_dic = st.sidebar.selectbox("Coluna de Repetição / ID", [c for c in colunas if c not in cols_ocupadas], on_change=reset_analise)
            cols_ocupadas.append(col_rep_dic)

        lista_resps = st.sidebar.multiselect("Variáveis Resposta (Selecione 1 ou mais)", [c for c in colunas if c not in cols_ocupadas], on_change=reset_analise)

        # Detecção de Modo
        modo_analise = "INDIVIDUAL"
        if col_local != OPCAO_PADRAO:
            n_locais = len(df[col_local].unique())
            if n_locais > 1:
                modo_analise = "CONJUNTA"
                st.sidebar.info(f"🌍 Modo Conjunta Ativado! ({n_locais} locais)")
            else:
                st.sidebar.warning("⚠️ Coluna de Local selecionada, mas há apenas 1 local. Rodando modo Individual.")
        
        # --- EDITOR DE RÓTULOS (MAPA DE SUBSTITUIÇÃO) ---
        mapa_renomeacao = {} 
        cols_para_editar = [c for c in cols_trats]
        if col_local != OPCAO_PADRAO:
            cols_para_editar.append(col_local)
        
        if cols_para_editar:
            st.sidebar.markdown("---")
            with st.sidebar.expander("🏷️ Renomear Rótulos (Opcional)", expanded=False):
                st.caption("Substitua códigos (ex: 1, 2) por nomes reais. Atualiza gráficos e tabelas.")
                for col_edit in set(cols_para_editar):
                    st.markdown(f"**📝 {col_edit}**")
                    vals_originais = sorted(df[col_edit].dropna().unique())
                    if len(vals_originais) > 50:
                        st.warning(f"Muitos níveis ({len(vals_originais)}). Mostrando apenas os 10 primeiros.")
                        vals_originais = vals_originais[:10]

                    col_map = {}
                    for val in vals_originais:
                        novo_val = st.text_input(f"{val} ➝", value=str(val), key=f"ren_{col_edit}_{val}")
                        if novo_val != str(val):
                            col_map[val] = novo_val
                    
                    if col_map:
                        mapa_renomeacao[col_edit] = col_map

        st.sidebar.markdown("---")

        if st.sidebar.button("🚀 Processar Estatística", type="primary"):
            st.session_state['processando'] = True

        st.sidebar.markdown("---")
        with st.sidebar.expander("🔧 Manutenção / Cache"):
            if st.button("🧹 Limpar Memória"):
                st.cache_data.clear()
                st.session_state['processando'] = False
                st.rerun()

# --- MODO 2: PLANEJAMENTO (Novo) ---
elif modo_app == "🎲 Planejamento (Sorteio)":
    st.sidebar.info("🛠️ Você está no modo de Pré-Experimento. Configure os tratamentos e sorteie o croqui na tela principal.")
    # Reseta o estado de processamento da análise para não misturar as coisas
    st.session_state['processando'] = False 
# ==============================================================================
# 🏁 FIM DO BLOCO 09
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 10: Execução, Alertas Explícitos e Diagnóstico (V10 - TRAVA DE MODO)
# ==============================================================================
# TRAVA DE SEGURANÇA: Só roda se o botão foi clicado E se estivermos no modo Análise
if st.session_state['processando'] and modo_app == "📊 Análise Estatística":
    if not lista_resps:
        st.error("⚠️ Por favor, selecione pelo menos uma Variável Resposta.")
    elif not cols_trats:
         st.error("⚠️ Por favor, selecione pelo menos um Fator/Tratamento.")
    else:
        # --- 0. APLICAÇÃO INTELIGENTE DE RENOMEAÇÃO ---
        df_analise = df.copy()
        
        if mapa_renomeacao:
            log_message("🏷️ Detectada solicitação de renomeação de rótulos...")
            try:
                for col_alvo, dic_troca in mapa_renomeacao.items():
                    dic_troca_str = {str(k): v for k, v in dic_troca.items()}
                    df_analise[col_alvo] = df_analise[col_alvo].astype(str).replace(dic_troca_str)
                    log_message(f"✅ Coluna '{col_alvo}': {len(dic_troca)} itens renomeados.")
            except Exception as e:
                st.error(f"Erro ao renomear rótulos: {e}")
        
        df = df_analise

        st.markdown(f"### 📋 Resultados: {len(lista_resps)} variáveis processadas")
        
        # --- 0.1 ANÁLISE DE DIMENSÕES (LOGS) ---
        log_message("🔍 Analisando dimensões dos fatores...")
        dimensoes = []
        for f in cols_trats:
            n_niveis = df[f].nunique()
            dimensoes.append(str(n_niveis))
            log_message(f"📊 Fator '{f}': {n_niveis} níveis detectados.")
        
        esquema_txt = "x".join(dimensoes)
        if len(cols_trats) > 1:
            log_message(f"✅ Esquema Fatorial [{esquema_txt}] identificado.")
            st.info(f"🔬 **Esquema Fatorial Detectado:** {esquema_txt} ({' x '.join(cols_trats)})")
        else:
            log_message(f"✅ Experimento Unifatorial [{esquema_txt}] identificado.")

        for i, col_resp_original in enumerate(lista_resps):
            with st.expander(f"📊 Variável: {col_resp_original}", expanded=(i==0)):
                
                # TRANSFORMAÇÃO
                transf_atual = get_transformacao_atual(col_resp_original)
                df_proc, col_resp = aplicar_transformacao(df.copy(), col_resp_original, transf_atual)
                
                # --- CRIAÇÃO DA COLUNA SINTÉTICA (COMBO) ---
                col_combo = "TRAT_COMBINADO"
                if len(cols_trats) > 1:
                    df_proc[col_combo] = df_proc[cols_trats].astype(str).agg(' + '.join, axis=1)
                else:
                    col_combo = cols_trats[0] 
                
                if transf_atual != "Nenhuma":
                    st.info(f"🔄 **Transformação Ativa:** {transf_atual} (Coluna: {col_resp})")
                
                st.markdown(f"### Análise de: **{col_resp}**")
                
                res_analysis = {}
                p_shap, p_bart, p_lev = None, None, None
                res_model = None
                anova_tab = None
                extras = {} 
                p_final_trat = 1.0
                modo_atual_txt = ""

                # --- 1. EXECUÇÃO DOS CÁLCULOS ---
                if modo_analise == "INDIVIDUAL":
                    modo_atual_txt = "INDIVIDUAL"
                    res = rodar_analise_individual(df_proc, cols_trats, col_resp, delineamento, col_bloco)
                    res_analysis = res
                    p_shap, p_bart, p_lev = res['shapiro'][1], res['bartlett'][1], res['levene'][1]
                    res_model = res['modelo']
                    anova_tab = formatar_tabela_anova(res['anova'])
                    p_final_trat = res['p_val']
                    extras = calcular_metricas_extras(anova_tab, res_model, cols_trats[0])
                    st.markdown("#### 📝 Métricas Estatísticas")
                    txt_metrics = gerar_relatorio_metricas(anova_tab, res_model, cols_trats[0], df_proc[col_resp].mean(), p_final_trat)
                    st.markdown(txt_metrics)

                else: # CONJUNTA
                    modo_atual_txt = "CONJUNTA"
                    res_conj = rodar_analise_conjunta(df_proc, col_combo, col_resp, col_local, delineamento, col_bloco)
                    res_analysis = res_conj
                    p_shap, p_bart, p_lev = res_conj['shapiro'][1], res_conj['bartlett'][1], res_conj['levene'][1]
                    res_model = res_conj['modelo']
                    anova_tab = formatar_tabela_anova(res_conj['anova'])
                    razao, _, _ = calcular_homogeneidade(df_proc, col_combo, col_resp, col_local, col_bloco, delineamento)
                    p_final_trat = res_conj['p_trat']
                    extras = calcular_metricas_extras(anova_tab, res_model, col_combo)
                    st.markdown("#### 📝 Métricas Estatísticas")
                    txt_metrics = gerar_relatorio_metricas(anova_tab, res_model, col_combo, df_proc[col_resp].mean(), p_final_trat, razao)
                    st.markdown(txt_metrics)
                    if razao and razao > 7: 
                        st.error(f"🚨 **Violação de Homogeneidade (MSE):** Razão {razao:.2f} > 7.")

                # --- 2. ALERTAS ---
                cv_val = (np.sqrt(res_model.mse_resid)/df_proc[col_resp].mean())*100
                if cv_val > 20: st.error(f"🚨 **CV Muito Alto ({cv_val:.2f}%):** Baixa precisão.")
                if "🔴" in extras['ac_class']: st.error("🚨 **Acurácia Baixa/Zerada.**")
                if "🔴" in extras['h2_class']: st.error("🚨 **Herdabilidade Baixa/Zerada.**")
                if "🔴" in extras['r2_class']: st.error(f"🚨 **R² Baixo ({extras['r2']:.2f}).**")

                # --- 3. EXIBIÇÃO ---
                if p_final_trat < 0.05:
                    st.success(f"✅ **Diferença Significativa (P < 0.05).**")
                else:
                    st.error(f"⚠️ **Não Significativo (P >= 0.05).**")

                st.markdown("### 📊 Análise de Variância (ANOVA)")
                st.dataframe(anova_tab)

                if modo_atual_txt == "CONJUNTA":
                     p_int = res_conj.get('p_interacao', 1.0)
                     if p_int < 0.05: st.error(f"⚠️ **Interação GxA Significativa (P={p_int:.4f}).**")
                     else: st.success(f"✅ **Interação GxA Não Significativa.**")

                st.markdown("---")
                st.markdown("#### 🩺 Diagnóstico dos Pressupostos")
                st.markdown(gerar_tabela_diagnostico(p_shap, p_bart, p_lev))
                
                log_message(f"🚀 Verificação de pressupostos para {col_resp}...")

                # ==========================================================
                # 🪄 A MÁGICA DO FATORIAL ACONTECE AQUI
                # ==========================================================
                col_trat_original_lista = cols_trats # Backup da lista (se precisar)
                col_trat = col_combo 
                # ==========================================================
# ==============================================================================
# 🏁 FIM DO BLOCO 10
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 11: A "Árvore de Decisão Universal" (Lógica de Pressupostos)
# ==============================================================================
                # Esta lógica agora se aplica tanto para DIC, DBC Individual quanto Conjunta.
                # As variáveis p_shap, p_bart, p_lev foram definidas no Bloco 10.

                is_nan_shap = pd.isna(p_shap)
                is_nan_bart = pd.isna(p_bart)
                is_nan_lev = pd.isna(p_lev)
                
                normal_ok = (p_shap >= 0.05) if not is_nan_shap else False
                bart_ok = (p_bart >= 0.05) if not is_nan_bart else False
                lev_ok = (p_lev >= 0.05) if not is_nan_lev else False

                if is_nan_shap:
                    analise_valida = False
                    if bart_ok and lev_ok:
                        log_message("Log: NaN.1"); st.error("🚨 Erro de Cálculo nos Pressupostos: Não foi possível calcular os testes estatísticos (retorno NaN). Isso geralmente ocorre quando há dados insuficientes ou variância zero (todos os valores iguais). Neste caso em específico, o teste de Shapiro-Wilk (teste de normalidade) apresentou erro, e por esse motivo não tem como verificar a normalidade dos dados, apesar que o teste de homogenecidade (testes de Bartlett e Levene) estão aprovados conforme desejamos. Porém, mesmo assim a análise não pode ser validada. Recomendo Transformar os dados ou usar a estatística não-paramétrica.")
                    elif is_nan_bart and lev_ok:
                        log_message("Log: NaN.2"); st.error("🚨 Erro de Cálculo nos Pressupostos: Não foi possível calcular os testes estatísticos (retorno NaN). Isso geralmente ocorre quando há dados insuficientes ou variância zero (todos os valores iguais). Neste caso em específico, o teste de Shapiro-Wilk (teste de normalidade) apresentou erro, e por esse motivo não tem como verificar a normalidade dos dados, apesar que a homogenecidade está aprovada com o teste de Levene, conforme desejamos. Porém, mesmo assim a análise não pode ser validada. Recomendo Transformar os dados ou usar a estatística não-paramétrica.")
                    elif bart_ok and is_nan_lev:
                        log_message("Log: NaN.3"); st.error("🚨 Erro de Cálculo nos Pressupostos: Não foi possível calcular os testes estatísticos (retorno NaN). Isso geralmente ocorre quando há dados insuficientes ou variância zero (todos os valores iguais). Neste caso em específico, o teste de Shapiro-Wilk (teste de normalidade) apresentou erro, e por esse motivo não tem como verificar a normalidade dos dados, apesar que a homogenecidade está aprovada com o teste de Bartlett, conforme desejamos. Porém, mesmo assim a análise não pode ser validada. Recomendo Transformar os dados ou usar a estatística não-paramétrica.")
                    else:
                        log_message("Log: NaN.4"); st.error("🚨 Erro de Cálculo nos Pressupostos: Não foi possível calcular os testes estatísticos (retorno NaN). Isso geralmente ocorre quando há dados insuficientes ou variância zero (todos os valores iguais). A análise não pode ser validada, pois todos os testes tiveram valores invalidados. Recomendo Transformar os dados , para verificar se algum teste fica válido, ou entao usar a estatística não-paramétrica.")

                elif normal_ok and (is_nan_bart or is_nan_lev):
                    if is_nan_bart and is_nan_lev:
                        log_message("Log: NaN.7"); analise_valida = False; st.error("🚨 Erro Crítico: Os dados seguem a normalidade (Shapiro-Wilk), mas nao é possivel verificar a Homogeneidade. Ambos os testes (Bartlett e Levene) retornaram erro de cálculo (NaN). A análise foi suspensa por segurança se caso permanecer mesmo após as transformaçoes. Recomendo Transformar os dados , para verificar se algum teste fica válido, ou entao usar a estatística não-paramétrica.")
                    elif is_nan_bart and lev_ok:
                        log_message("Log: NaN.5"); analise_valida = True; st.success("✅ Os Dados segeum normalidade (teste de Shapiro-Wilk). Apesar do teste de Bartlett não pôder ser calculado (NaN), mas foi ignorado pois o teste de Levene confirmou a homogeneidade das variâncias com sucesso.")
                    elif is_nan_lev and bart_ok:
                        log_message("Log: NaN.6"); analise_valida = True; st.success("✅ Os Dados segeum normalidade (teste de Shapiro-Wilk). Apesar do teste de Levene não pôder ser calculado (NaN), mas foi ignorado pois o teste de Bartlett confirmou a homogeneidade das variâncias com sucesso.")
                    else:
                        log_message("Log: Normal OK, um NaN falhou"); analise_valida = False; st.error("🚨 Violação de Homogeneidade: Os dados seguem distribuição normal (teste de Shapiro-Wilk), mas o único teste de homogeneidade válido indicou heterogeneidade. A análise não é segura.")

                else:
                    if normal_ok:
                        if bart_ok and lev_ok:
                            st.success("✅ Todos os pressupostos foram atendidos. Os dados possuem distribuição normal (normalidade; teste de Shapiro-Wilk) e possuem variâncias dos seus grupos iguais (homocedasticidade; testes de Bartlett e de Levene). Pode confiar na ANOVA.")
                            analise_valida = True
                        elif bart_ok and not lev_ok:
                            st.success("✅ Os pressupostos de normalidade (Shapiro-Wilk) e o teste de homogeneidade foram atendidos (Bartlett). Os testes de Shapiro-Wilk e Bartlett foram aprovados.")
                            analise_valida = True
                        elif not bart_ok and lev_ok:
                            st.success("✅ Os pressupostos de normalidade (Shapiro-Wilk) e o teste de homogeneidade foram atendidos (Levene). Levene confirmou a homogeneidade.")
                            analise_valida = True
                        else:
                            st.error("🚨 Embora os dados sigam distribuição normal, as variâncias são heterogêneas. Transforme os dados.")
                            analise_valida = False
                    else:
                        if bart_ok and lev_ok:
                            # Cenário 5
                            st.error("🚨 Violação de Normalidade: Embora as variâncias sejam homogêneas (testes de Bartlett e Levene foram aprovados), os dados não seguem distribuição normal (Shapiro-Wilk reprovado). Como a normalidade é um pré-requisito obrigatório, a ANOVA não deve ser realizada sem antes tentar a transformação dos dados. Transforme os dados ou use estatística não-paramétrica.")
                            analise_valida = False
                        elif bart_ok and not lev_ok:
                            # Cenário 6
                            st.error("🚨 Violação de Normalidade: Os dados não possuem distribuição normal (Shapiro-Wilk reprovado) e também houve reprovação da homocedasticidade do teste de Levene, apesar de que o teste de Bartlett foi aprovado.Mas independente da aprovaçao de qualquer teste de homocedasticidade o teste de normalidade foi reprovado e portanto a ANOVA é inválida. Transforme os dados ou use estatística não-paramétrica.")
                            analise_valida = False
                        elif not bart_ok and lev_ok:
                            # Cenário 7
                            st.error("🚨 Violação de Normalidade: Embora as variâncias sejam homogêneas (apenas o teste de Levene foi aprovado), os dados não seguem distribuição normal (Shapiro-Wilk reprovado). Como a normalidade é um pré-requisito obrigatório, a ANOVA não deve ser realizada sem antes tentar a transformação dos dados. Transforme os dados ou use estatística não-paramétrica.")
                            analise_valida = False
                        else:
                            # Cenário 8
                            st.error("🚨 Violação Crítica Total: Os dados não possuem distribuição normal (Shapiro-Wilk reprovado) e as variâncias são heterogêneas (testes de Bartlett e Levene foram reprovados). A ANOVA é inválida. Transforme os dados ou use estatística não-paramétrica.")
                            analise_valida = False
# ==============================================================================
# 🏁 FIM DO BLOCO 11
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 12: Visualização (V28 - Correção: Edição de Grupos na Interação)
# ==============================================================================
                # --- FUNÇÃO INTERNA: GERADOR DE MATRIZ DE DESDOBRAMENTO ---
                def gerar_dataframe_matriz_total(df_input, f_linha, f_coluna, metodo_func, mse_global, df_res_global):
                    """
                    Gera a matriz de dupla entrada com letras Maiúsculas (Linha) e Minúsculas (Coluna).
                    RETORNA: Objeto Styler formatado com CSS forçado para centralização.
                    """
                    log_message(f"🚀 Iniciando desdobramento: {f_linha} x {f_coluna}...")
                    
                    # A) MAIÚSCULAS: Fixa Linha -> Compara Colunas
                    dict_upper = {}
                    niveis_l = df_input[f_linha].unique()
                    for nl in niveis_l:
                        df_s = df_input[df_input[f_linha] == nl]
                        meds = df_s.groupby(f_coluna)[col_resp].mean()
                        reps = df_s.groupby(f_coluna)[col_resp].count().mean()
                        res_comp = metodo_func(meds, mse_global, df_res_global, reps, len(meds))
                        for nc, row in res_comp.iterrows():
                            dict_upper[(str(nl), str(nc))] = str(row.iloc[1]).upper()

                    # B) MINÚSCULAS: Fixa Coluna -> Compara Linhas
                    dict_lower = {}
                    niveis_c = df_input[f_coluna].unique()
                    for nc in niveis_c:
                        df_s = df_input[df_input[f_coluna] == nc]
                        meds = df_s.groupby(f_linha)[col_resp].mean()
                        reps = df_s.groupby(f_linha)[col_resp].count().mean()
                        res_comp = metodo_func(meds, mse_global, df_res_global, reps, len(meds))
                        for nl, row in res_comp.iterrows():
                            dict_lower[(str(nl), str(nc))] = str(row.iloc[1]).lower()

                    # C) MONTAGEM DA MATRIZ
                    pivot = df_input.pivot_table(index=f_linha, columns=f_coluna, values=col_resp, aggfunc='mean')
                    df_matriz = pivot.copy().astype(object)
                    for l in pivot.index:
                        for c in pivot.columns:
                            val = pivot.loc[l, c]
                            u = dict_upper.get((str(l), str(c)), "?")
                            low = dict_lower.get((str(l), str(c)), "?")
                            df_matriz.loc[l, c] = f"{val:.2f} {u} {low}"
                    
                    # --- FORMATAÇÃO E CSS ---
                    df_matriz.index.name = f_linha
                    cols_atuais = df_matriz.columns.tolist()
                    multi_cols = pd.MultiIndex.from_product([[f_coluna], cols_atuais])
                    df_matriz.columns = multi_cols
                    
                    styler = df_matriz.style.set_properties(**{
                        'text-align': 'center'
                    }).set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center !important'), ('vertical-align', 'middle !important')]},
                        {'selector': 'th.col_heading', 'props': [('text-align', 'center !important')]},
                        {'selector': 'th.index_name', 'props': [('text-align', 'center !important')]},
                        {'selector': 'td', 'props': [('text-align', 'center !important'), ('vertical-align', 'middle !important')]}
                    ])

                    log_message("✅ Matriz de dupla entrada concluída.")
                    return styler

                # --- FUNÇÃO AUXILIAR DE ESTILO AVANÇADO (CORRIGIDA) ---
                def estilizar_grafico_avancado(fig, configs, dados_max=None):
                    range_y = None
                    if dados_max is not None:
                        margem_topo = 1.25 if configs['posicao_texto'] == 'outside' else 1.05
                        range_y = [0, dados_max * margem_topo]

                    template_texto = '<b>%{text}</b>' if configs['letras_negrito'] else '%{text}'
                    show_line = True
                    mirror_bool = False
                    if configs['estilo_borda'] == "Caixa (Espelhado)": mirror_bool = True
                    elif configs['estilo_borda'] == "Sem Bordas": show_line = False

                    tick_mode = "outside" if configs['mostrar_ticks'] else ""
                    mapa_dash = {"Pontilhado": "dot", "Tracejado": "dash", "Sólido": "solid"}
                    estilo_grid = mapa_dash.get(configs['estilo_subgrade'], 'dot')

                    pos_texto_final = configs['posicao_texto']
                    if fig.data and fig.data[0].type == 'scatter':
                        if pos_texto_final == 'outside': pos_texto_final = 'top center'
                        elif pos_texto_final == 'inside': pos_texto_final = 'bottom center'
                        elif pos_texto_final == 'auto': pos_texto_final = 'top center'

                    fig.update_layout(
                        title=dict(text=f"<b>{configs['titulo_custom']}</b>", x=0.5, xanchor='center', font=dict(size=configs['font_size'] + 4, color=configs['cor_texto'])),
                        paper_bgcolor=configs['cor_fundo'], plot_bgcolor=configs['cor_fundo'],
                        margin=dict(l=60, r=40, t=60, b=60), height=configs['altura'],
                        yaxis=dict(
                            title=dict(text=f"<b>{configs['label_y']}</b>", font=dict(color=configs['cor_texto'])),
                            showgrid=configs['mostrar_grid'], gridcolor=configs['cor_grade'],
                            showline=show_line, linewidth=1, linecolor=configs['cor_texto'], mirror=mirror_bool,
                            ticks=tick_mode, ticklen=6, tickcolor=configs['cor_texto'], tickwidth=1,
                            minor=dict(showgrid=configs['mostrar_subgrade'], gridcolor=configs['cor_subgrade'], gridwidth=0.5, griddash=estilo_grid),
                            rangemode='tozero', range=range_y, tickfont=dict(color=configs['cor_texto'], size=configs['font_size'])
                        ),
                        xaxis=dict(
                            title=dict(text=f"<b>{configs['label_x']}</b>", font=dict(color=configs['cor_texto'])),
                            tickfont=dict(color=configs['cor_texto'], size=configs['font_size']),
                            showline=show_line, linewidth=1, linecolor=configs['cor_texto'], mirror=mirror_bool
                        ),
                        font=dict(family=configs['font_family'], size=configs['font_size'], color=configs['cor_texto']),
                        showlegend=configs['mostrar_legenda'],
                        legend=dict(title=dict(text=f"<b>{configs['titulo_legenda']}</b>", font=dict(color=configs['cor_texto'])), bgcolor=configs['cor_fundo'], borderwidth=0)
                    )
                    
                    # Atualiza propriedades gerais e FORÇA a exibição da legenda nos traços
                    fig.update_traces(
                        texttemplate=template_texto, 
                        textposition=pos_texto_final, 
                        textfont=dict(size=configs['font_size'], color=configs['cor_texto']), 
                        cliponaxis=False, 
                        marker_line_color=configs['cor_texto'], 
                        marker_line_width=1.0,
                        showlegend=configs['mostrar_legenda'] # IMPORTANTE: Força a legenda no traço
                    )
                    
                    # Aplica a cor da barra se for cor única
                    if configs.get('cor_barras'):
                        fig.update_traces(marker_color=configs['cor_barras'])
                    
                    # Atualiza os nomes das legendas (Renomeação)
                    if configs['mapa_nomes_grupos']:
                        for trace in fig.data:
                            if trace.name in configs['mapa_nomes_grupos']:
                                novo_nome = configs['mapa_nomes_grupos'][trace.name]
                                trace.name = novo_nome
                                trace.legendgroup = novo_nome # Atualiza o grupo também para consistência
                                
                    return fig

                # --- COMPONENTE DE EDITOR COM FORMULÁRIO (ORIGINAL) ---
                def mostrar_editor_grafico(key_prefix, titulo_padrao, label_x_padrao, label_y_padrao, usar_cor_unica=True, grupos_sk=None):
                    with st.expander(f"✏️ Personalizar Visual do Gráfico"):
                        with st.form(key=f"form_{key_prefix}"):
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.markdown("##### 🎨 Cores & Linhas")
                                cor_fundo = st.color_picker("Fundo", "#FFFFFF", key=f"{key_prefix}_bg")
                                cor_texto = st.color_picker("Textos/Eixos", "#000000", key=f"{key_prefix}_txt")
                                c_g1, c_g2 = st.columns(2)
                                with c_g1: cor_grade = st.color_picker("Grade Princ.", "#E5E5E5", key=f"{key_prefix}_grd_c")
                                with c_g2: cor_subgrade = st.color_picker("Sub-grade", "#F5F5F5", key=f"{key_prefix}_subgrd_c")
                                st.markdown("---")
                                cor_barras = None; cores_map = {}; mapa_nomes_grupos = {}
                                if usar_cor_unica: cor_barras = st.color_picker("Barras (Principal)", "#D6D6D6", key=f"{key_prefix}_bar")
                                else:
                                    st.caption("Configurar Grupos:")
                                    for idx, grp in enumerate(grupos_sk or []):
                                        col_g1, col_g2 = st.columns([1, 2])
                                        cp = px.colors.qualitative.Plotly[idx % len(px.colors.qualitative.Plotly)]
                                        cores_map[grp] = col_g1.color_picker(f"Cor {grp}", value=cp, key=f"{key_prefix}_sk_{grp}")
                                        novo_nome = col_g2.text_input(f"Legenda {grp}", value=str(grp), key=f"{key_prefix}_nm_{grp}")
                                        mapa_nomes_grupos[str(grp)] = novo_nome
                            with c2:
                                st.markdown("##### 📝 Textos")
                                titulo_custom = st.text_input("Título", value=titulo_padrao, key=f"{key_prefix}_tit")
                                label_y = st.text_input("Eixo Y", value=label_y_padrao, key=f"{key_prefix}_ly")
                                label_x = st.text_input("Eixo X", value=label_x_padrao, key=f"{key_prefix}_lx")
                                
                                val_legenda_padrao = False if usar_cor_unica else True
                                mostrar_legenda = st.checkbox("Mostrar Legenda", value=val_legenda_padrao, key=f"{key_prefix}_show_leg")
                                
                                if mostrar_legenda: titulo_legenda = st.text_input("Título da Legenda", value="Grupos", key=f"{key_prefix}_leg_tit")
                                else: titulo_legenda = ""
                                
                                st.markdown("##### 🔠 Rótulos")
                                mapa_pos = {"Externo (Topo)": "outside", "Interno (Dentro)": "inside", "Automático": "auto"}
                                pos_escolhida = st.selectbox("Posição", list(mapa_pos.keys()), key=f"{key_prefix}_pos")
                                letras_negrito = st.checkbox("Negrito", value=False, key=f"{key_prefix}_bold")
                            with c3:
                                st.markdown("##### 📐 Estrutura")
                                font_family = st.selectbox("Fonte", ["Arial", "Times New Roman", "Courier New", "Verdana"], key=f"{key_prefix}_font")
                                font_size = st.number_input("Tamanho Fonte", 10, 30, 14, key=f"{key_prefix}_fs")
                                altura = st.slider("Altura (px)", 300, 800, 450, key=f"{key_prefix}_h")
                                st.markdown("---")
                                st.markdown("##### ▦ Grades & Eixos")
                                mostrar_grid = st.checkbox("Grade Principal", True, key=f"{key_prefix}_grid")
                                mostrar_subgrade = st.checkbox("Sub-grades", False, key=f"{key_prefix}_subgrid")
                                estilo_subgrade = st.selectbox("Estilo Sub-grade", ["Pontilhado", "Tracejado", "Sólido"], key=f"{key_prefix}_stl_sub")
                                st.markdown("---")
                                st.markdown("##### 🔳 Bordas & Ticks")
                                estilo_borda = st.selectbox("Estilo da Borda", ["Apenas L (Eixos)", "Caixa (Espelhado)", "Sem Bordas"], key=f"{key_prefix}_borda")
                                mostrar_ticks = st.checkbox("Mostrar Ticks (Traços)", True, key=f"{key_prefix}_ticks")
                            st.markdown("---")
                            submit_button = st.form_submit_button("🔄 Atualizar Gráfico")
                    return {"cor_fundo": cor_fundo, "cor_texto": cor_texto, "cor_grade": cor_grade, "cor_subgrade": cor_subgrade, "cor_barras": cor_barras, "cores_map": cores_map, "mapa_nomes_grupos": mapa_nomes_grupos, "titulo_custom": titulo_custom, "label_y": label_y, "label_x": label_x, "titulo_legenda": titulo_legenda, "font_family": font_family, "font_size": font_size, "altura": altura, "mostrar_grid": mostrar_grid, "mostrar_subgrade": mostrar_subgrade, "estilo_subgrade": estilo_subgrade, "estilo_borda": estilo_borda, "mostrar_ticks": mostrar_ticks, "posicao_texto": mapa_pos[pos_escolhida], "letras_negrito": letras_negrito, "mostrar_legenda": mostrar_legenda}

                # ----------------------------------------------------------
                # CENÁRIO A: ANÁLISE INDIVIDUAL (Ou Fatorial Combinação)
                # ----------------------------------------------------------
                if modo_analise == "INDIVIDUAL":
                    medias_ind = df_proc.groupby(col_trat)[col_resp].mean()
                    reps_ind = df_proc.groupby(col_trat)[col_resp].count().mean()
                    n_trats_ind = len(medias_ind)
                    max_val_ind = medias_ind.max()
                    
                    df_tukey_ind = tukey_manual_preciso(medias_ind, res['mse'], res['df_resid'], reps_ind, n_trats_ind)
                    df_sk_ind = scott_knott(medias_ind, res['mse'], res['df_resid'], reps_ind, n_trats_ind)

                    tabs_ind = st.tabs(["📦 Teste de Tukey", "📦 Teste de Scott-Knott", "📈 Gráficos"])
                    interacao_sig = (len(cols_trats) >= 2 and res['p_val'] < 0.05)

                    with tabs_ind[0]: # TUKEY
                        st.markdown("#### Ranking Geral (Tukey)")
                        st.dataframe(df_tukey_ind.style.format({"Media": "{:.2f}"}))
                        if interacao_sig:
                            st.markdown("---")
                            st.subheader("🔠 Matriz de Desdobramento (Tukey)")
                            fl_tk = st.selectbox("Fator na Linha", cols_trats, key=f"mat_tk_l_{col_resp}")
                            fc_tk = [f for f in cols_trats if f != fl_tk][0]
                            df_m_tk = gerar_dataframe_matriz_total(df_proc, fl_tk, fc_tk, tukey_manual_preciso, res['mse'], res['df_resid'])
                            st.dataframe(df_m_tk)
                            st.caption("Médias seguidas por letras Maiúsculas na linha e minúsculas na coluna não diferem (Tukey 5%).")

                    with tabs_ind[1]: # SCOTT-KNOTT
                        st.markdown("#### Ranking Geral (Scott-Knott)")
                        st.dataframe(df_sk_ind.style.format({"Media": "{:.2f}"}))
                        if interacao_sig:
                            st.markdown("---")
                            st.subheader("🔠 Matriz de Desdobramento (Scott-Knott)")
                            fl_sk = st.selectbox("Fator na Linha", cols_trats, key=f"mat_sk_l_{col_resp}")
                            fc_sk = [f for f in cols_trats if f != fl_sk][0]
                            df_m_sk = gerar_dataframe_matriz_total(df_proc, fl_sk, fc_sk, scott_knott, res['mse'], res['df_resid'])
                            st.dataframe(df_m_sk)
                            st.caption("Médias seguidas por letras Maiúsculas na linha e minúsculas na coluna não diferem (Scott-Knott 5%).")

                    with tabs_ind[2]: # GRÁFICOS
                        st.markdown("#### 1. Gráfico de Tukey")
                        cfg_tk = mostrar_editor_grafico(f"tk_ind_{col_resp}", "Teste de Tukey", col_trat, col_resp, usar_cor_unica=True)
                        f_tk = px.bar(df_tukey_ind.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Letras')
                        st.plotly_chart(estilizar_grafico_avancado(f_tk, cfg_tk, max_val_ind), use_container_width=True)
                        st.markdown("---")
                        st.markdown("#### 2. Gráfico de Scott-Knott")
                        grps_sk = sorted(df_sk_ind['Grupo'].unique())
                        cfg_sk = mostrar_editor_grafico(f"sk_ind_{col_resp}", "Teste de Scott-Knott", col_trat, col_resp, usar_cor_unica=False, grupos_sk=grps_sk)
                        f_sk = px.bar(df_sk_ind.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupo', color='Grupo', color_discrete_map=cfg_sk['cores_map'])
                        st.plotly_chart(estilizar_grafico_avancado(f_sk, cfg_sk, max_val_ind), use_container_width=True)

                # ----------------------------------------------------------
                # CENÁRIO B: ANÁLISE CONJUNTA
                # ----------------------------------------------------------
                else:
                    locais_unicos = sorted(df_proc[col_local].unique())
                    titulos_abas = ["📊 Média Geral"] + [f"📍 {loc}" for loc in locais_unicos] + ["📈 Interação"]
                    abas = st.tabs(titulos_abas)
                    p_int_conj = res_conj.get('p_interacao', 1.0)
                    
                    # --- ABA 0: MÉDIA GERAL + GRÁFICO GERAL ---
                    with abas[0]: 
                        # Lógica de Avisos (Warnings)
                        if p_int_conj < 0.05:
                            st.warning("⚠️ Interação Significativa Detectada: O comportamento dos tratamentos varia entre os ambientes. A Média Geral pode mascarar o desempenho real. Analise os desdobramentos dentro de cada local.")
                        
                        if res_conj['p_trat'] >= 0.05:
                            st.warning("⚠️ Sem diferença significativa nos tratamentos (Média Geral).")

                        # Cálculos (Tukey + Scott-Knott)
                        medias_geral = df_proc.groupby(col_trat)[col_resp].mean()
                        reps_geral = df_proc.groupby(col_trat)[col_resp].count().mean()
                        max_val_geral = medias_geral.max()

                        df_tukey_geral = tukey_manual_preciso(medias_geral, res_conj['mse'], res_conj['df_resid'], reps_geral, len(medias_geral))
                        df_sk_geral = scott_knott(medias_geral, res_conj['mse'], res_conj['df_resid'], reps_geral, len(medias_geral))

                        # SUB-ABAS PARA ESCOLHER O TESTE
                        sub_abas_geral = st.tabs(["📦 Teste de Tukey", "📦 Teste de Scott-Knott"])
                        
                        # --- SUB-ABA TUKEY ---
                        with sub_abas_geral[0]:
                            st.markdown("#### Ranking Geral da Rede (Tukey)")
                            st.dataframe(df_tukey_geral.style.format({"Media": "{:.2f}"}))
                            st.markdown("---")
                            st.markdown("#### 📊 Gráfico (Tukey)")
                            cfg_tk_geral = mostrar_editor_grafico(f"tk_geral_{col_resp}", "Média Geral (Tukey)", col_trat, col_resp, usar_cor_unica=True)
                            f_tk_geral = px.bar(df_tukey_geral.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Letras')
                            st.plotly_chart(estilizar_grafico_avancado(f_tk_geral, cfg_tk_geral, max_val_geral), use_container_width=True)

                        # --- SUB-ABA SCOTT-KNOTT ---
                        with sub_abas_geral[1]:
                            st.markdown("#### Ranking Geral da Rede (Scott-Knott)")
                            st.dataframe(df_sk_geral.style.format({"Media": "{:.2f}"}))
                            st.markdown("---")
                            st.markdown("#### 📊 Gráfico (Scott-Knott)")
                            grps_sk_geral = sorted(df_sk_geral['Grupo'].unique())
                            cfg_sk_geral = mostrar_editor_grafico(f"sk_geral_{col_resp}", "Média Geral (Scott-Knott)", col_trat, col_resp, usar_cor_unica=False, grupos_sk=grps_sk_geral)
                            f_sk_geral = px.bar(df_sk_geral.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupo', color='Grupo', color_discrete_map=cfg_sk_geral['cores_map'])
                            st.plotly_chart(estilizar_grafico_avancado(f_sk_geral, cfg_sk_geral, max_val_geral), use_container_width=True)

                    # --- ABAS DE LOCAIS INDIVIDUAIS ---
                    for k, loc in enumerate(locais_unicos): 
                        with abas[k+1]:
                            # Aviso se interação não for significativa
                            if p_int_conj >= 0.05:
                                st.warning(f"⚠️ Sem diferença significativa na interação, não é possível analisar {loc} separadamente.")
                            
                            df_loc = df_proc[df_proc[col_local] == loc]
                            res_loc = rodar_analise_individual(df_loc, [col_trat], col_resp, delineamento, col_bloco)
                            
                            # AQUI ESTÁ A MUDANÇA: MOSTRA A TABELA MESMO SE P >= 0.05
                            if res_loc['p_val'] >= 0.05:
                                st.warning(f"⚠️ Sem diferença significativa (Teste F) em {loc}.")
                            
                            # Calcula e Mostra SEMPRE (Incondicional)
                            meds_loc = df_loc.groupby(col_trat)[col_resp].mean()
                            reps_loc = df_loc.groupby(col_trat)[col_resp].count().mean()
                            df_tk_loc = tukey_manual_preciso(meds_loc, res_loc['mse'], res_loc['df_resid'], reps_loc, len(meds_loc))
                            st.dataframe(df_tk_loc.style.format({"Media": "{:.2f}"}))
                            
                            # Matriz de Desdobramento (Só faz sentido se houver diferença)
                            if res_loc['p_val'] < 0.05 and len(cols_trats) >= 2: 
                                st.markdown("---")
                                fl_loc = st.selectbox(f"Fator Linha ({loc})", cols_trats, key=f"mat_tk_l_{loc}_{col_resp}")
                                fc_loc = [f for f in cols_trats if f != fl_loc][0]
                                df_m_loc = gerar_dataframe_matriz_total(df_loc, fl_loc, fc_loc, tukey_manual_preciso, res_loc['mse'], res_loc['df_resid'])
                                st.dataframe(df_m_loc)

                    # --- ABA INTERAÇÃO ---
                    with abas[-1]: 
                        # Identifica os tratamentos para permitir edição de cor
                        trats_inter = sorted(df_proc[col_trat].unique())
                        
                        if p_int_conj < 0.05:
                            st.success("✅ Interação Significativa.")
                            st.markdown("#### Matriz: Local (Linha) x Tratamento (Coluna)")
                            df_m_conj = gerar_dataframe_matriz_total(df_proc, col_local, col_trat, tukey_manual_preciso, res_conj['mse'], res_conj['df_resid'])
                            st.dataframe(df_m_conj)
                            st.markdown("---")
                            df_inter = df_proc.groupby([col_trat, col_local])[col_resp].mean().reset_index()
                            
                            # CORREÇÃO AQUI: Passar os grupos (tratamentos) para o editor
                            cfg_int = mostrar_editor_grafico(f"int_{col_resp}", f"Interação: {col_resp}", col_local, col_resp, usar_cor_unica=False, grupos_sk=trats_inter)
                            f_i = px.line(df_inter, x=col_local, y=col_resp, color=col_trat, markers=True, color_discrete_map=cfg_int['cores_map'])
                            st.plotly_chart(estilizar_grafico_avancado(f_i, cfg_int), use_container_width=True)
                        else: 
                            st.warning("⚠️ Sem diferença significativa na interação.")
                            st.caption("Visualização exploratória (sem valor estatístico de desdobramento):")
                            df_inter = df_proc.groupby([col_trat, col_local])[col_resp].mean().reset_index()
                            
                            # CORREÇÃO AQUI TAMBÉM
                            cfg_int = mostrar_editor_grafico(f"int_ns_{col_resp}", f"Gráfico Exploratório (NS)", col_local, col_resp, usar_cor_unica=False, grupos_sk=trats_inter)
                            f_i = px.line(df_inter, x=col_local, y=col_resp, color=col_trat, markers=True, color_discrete_map=cfg_int['cores_map'])
                            st.plotly_chart(estilizar_grafico_avancado(f_i, cfg_int), use_container_width=True)
# ==============================================================================
# 🏁 FIM DO BLOCO 12
# ==============================================================================

# ==============================================================================
# 📂 BLOCO 13: Lógica de Fallback (Botões de Erro) e Encerramento
# ==============================================================================
                if analise_valida:
                    if transf_atual != "Nenhuma":
                        st.markdown("---"); st.markdown("### 🛡️ Solução Final: Análise Paramétrica (ANOVA)")
                        st.success(f"✅ **Transformação Eficaz!** Com **{transf_atual}**, os pressupostos foram atendidos ou a robustez da ANOVA permite prosseguir.")
                        if st.button("Voltar ao Original", key=f"reset_success_{col_resp_original}"):
                            set_transformacao(col_resp_original, "Nenhuma"); st.rerun()
                else:
                    st.markdown("---"); st.error("🚨 ALERTA ESTATÍSTICO GRAVE: ANOVA INVÁLIDA")
                    st.markdown("""
                    Como os dados não seguem a **Normalidade** e/ou **Homogeneidade** de forma crítica, a média e o desvio padrão perdem o sentido.
                    **NÃO USE A ANOVA (Teste F)** para tomar decisões, pois ela pode apresentar resultados falsos (falso positivo ou negativo).
                    
                    **O que fazer?**
                    1. Tente realizar a **Transformação dos Dados** nas opções abaixo.
                    2. Se o problema persistir, analise cada local individualmente usando testes Não-Paramétricos.
                    """)
                    
                    if transf_atual == "Nenhuma":
                        col_btn1, col_btn2 = st.columns([1, 4])
                        with col_btn1:
                            if st.button("🧪 Tentar Log10", key=f"btn_log_{col_resp_original}"):
                                set_transformacao(col_resp_original, "Log10"); st.rerun()
                        with col_btn2: st.caption("Clique para aplicar transformação Logarítmica apenas nesta variável.")

                    elif transf_atual == "Log10":
                        st.warning(f"A transformação **Log10** não resolveu o problema.")
                        col_btn1, col_btn2 = st.columns([1, 4])
                        with col_btn1:
                            if st.button("🌱 Tentar Raiz Quadrada", key=f"btn_sqrt_{col_resp_original}"):
                                set_transformacao(col_resp_original, "Raiz Quadrada (SQRT)"); st.rerun()
                        if st.button("Voltar ao Original", key=f"reset_log_{col_resp_original}"):
                            set_transformacao(col_resp_original, "Nenhuma"); st.rerun()

                    elif transf_atual == "Raiz Quadrada (SQRT)":
                        st.warning(f"A transformação **Raiz Quadrada** também não resolveu.")
                        st.markdown("### 🛡️ Solução Final: Estatística Não-Paramétrica")
                        key_np = f"show_np_{col_resp_original}"
                        if key_np not in st.session_state: st.session_state[key_np] = False
                        
                        if not st.session_state[key_np]:
                            if st.button("🛡️ Rodar Estatística Não-Paramétrica", key=f"btn_run_np_{col_resp_original}"):
                                st.session_state[key_np] = True; st.rerun()
                        else:
                            nome_np, p_np = calcular_nao_parametrico(df_proc, col_trat, col_resp, delineamento, col_bloco)
                            if p_np is not None:
                                st.success(f"Resultado do Teste de **{nome_np}**:")
                                if p_np < 0.05:
                                    st.metric(label="P-valor Não-Paramétrico", value=f"{p_np:.4f}", delta="↑ Significativo (Diferença Real)", delta_color="normal")
                                else:
                                    st.metric(label="P-valor Não-Paramétrico", value=f"{p_np:.4f}", delta="↓ Não Significativo (Iguais)", delta_color="inverse")
                                    st.error(f"""
                                    🚨 **Não houve variação significativa entre os tratamentos.** Aceita-se a Hipótese Nula ($H_0$).
                                    
                                    **O que isso significa na prática?**
                                    1.  **Não há 'Ganhador':** Estatisticamente, todos os tratamentos tiveram o mesmo desempenho. As diferenças numéricas na tabela são fruto do acaso.
                                    2.  **Pare aqui:** Você **não deve** tentar fazer testes de médias ou separar letras ("a", "b"). Todos são "a".
                                    3.  **O Valor do 'Não Significativo':** Esse resultado é valioso! Ele prova equivalência (ex: o produto barato funciona igual ao caro).
                                    
                                    **📝 Como relatar no seu trabalho:**
                                    _"Para a variável analisada, o teste de {nome_np} (aplicado devido à violação dos pressupostos da ANOVA) não detectou diferença significativa (p = {p_np:.4f}). Portanto, todos os genótipos apresentaram desempenho estatisticamente semelhante."_
                                    """)

                                st.markdown("---")
                                st.markdown("### 💡 Guia de Interpretação: Análise de Dados")
                                msg_guia_intro = "**Seus dados são válidos, apenas a 'régua' mudou.**\n\n1. **A Média morreu:** Em dados não-normais, use a **Mediana** e **Quartis**.\n2. **O Gráfico:** Use o **Boxplot** abaixo para visualizar a distribuição real."
                                if p_np >= 0.05: msg_guia_conclusao = "\n3. **Conclusão:** Use a tabela e o gráfico abaixo para demonstrar que as medianas são visualmente próximas ou se sobrepõem."
                                else: msg_guia_conclusao = "\n3. **Conclusão:** Como houve diferença (P < 0.05), observe na tabela quem tem a maior Mediana para definir o superior."
                                st.info(msg_guia_intro + msg_guia_conclusao)
                                
                                st.markdown("### 📊 Dados para Relatório (Medianas e Postos)")
                                df_desc = df_proc.groupby(col_trat)[col_resp].agg(
                                    n='count', Mediana='median',
                                    Q1=lambda x: x.quantile(0.25), Q3=lambda x: x.quantile(0.75),
                                    Min='min', Max='max'
                                ).sort_values('Mediana', ascending=False)
                                st.dataframe(df_desc.style.format("{:.2f}"))
                                st.caption("Use esta tabela para descrever seus resultados no artigo/trabalho.")
                                
                                st.markdown("### 📉 Recomendação Visual: Boxplot")
                                fig_box = px.box(df_proc, x=col_trat, y=col_resp, points="all", title=f"Distribuição Real: {col_resp}")
                                st.plotly_chart(fig_box, use_container_width=True)

                            else: st.error("Não foi possível calcular o teste não-paramétrico (verifique dados faltantes ou delineamento).")
                            
                            if st.button("Ocultar Resultado", key=f"btn_hide_np_{col_resp_original}"):
                                st.session_state[key_np] = False; st.rerun()
                        
                        if st.button("Voltar ao Original", key=f"reset_sqrt_{col_resp_original}"):
                            set_transformacao(col_resp_original, "Nenhuma"); st.rerun()

else: st.info("👈 Faça upload do arquivo para começar.")
# ==============================================================================
# 🏁 FIM DO BLOCO 13
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 14: Planejamento (V6 - Identificação e Numeração Personalizada)
# ==============================================================================
import random
import pandas as pd
import itertools

if modo_app == "🎲 Planejamento (Sorteio)":
    st.title("🎲 Planejamento Experimental Pro")
    st.markdown("Gere sua planilha de campo com numeração personalizada e identificação do ensaio.")

    with st.form("form_planejamento"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### ⚙️ Design")
            tipo_exp = st.selectbox("Delineamento", ["DIC (Inteiramente Casualizado)", "DBC (Blocos Casualizados)"])
        with c2:
            st.markdown("#### 🔢 Repetições")
            n_reps = st.number_input("Nº de Repetições/Blocos", min_value=2, value=4)
        with c3:
            st.markdown("#### 📍 Identificação")
            nome_ensaio = st.text_input("Nome do Ensaio/Área", value="Ensaio_01")

        st.markdown("---")
        
        c_num1, c_num2 = st.columns([1, 2])
        with c_num1:
            num_inicial = st.number_input("Nº Inicial da Parcela", value=1, min_value=0)
        with c_num2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("A numeração das parcelas começará a partir deste valor.")

        st.markdown("---")
        
        tipo_entrada = st.radio("Como definir os tratamentos?", ["📝 Lista Simples", "✖️ Esquema Fatorial (A x B ...)"], horizontal=True)
        
        lista_trats_final = []
        if tipo_entrada == "📝 Lista Simples":
            txt_trats = st.text_area("Digite os Tratamentos (um por linha):", "Controle\nT1\nT2\nT3")
            if txt_trats:
                lista_trats_final = [t.strip() for t in txt_trats.split('\n') if t.strip()]
        else:
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                fator1_nome = st.text_input("Nome Fator 1", "Genotipo")
                fator1_niveis = st.text_area("Níveis Fator 1 (um por linha)", "G1\nG2\nG3")
            with c_f2:
                fator2_nome = st.text_input("Nome Fator 2 (Opcional)", "Dose")
                fator2_niveis = st.text_area("Níveis Fator 2 (um por linha)", "0%\n50%\n100%")
            
            if fator1_niveis:
                l1 = [x.strip() for x in fator1_niveis.split('\n') if x.strip()]
                l2 = [x.strip() for x in fator2_niveis.split('\n') if x.strip()] if fator2_niveis else []
                if l2:
                    combos = list(itertools.product(l1, l2))
                    lista_trats_final = [f"{a} + {b}" for a, b in combos]
                else:
                    lista_trats_final = l1

        st.markdown("---")
        st.markdown("#### 📝 Variáveis a Coletar")
        txt_vars = st.text_area("Cabeçalho da Planilha:", "Altura_cm\nProdutividade_kg")

        st.markdown("---")
        submitted = st.form_submit_button("🎲 Gerar Sorteio Oficial")

    if submitted:
        if not lista_trats_final:
            st.error("⚠️ Nenhum tratamento definido.")
        else:
            # Sorteio puramente aleatório (sem Seed fixa)
            parcelas = []
            info_blocos = []
            info_reps = [] 
            
            if "DIC" in tipo_exp:
                base_trats = lista_trats_final * n_reps
                random.shuffle(base_trats)
                parcelas = base_trats
                contadores = {t: 0 for t in lista_trats_final}
                for t in parcelas:
                    contadores[t] += 1
                    info_reps.append(contadores[t])
                info_blocos = ["-"] * len(parcelas) 

            else: # DBC
                for i in range(n_reps):
                    bloco = lista_trats_final.copy()
                    random.shuffle(bloco) 
                    parcelas.extend(bloco)
                    info_blocos.extend([f"Bloco {i+1}"] * len(bloco))
                    info_reps.extend([i+1] * len(bloco))
            
            # --- MONTAGEM DO DATAFRAME COM NUMERAÇÃO INICIAL ---
            total_sorteadas = len(parcelas)
            ids_personalizados = range(num_inicial, num_inicial + total_sorteadas)
            
            dados_planilha = {"ID_Parcela": ids_personalizados}
            
            if "DBC" in tipo_exp:
                dados_planilha["Bloco"] = info_blocos
            
            dados_planilha["Repeticao"] = info_reps
            dados_planilha["Tratamento"] = parcelas
            
            if tipo_entrada != "📝 Lista Simples" and len(parcelas) > 0 and " + " in str(parcelas[0]):
                try:
                    split_data = [str(t).split(" + ") for t in parcelas]
                    dados_planilha[f"Fator_{fator1_nome}"] = [x[0] for x in split_data]
                    dados_planilha[f"Fator_{fator2_nome}"] = [x[1] for x in split_data]
                except: pass 

            df_sorteio = pd.DataFrame(dados_planilha)
            
            lista_vars = [v.strip() for v in txt_vars.split('\n') if v.strip()]
            for var in lista_vars:
                df_sorteio[var] = "" 
            df_sorteio["Observacoes"] = ""

            st.success(f"✅ Sorteio Realizado para: {nome_ensaio}")
            st.markdown("### 📋 Planilha de Campo")
            st.dataframe(df_sorteio, use_container_width=True, hide_index=True)
            
            csv = df_sorteio.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Planilha (.csv)",
                data=csv,
                file_name=f"sorteio_{nome_ensaio}.csv",
                mime="text/csv"
            )
# ==============================================================================
# 🏁 FIM DO BLOCO 14
# ==============================================================================
