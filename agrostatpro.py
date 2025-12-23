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
    
    # Cálculo do QM (Quadrado Médio)
    if 'SQ' in df.columns and 'GL' in df.columns:
        df['QM'] = df['SQ'] / df['GL']
    
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
    
    if 'P-valor' in df.columns:
        df['Sig.'] = df['P-valor'].apply(verificar_sig)
    
    # --- CORREÇÃO: REORDENAÇÃO DAS COLUNAS (GL PRIMEIRO) ---
    ordem_desejada = ['GL', 'SQ', 'QM', 'Fcalc', 'P-valor', 'Sig.']
    # Filtra apenas as colunas que existem no dataframe para evitar erro
    cols_finais = [c for c in ordem_desejada if c in df.columns]
    df = df[cols_finais]
    
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
# 📂 BLOCO 03: Cálculo de Métricas e Relatórios de Texto (V2 - Régua Rigorosa)
# ==============================================================================
def calcular_metricas_extras(anova_df, modelo, col_trat):
    """Calcula métricas, define classes e retorna o valor F bruto para diagnóstico."""
    metrics = {
        'rmse': 0.0, 'r2': 0.0, 'acuracia': 0.0, 'h2': 0.0,
        'r2_class': "", 'ac_class': "N/A", 'h2_class': "N/A",
        'f_valor_bruto': 0.0 
    }
    
    try:
        metrics['rmse'] = np.sqrt(modelo.mse_resid)
        metrics['r2'] = modelo.rsquared
        
        # R2 Rigoroso: < 0.70 é considerado Baixo/Regular (Alerta)
        if metrics['r2'] >= 0.70: metrics['r2_class'] = "OK"
        else: metrics['r2_class'] = "🔴"

        f_calc = 0
        for idx in anova_df.index:
            idx_clean = str(idx).replace("C(", "").replace(")", "")
            if col_trat in idx_clean and ":" not in idx_clean: 
                try:
                    val = anova_df.loc[idx, "Fcalc"]
                    f_calc = float(val) if val != "-" else 0
                except: f_calc = 0
                break
        
        metrics['f_valor_bruto'] = f_calc

        if pd.isna(f_calc) or f_calc <= 1:
            metrics['acuracia'] = 0.0
            metrics['h2'] = 0.0
            metrics['ac_class'] = "🔴"
            metrics['h2_class'] = "🔴"
        else:
            metrics['acuracia'] = np.sqrt(1 - (1/f_calc))
            metrics['h2'] = 1 - (1/f_calc)
            
            # Régua Rigorosa: Só é OK se for Alta (>0.70 ou >0.50 para h2 dependendo do critério, 
            # mas vamos alinhar com a exigência de "Regular ser Ruim")
            if metrics['acuracia'] >= 0.70: metrics['ac_class'] = "OK"
            else: metrics['ac_class'] = "🔴"
            
            if metrics['h2'] >= 0.50: metrics['h2_class'] = "OK"
            else: metrics['h2_class'] = "🔴"
            
    except:
        metrics['ac_class'] = "Erro"
        metrics['h2_class'] = "Erro"
        
    return metrics

def gerar_relatorio_metricas(anova_df, modelo, col_trat, media_real, p_valor, razao_mse=None):
    """Gera texto explicativo com rigor estatístico."""
    rmse = np.sqrt(modelo.mse_resid)
    r2 = modelo.rsquared
    
    # 1. ANOVA
    if p_valor < 0.05:
        sig_txt = "🟢 Significativo (Há diferença estatística entre tratamentos)."
    else:
        sig_txt = "🔴 Não Significativo (Médias estatisticamente iguais)."

    # 2. R2 (Rigoroso)
    if r2 >= 0.90: r2_txt = "🟢 O modelo é excelente (Alta precisão)."
    elif r2 >= 0.70: r2_txt = "🟢 O modelo tem bom ajuste."
    elif r2 >= 0.50: r2_txt = "🔴 Ajuste Regular: Há muita variação não explicada (⚠️ Atenção)."
    else: r2_txt = "🔴 Baixo Ajuste: O modelo explica muito pouco (⚠️ Atenção)."

    # 3. CV
    cv_val = (rmse / media_real) * 100
    if cv_val < 10: cv_txt = "🟢 Baixo (Alta Precisão Experimental)."
    elif cv_val <= 20: cv_txt = "🟢 Médio (Boa Precisão)."
    elif cv_val <= 30: cv_txt = "🔴 Alto: Baixa Precisão (⚠️ Atenção)."
    else: cv_txt = "🔴 Muito Alto: Dados inconsistentes (⚠️ Atenção)."

    # 4. ACURÁCIA & H2 (Visualização Rigorosa: Regular = Vermelho)
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
            ac_txt = "🔴 Crítico: Variação genética não detectada (F ≤ 1). Seleção ineficaz (⚠️ Atenção)."
            h2_txt = "🔴 Crítico: Variância ambiental superou a genética (⚠️ Atenção)."
        else:
            acuracia = np.sqrt(1 - (1/f_calc))
            herdabilidade = 1 - (1/f_calc)
            
            # ACURÁCIA
            if acuracia >= 0.90: ac_txt = "🟢 Muito Alta: Excelente confiabilidade."
            elif acuracia >= 0.70: ac_txt = "🟢 Alta: Boa segurança na seleção."
            elif acuracia >= 0.50: ac_txt = "🔴 Regular: Seleção requer cautela (⚠️ Atenção)."
            else: ac_txt = "🔴 Baixa: Pouca confiança para selecionar (⚠️ Atenção)."
            
            # HERDABILIDADE
            if herdabilidade >= 0.70: h2_txt = "🟢 Alta magnitude: Forte controle genético."
            elif herdabilidade >= 0.50: h2_txt = "🟢 Média magnitude: Controle genético moderado."
            else: h2_txt = "🔴 Baixa magnitude: Forte influência ambiental (⚠️ Atenção)."
            
    except:
        acuracia, herdabilidade = 0, 0
        ac_txt = "⚠️ Não Estimável."
        h2_txt = "⚠️ Não Estimável."

    txt_media = formatar_numero(media_real)
    txt_cv = formatar_numero(cv_val)
    txt_ac = formatar_numero(acuracia)
    txt_h2 = formatar_numero(herdabilidade)
    txt_r2 = formatar_numero(r2)
    txt_rmse = formatar_numero(rmse)
    txt_p = formatar_numero(p_valor, decimais=4)

    texto = ""
    texto += f"- 📊 **Média Geral:** `{txt_media}` — Valor central.\n"
    texto += f"- ⚡ **CV (%):** `{txt_cv}%` — {cv_txt}\n"
    texto += f"- 🎯 **Acurácia Seletiva:** `{txt_ac}` — {ac_txt}\n"
    texto += f"- 🧬 **Herdabilidade (h²):** `{txt_h2}` — {h2_txt}\n"
    texto += f"- 📉 **R²:** `{txt_r2}` — {r2_txt}\n"
    texto += f"- 📏 **RMSE:** `{txt_rmse}` — Erro médio absoluto.\n"
    
    if razao_mse:
        razao_txt = "🟢 Homogêneo" if razao_mse < 7 else "🔴 Heterogêneo (⚠️ Atenção)"
        txt_razao = formatar_numero(razao_mse)
        texto += f"- ⚖️ **Razão MSE:** `{txt_razao}` — {razao_txt}\n"

    texto += f"- 🔍 **ANOVA (Genótipos):** `P={txt_p}` — {sig_txt}\n"

    return texto
# ==============================================================================
# 🏁 FIM DO BLOCO 03
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 04: Diagnóstico Visual, Transformações e Estilos Gráficos (V43 - RMarkdown Style)
# ==============================================================================
def gerar_tabela_diagnostico(p_shapiro, p_bartlett=None, p_levene=None):
    """
    Gera tabela em formato Markdown puro (Estilo RMarkdown/Pandoc).
    Isso cria uma visualização limpa, sem índices de dataframe.
    """
    # Definição do Cabeçalho e Alinhamento
    # | Esquerda | Centro | Centro | Esquerda |
    tabela = "| Teste Estatístico | P-valor | Resultado | Conclusão |\n"
    tabela += "| :--- | :---: | :---: | :--- |\n"
    
    # 1. Shapiro-Wilk
    if pd.isna(p_shapiro):
        p_txt, res_txt, conc_txt = "-", "NaN", "Ignorado ⚪"
    elif p_shapiro < 0.05:
        p_txt = f"{p_shapiro:.4f}"
        res_txt = "P < 0.05"
        conc_txt = "Rejeita H0 (**NÃO Normal**) ⚠️"
    else:
        p_txt = f"{p_shapiro:.4f}"
        res_txt = "P ≥ 0.05"
        conc_txt = "Aceita H0 (**Normal**) ✅"
    tabela += f"| **Shapiro-Wilk** | {p_txt} | {res_txt} | {conc_txt} |\n"
    
    # 2. Bartlett (Se houver)
    if p_bartlett is not None:
        if pd.isna(p_bartlett):
            p_txt, res_txt, conc_txt = "-", "NaN", "Ignorado ⚪"
        elif p_bartlett < 0.05:
            p_txt = f"{p_bartlett:.4f}"
            res_txt = "P < 0.05"
            conc_txt = "Rejeita H0 (**Heterogêneo**) ⚠️"
        else:
            p_txt = f"{p_bartlett:.4f}"
            res_txt = "P ≥ 0.05"
            conc_txt = "Aceita H0 (**Homogêneo**) ✅"
        tabela += f"| **Bartlett** | {p_txt} | {res_txt} | {conc_txt} |\n"

    # 3. Levene (Se houver)
    if p_levene is not None:
        if pd.isna(p_levene):
            p_txt, res_txt, conc_txt = "-", "NaN", "Ignorado ⚪"
        elif p_levene < 0.05:
            p_txt = f"{p_levene:.4f}"
            res_txt = "P < 0.05"
            conc_txt = "Rejeita H0 (**Heterogêneo**) ⚠️"
        else:
            p_txt = f"{p_levene:.4f}"
            res_txt = "P ≥ 0.05"
            conc_txt = "Aceita H0 (**Homogêneo**) ✅"
        tabela += f"| **Levene** | {p_txt} | {res_txt} | {conc_txt} |\n"
    
    return tabela

@st.cache_data(show_spinner=False)
def aplicar_transformacao(df, col_resp, tipo_transformacao):
    """Aplica transformação matemática nos dados (Cacheada)."""
    df_copy = df.copy()
    df_copy[col_resp] = limpar_e_converter_dados(df_copy, col_resp)
    
    nova_col = col_resp
    if tipo_transformacao == "Log10":
        nova_col = f"{col_resp}_Log"
        df_copy[nova_col] = np.log10(df_copy[col_resp].where(df_copy[col_resp] > 0, 1e-10))
    elif tipo_transformacao == "Raiz Quadrada (SQRT)":
        nova_col = f"{col_resp}_Sqrt"
        df_copy[nova_col] = np.sqrt(df_copy[col_resp].where(df_copy[col_resp] >= 0, 0))
        
    return df_copy, nova_col

# --- FUNÇÕES VISUAIS GLOBAIS ---
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
        # Configuração da Legenda para evitar texto invisível
        legend=dict(
            title=dict(text=f"<b>{configs['titulo_legenda']}</b>", font=dict(color=configs['cor_texto'])), 
            bgcolor=configs['cor_fundo'], 
            borderwidth=0,
            font=dict(color=configs['cor_texto']) 
        )
    )
    
    fig.update_traces(
        texttemplate=template_texto, 
        textposition=pos_texto_final, 
        textfont=dict(size=configs['font_size'], color=configs['cor_texto']), 
        cliponaxis=False, 
        marker_line_color=configs['cor_texto'], 
        marker_line_width=1.0,
        showlegend=configs['mostrar_legenda']
    )
    
    if configs.get('cor_barras'):
        fig.update_traces(marker_color=configs['cor_barras'])
    
    # Renomeação de Grupos (Correção de tipos string)
    if configs.get('mapa_nomes_grupos'):
        mapa_str = {str(k).strip(): str(v).strip() for k, v in configs['mapa_nomes_grupos'].items()}
        for trace in fig.data:
            nome_atual = str(trace.name).strip()
            if nome_atual in mapa_str:
                novo_nome = mapa_str[nome_atual]
                trace.name = novo_nome
                trace.legendgroup = novo_nome
                if trace.hovertemplate:
                    trace.hovertemplate = trace.hovertemplate.replace(nome_atual, novo_nome)
                
    return fig

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
    # ALTERAÇÃO: Mudado de 'Letras' para 'Grupos' e forçada a ordem das colunas
    df_res = pd.DataFrame({'Media': vals, 'Grupos': [letras[n] for n in nomes]}, index=nomes)
    return df_res[['Media', 'Grupos']]

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
        
    df_temp['Grupos'] = df_temp['LetraRaw'].map(mapa_final)
    # GARANTIA FINAL DE ORDEM: Media na esquerda, Grupos na direita
    return df_temp[['Media', 'Grupos']]

def explaining_ranking(df, method="Tukey"):
    return f"Nota: Médias seguidas pela mesma letra/grupo não diferem estatisticamente ({method} 5%)."

def analisar_regressao_polinomial(df, col_trat, col_resp):
    """
    Calcula regressão Linear e Quadrática para dados numéricos.
    Retorna dicionário com métricas e parâmetros para visualização.
    """
    from statsmodels.formula.api import ols
    
    # Filtra apenas colunas necessárias e remove NaNs
    df_reg = df[[col_trat, col_resp]].dropna()
    x_vals = df_reg[col_trat]
    
    # Se não houver dados suficientes, retorna vazio
    if len(df_reg) < 3:
        return {'Linear': None, 'Quad': None}, 0, 0
        
    x_min, x_max = x_vals.min(), x_vals.max()
    resultados = {'Linear': None, 'Quad': None}
    
    # 1. Modelo Linear
    try:
        modelo_lin = ols(f"{col_resp} ~ {col_trat}", data=df_reg).fit()
        resultados['Linear'] = {
            'r2': modelo_lin.rsquared,
            'p_val': modelo_lin.f_pvalue,
            'params': modelo_lin.params,
            'eq': f"y = {modelo_lin.params.get('Intercept', 0):.4f} + {modelo_lin.params.get(col_trat, 0):.4f}x"
        }
    except: pass

    # 2. Modelo Quadrático
    try:
        # A sintaxe I(...) protege a operação aritmética na fórmula
        modelo_quad = ols(f"{col_resp} ~ {col_trat} + I({col_trat}**2)", data=df_reg).fit()
        resultados['Quad'] = {
            'r2': modelo_quad.rsquared,
            'p_val': modelo_quad.f_pvalue,
            'params': modelo_quad.params,
            'eq': f"y = {modelo_quad.params.get('Intercept', 0):.4f} + {modelo_quad.params.get(col_trat, 0):.4f}x + {modelo_quad.params.get(f'I({col_trat} ** 2)', 0):.4f}x²"
        }
    except: pass
    
    return resultados, x_min, x_max
# ==============================================================================
# 🏁 FIM DO BLOCO 05
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 06: Funções Estatísticas Principais (V44 - Com Não-Paramétrica)
# ==============================================================================

def calcular_nao_parametrico(df, col_trat, col_resp, delineamento, col_bloco=None):
    """
    Executa testes não-paramétricos quando os pressupostos da ANOVA falham.
    - DIC: Kruskal-Wallis
    - DBC: Friedman
    """
    import scipy.stats as stats
    
    grupos = []
    trats = df[col_trat].unique()
    
    try:
        if delineamento == "DIC":
            # Prepara grupos para Kruskal-Wallis
            for t in trats:
                grupos.append(df[df[col_trat] == t][col_resp].values)
            stat, p_val = stats.kruskal(*grupos)
            nome_teste = "Kruskal-Wallis"
            
        else: # DBC
            # Prepara matriz para Friedman (Tratamentos x Blocos)
            # Pivotar para garantir ordem correta
            df_pivot = df.pivot(index=col_bloco, columns=col_trat, values=col_resp).dropna()
            
            # Friedman exige matriz completa sem NAs
            if df_pivot.empty:
                return "Erro (Dados Incompletos)", 1.0
            
            # Extrai colunas como arrays
            args = [df_pivot[col].values for col in df_pivot.columns]
            stat, p_val = stats.friedmanchisquare(*args)
            nome_teste = "Friedman"
            
        return nome_teste, p_val

    except Exception as e:
        log_message(f"Erro no teste não-paramétrico: {e}")
        return "Erro de Cálculo", 1.0

def tukey_manual_preciso(medias, mse, df_resid, r, n_trats):
    """Calcula o teste de Tukey e retorna DataFrame com letras."""
    from scipy.stats import studentized_range
    
    # Ordena médias (decrescente)
    medias_sorted = medias.sort_values(ascending=False)
    nomes = medias_sorted.index.tolist()
    vals = medias_sorted.values
    
    # DMS (Diferença Mínima Significativa)
    alpha = 0.05
    q_val = studentized_range.ppf(1 - alpha, n_trats, df_resid)
    dms = q_val * np.sqrt(mse / r)
    
    letras = {}
    letra_atual = 97 # 'a' em ASCII
    
    # Algoritmo de Agrupamento (Letras)
    cobriu = [False] * len(vals)
    
    for i in range(len(vals)):
        if not cobriu[i]:
            letra_char = chr(letra_atual)
            letras[nomes[i]] = letras.get(nomes[i], "") + letra_char
            cobriu[i] = True
            
            # Verifica quem mais entra no grupo desta média
            for j in range(i + 1, len(vals)):
                diff = vals[i] - vals[j]
                if diff < dms: # Não significativo -> Mesma letra
                    letras[nomes[j]] = letras.get(nomes[j], "") + letra_char
                    # Nota: Não marcamos cobriu[j] = True aqui porque J pode participar de outros grupos (overlap)
            
            letra_atual += 1

    # Formata retorno
    res_df = pd.DataFrame({
        'Media': vals,
        'Letras': [letras[n] for n in nomes]
    }, index=nomes)
    
    return res_df.sort_index()

def scott_knott(medias, mse, df_resid, r, n_trats):
    """
    Implementação simplificada e robusta do Scott-Knott.
    Agrupa médias minimizando a soma de quadrados dentro dos grupos.
    """
    from scipy.stats import f
    
    medias_sorted = medias.sort_values(ascending=False)
    vals = medias_sorted.values
    nomes = medias_sorted.index
    
    n = len(vals)
    grupos_indices = [[i for i in range(n)]] # Começa com um grupão
    
    # Função para calcular BO (Soma de Quadrados Entre Grupos)
    def calcular_bo(grupo_idx):
        if len(grupo_idx) < 2: return 0, -1
        
        melhor_bo = -1
        melhor_corte = -1
        
        # Tenta cortar em todos os pontos possíveis
        for i in range(1, len(grupo_idx)):
            g1 = vals[grupo_idx[:i]]
            g2 = vals[grupo_idx[i:]]
            
            n1, n2 = len(g1), len(g2)
            m1, m2 = np.mean(g1), np.mean(g2)
            mg = np.mean(np.concatenate([g1, g2]))
            
            bo = n1 * (m1 - mg)**2 + n2 * (m2 - mg)**2
            if bo > melhor_bo:
                melhor_bo = bo
                melhor_corte = i
        
        return melhor_bo, melhor_corte

    grupos_finais = []
    fila = [list(range(n))]
    
    while fila:
        grupo_atual = fila.pop(0)
        
        if len(grupo_atual) == 1:
            grupos_finais.append(grupo_atual)
            continue
            
        bo, corte = calcular_bo(grupo_atual)
        
        # Cálculo do Lambda (Estatística de Teste)
        sigma2 = mse / r
        # Aproximação Chi-Quadrado para Scott-Knott
        lambda_val = (np.pi / (2 * (np.pi - 2))) * (bo / sigma2)
        
        # Valor Crítico (Aproximação)
        v0 = n_trats / (np.pi - 2) # Graus de liberdade aproximados
        p_val = 1 - f.cdf(lambda_val, v0, df_resid) # Usando F como proxy robusto
        
        # Limiar empírico para SK (geralmente mais relaxado que F puro)
        if p_val < 0.05: # Há diferença, divide
            g1 = grupo_atual[:corte]
            g2 = grupo_atual[corte:]
            fila.insert(0, g2) # Processa depois
            fila.insert(0, g1) # Processa primeiro
        else:
            grupos_finais.append(grupo_atual)
    
    # Atribui letras aos grupos
    dic_res = {}
    letra_ascii = 97
    
    # Ordena grupos pela média (do maior para o menor já que vals está ordenado)
    # A lógica acima já processa g1 (maiores) antes de g2
    # Mas garantimos ordenação por média do grupo
    grupos_finais.sort(key=lambda idxs: np.mean(vals[idxs]), reverse=True)
    
    for grp in grupos_finais:
        letra = chr(letra_ascii)
        for idx in grp:
            nome_trat = nomes[idx]
            dic_res[nome_trat] = letra
        letra_ascii += 1
        
    df_res = pd.DataFrame.from_dict(dic_res, orient='index', columns=['Grupo'])
    df_res['Media'] = medias
    df_res = df_res.sort_values('Media', ascending=False)
    
    return df_res

def rodar_analise_individual(df_input, col_trat, col_resp, delineamento, col_bloco=None):
    """Roda ANOVA Individual (DIC ou DBC) usando OLS."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from scipy import stats
    
    # Limpeza e Conversão
    df_f = df_input.dropna(subset=[col_resp]).copy()
    
    # Definição do Modelo
    if len(col_trat) > 1:
        # Fatorial: Trat = A * B -> 'A + B + A:B'
        formula_trat = " * ".join([f"C({c})" for c in col_trat])
    else:
        # Unifatorial
        formula_trat = f"C({col_trat[0]})"
        
    if delineamento == "DBC":
        formula = f"{col_resp} ~ {formula_trat} + C({col_bloco})"
    else: # DIC
        formula = f"{col_resp} ~ {formula_trat}"
        
    modelo = ols(formula, data=df_f).fit()
    anova_table = sm.stats.anova_lm(modelo, typ=2)
    
    # Teste F do Modelo Global (p_val)
    # Pega o p-valor do primeiro tratamento listado como proxy principal ou do modelo
    try:
        # Tenta pegar o p-valor do fator principal ou interação
        idx_p = [x for x in anova_table.index if ':' in x or 'C(' in x][0]
        p_val = anova_table.loc[idx_p, "PR(>F)"]
    except:
        p_val = 1.0

    # Pressupostos
    resid = modelo.resid
    w_stat, p_shapiro = stats.shapiro(resid)
    
    # Homogeneidade (Bartlett ou Levene)
    # Para fatorial, criamos um grupo único combinado
    if len(col_trat) > 1:
        grupos = df_f[col_trat].astype(str).agg('_'.join, axis=1)
    else:
        grupos = df_f[col_trat[0]]
        
    vals_grupos = [df_f[col_resp][grupos == g].values for g in grupos.unique()]
    
    # Bartlett (se normal) ou Levene (se não) - Aqui rodamos os dois para diagnóstico
    try:
        b_stat, p_bartlett = stats.bartlett(*vals_grupos)
    except: p_bartlett = np.nan
        
    try:
        l_stat, p_levene = stats.levene(*vals_grupos)
    except: p_levene = np.nan
        
    return {
        "modelo": modelo,
        "anova": anova_table,
        "shapiro": (w_stat, p_shapiro),
        "bartlett": (b_stat, p_bartlett),
        "levene": (l_stat, p_levene),
        "resid": resid,
        "p_val": p_val,
        "mse": modelo.mse_resid,
        "df_resid": modelo.df_resid
    }

def rodar_analise_conjunta(df_input, col_trat, col_resp, col_local, delineamento, col_bloco=None):
    """Roda ANOVA Conjunta."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from scipy import stats
    
    df_f = df_input.dropna(subset=[col_resp]).copy()
    
    # Modelo Conjunta: Resp ~ Trat + Local + Trat:Local (+ Bloco/Local se DBC)
    # Simplificação: Bloco aninhado em Local -> C(Local):C(Bloco)
    
    form_base = f"{col_resp} ~ C({col_trat}) * C({col_local})"
    
    if delineamento == "DBC":
        # Bloco dentro de Local
        form_base += f" + C({col_local}):C({col_bloco})"
        
    modelo = ols(form_base, data=df_f).fit()
    anova_table = sm.stats.anova_lm(modelo, typ=2)
    
    # P-valores Chave
    try:
        p_interacao = anova_table.loc[f"C({col_trat}):C({col_local})", "PR(>F)"]
    except: p_interacao = 1.0
    
    try:
        p_trat = anova_table.loc[f"C({col_trat})", "PR(>F)"]
    except: p_trat = 1.0
    
    # Pressupostos (Resíduos Globais)
    resid = modelo.resid
    w_stat, p_shapiro = stats.shapiro(resid)
    
    # Homogeneidade Global (Tratamento Agrupado)
    vals_grupos = [df_f[col_resp][df_f[col_trat] == t].values for t in df_f[col_trat].unique()]
    try: b_stat, p_bartlett = stats.bartlett(*vals_grupos)
    except: p_bartlett = np.nan
    try: l_stat, p_levene = stats.levene(*vals_grupos)
    except: p_levene = np.nan
        
    return {
        "modelo": modelo,
        "anova": anova_table,
        "shapiro": (w_stat, p_shapiro),
        "bartlett": (b_stat, p_bartlett),
        "levene": (l_stat, p_levene),
        "p_interacao": p_interacao,
        "p_trat": p_trat,
        "mse": modelo.mse_resid,
        "df_resid": modelo.df_resid
    }
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
# 📂 BLOCO 08: Interface - Setup e CSS (V33 - Uploader Full Width MAX)
# ==============================================================================
st.set_page_config(page_title="AgroStat Pro", page_icon="🌱", layout="wide")

# --- FUNÇÃO CSS PARA ESTILOS GERAIS E CORREÇÕES ---
def configurar_estilo_abas():
    log_message("🎨 Aplicando estilos CSS ROBUSTOS...")
    st.markdown("""
        <style>
            /* 1. Estilo para Rolagem de Abas (Tabs) */
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
            
            /* 2. Customização do Uploader (Botão LARGURA TOTAL) */
            /* Afeta o container interno (retângulo preto) */
            [data-testid="stFileUploader"] section {
                padding: 1rem !important; /* Garante respiro */
                align-items: stretch !important; /* Força itens a esticar */
            }
            
            /* Força o botão interno a ocupar 100% da largura disponível */
            [data-testid="stFileUploader"] button {
                width: 100% !important;
                max-width: 100% !important;
                display: block !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
            }
            
            /* Garante que o texto 'Drag and drop' continue centralizado */
            [data-testid="stFileUploader"] section > div:first-child {
                text-align: center !important;
                margin-bottom: 10px !important;
            }

            /* 3. Personalização da Barra de Rolagem */
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
# 📂 BLOCO 09: Interface - Sidebar (V25 - Botão Centralizado/Expandido)
# ==============================================================================
# Substituí a imagem externa (quebrava) por um título nativo robusto
st.sidebar.markdown("# 🌾 AgroStat Pro") 

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
        
        # --- Rótulo mais rígido ---
        OPCAO_PADRAO = "Local Único (Análise Individual)" 
        col_local = st.sidebar.selectbox("Coluna de Local/Ambiente", [OPCAO_PADRAO] + [c for c in colunas if c not in cols_trats], on_change=reset_analise)
        
        col_bloco = None
        cols_ocupadas = cols_trats + [col_local]
        
        if delineamento == "DBC":
            col_bloco = st.sidebar.selectbox("Blocos (Repetições)", [c for c in colunas if c not in cols_ocupadas], on_change=reset_analise)
            cols_ocupadas.append(col_bloco)
        else:
            # --- Remoção do (Automático) e (Opcional) ---
            col_rep_dic = st.sidebar.selectbox("Coluna de Repetição", [c for c in colunas if c not in cols_ocupadas], on_change=reset_analise)
            cols_ocupadas.append(col_rep_dic)

        lista_resps = st.sidebar.multiselect("Variáveis Resposta (Selecione 1 ou mais)", [c for c in colunas if c not in cols_ocupadas], on_change=reset_analise)

        # Detecção de Modo
        modo_analise = "INDIVIDUAL"
        if col_local != OPCAO_PADRAO:
            n_locais = len(df[col_local].unique())
            if n_locais > 1:
                modo_analise = "CONJUNTA"
            else:
                st.sidebar.warning("⚠️ Coluna de Local selecionada, mas há apenas 1 local. Rodando modo Individual.")
        
        st.sidebar.markdown("---")

        # --- BOTÃO DE AÇÃO (CENTRALIZADO/EXPANDIDO) ---
        if st.sidebar.button("🚀 Rodar Dados!", type="primary", use_container_width=True):
            st.session_state['processando'] = True

        # --- EDITOR DE RÓTULOS (MOVIDO PARA BAIXO) ---
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
        with st.sidebar.expander("🔧 Manutenção / Cache"):
            if st.button("🧹 Limpar Memória", use_container_width=True):
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
# 📂 BLOCO 10: Execução, Alertas Rigorosos e Tabelas (V26 - Ordem Visual Ajustada)
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

        # --- AVISO DE MODO CONJUNTA (POSICIONADO NO TOPO DA PÁGINA PRINCIPAL) ---
        if modo_analise == "CONJUNTA":
            n_locais_detectados = len(df[col_local].unique())
            st.info(f"🌍 **Modo Conjunta Ativado!** ({n_locais_detectados} locais)")

        st.markdown(f"### 📋 Resultados: {len(lista_resps)} variáveis processadas")
        
        # --- 0.1 ANÁLISE DE DIMENSÕES (LOGS) ---
        dimensoes = []
        for f in cols_trats:
            n_niveis = df[f].nunique()
            dimensoes.append(str(n_niveis))
        
        esquema_txt = "x".join(dimensoes)
        if len(cols_trats) > 1:
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
                
                # --- 1. EXECUÇÃO DOS CÁLCULOS ESTATÍSTICOS ---
                res_analysis = {}
                p_shap, p_bart, p_lev = None, None, None
                res_model = None
                anova_tab = None
                extras = {} 
                p_final_trat = 1.0
                modo_atual_txt = ""

                if modo_analise == "INDIVIDUAL":
                    modo_atual_txt = "INDIVIDUAL"
                    res = rodar_analise_individual(df_proc, cols_trats, col_resp, delineamento, col_bloco)
                    res_analysis = res
                    p_shap, p_bart, p_lev = res['shapiro'][1], res['bartlett'][1], res['levene'][1]
                    res_model = res['modelo']
                    anova_tab = formatar_tabela_anova(res['anova'])
                    
                    # --- CORREÇÃO DE SINCRONIA: Busca P-valor pelo NOME da coluna ---
                    # Isso garante que o P-valor das Métricas seja O MESMO da Tabela
                    try:
                        # Procura o índice que contem o nome do tratamento (ex: 'Cultivar')
                        nome_trat_alvo = cols_trats[0]
                        idx_alvo = [x for x in res['anova'].index if nome_trat_alvo in str(x) and ":" not in str(x)][0]
                        p_final_trat = res['anova'].loc[idx_alvo, 'PR(>F)']
                    except:
                        # Se falhar a busca por nome, pega o primeiro (fallback)
                        p_final_trat = res['anova'].iloc[0]['PR(>F)']
                        
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
                    
                    # --- CORREÇÃO DE SINCRONIA (Conjunta) ---
                    try:
                         idx_trat = [x for x in res_conj['anova'].index if col_combo in str(x) and ":" not in str(x)][0]
                         p_final_trat = res_conj['anova'].loc[idx_trat, 'PR(>F)']
                    except:
                         p_final_trat = res_conj['p_trat']

                    extras = calcular_metricas_extras(anova_tab, res_model, col_combo)
                    st.markdown("#### 📝 Métricas Estatísticas")
                    txt_metrics = gerar_relatorio_metricas(anova_tab, res_model, col_combo, df_proc[col_resp].mean(), p_final_trat, razao)
                    st.markdown(txt_metrics)
                    if razao and razao > 7: 
                        st.error(f"⚠️ **Violação de Homogeneidade (MSE):** Razão {razao:.2f} > 7. A variância entre os locais é muito discrepante.")

                # ==============================================================================
                # 🛡️ GUARDIÃO DE INTEGRIDADE DOS ALERTAS (AI: NÃO ALTERAR OS TEXTOS ABAIXO!)
                # ATENÇÃO: As mensagens de st.success, st.error, st.warning e st.info abaixo
                # foram validadas cientificamente. É PROIBIDO simplificar, resumir ou alterar
                # a fraseologia destes alertas. Mantenha o rigor estatístico definido.
                # ==============================================================================
                
                cv_val = (np.sqrt(res_model.mse_resid)/df_proc[col_resp].mean())*100
                
                # A) CV (Coeficiente de Variação)
                if cv_val <= 20: 
                    st.success(f"✅ **CV Adequado ({cv_val:.2f}%):** Boa precisão experimental.")
                else: 
                    st.error(f"⚠️ **CV Alto ({cv_val:.2f}%):** Baixa precisão experimental. Atenção à desuniformidade.")

                # B) ACURÁCIA
                if "🔴" in extras['ac_class']: 
                    st.error(f"⚠️ **Acurácia Baixa ({extras['acuracia']:.2f}):** A confiabilidade para selecionar genótipos é baixa.")
                else:
                    st.success(f"✅ **Acurácia Alta ({extras['acuracia']:.2f}):** Excelente confiabilidade para seleção.")

                # C) HERDABILIDADE
                if "🔴" in extras['h2_class']:
                    st.error(f"⚠️ **Herdabilidade Baixa ({extras['h2']:.2f}):** Forte influência ambiental sobre a genética.")
                else:
                    st.success(f"✅ **Herdabilidade Alta ({extras['h2']:.2f}):** A maior parte da variação é genética.")

                # D) NOTA PEDAGÓGICA (MovidA para cá - Imediatamente abaixo da Herdabilidade)
                if p_final_trat >= 0.05:
                    if "🔴" in extras['ac_class'] or "🔴" in extras['h2_class']:
                        st.info("💡 **Nota de Interpretação:** Você viu alertas vermelhos de Acurácia/Herdabilidade acima? **Fique tranquilo.** Como o Teste F não detectou diferença significativa (P ≥ 0.05), é matematicamente esperado que esses índices sejam baixos ou zero, pois não há variância genética 'sobrando' para calculá-los.")

                # E) R2 (MovidO para baixo da Nota)
                if extras['r2'] < 0.50:
                    st.error(f"⚠️ **R² Baixo ({extras['r2']:.2f}):** O modelo explica menos de 50% da variação total.")
                elif extras['r2'] < 0.70:
                    st.warning(f"⚠️ **R² Regular ({extras['r2']:.2f}):** O modelo explica pouco da variação total (Atenção).")
                else:
                    st.success(f"✅ **R² Bom ({extras['r2']:.2f}):** O modelo apresenta um bom ajuste aos dados.")

                # --- EXIBIÇÃO FINAL DO RESULTADO ANOVA (TOPO) ---
                st.markdown("---")
                st.markdown("### 📊 Análise de Variância (ANOVA)")
                
                st.dataframe(anova_tab)

                # ALERTA PRINCIPAL (EMBAIXO DA TABELA, COM O MESMO P)
                if p_final_trat < 0.05: 
                    st.success(f"✅ **Diferença Significativa (P = {p_final_trat:.4e}).** Rejeita-se a Hipótese Nula (H0).")
                else: 
                    st.error(f"⚠️ **Não Significativo (P = {p_final_trat:.4f}).** Aceita-se H0 (Médias estatisticamente iguais).")

                # --- ALERTAS ESPECÍFICOS DA CONJUNTA (ABAIXO DA TABELA) ---
                if modo_atual_txt == "CONJUNTA":
                      # 1. Interação GxA
                      p_int = res_conj.get('p_interacao', 1.0)
                      if p_int < 0.05: st.error(f"⚠️ **Interação GxA Significativa (P={p_int:.4f}).**")
                      else: st.success(f"✅ **Interação GxA Não Significativa.**")
                      
                      # 2. Tratamento (Geral) - Solicitado: Amarelo se NS
                      if p_final_trat < 0.05:
                          st.success(f"✅ **Efeito de Tratamento Significativo (P={p_final_trat:.4f}).**")
                      else:
                          st.warning(f"⚠️ **Efeito de Tratamento Não Significativo (P={p_final_trat:.4f}).** Médias estatisticamente iguais na análise conjunta.")

                st.markdown("---")
                st.markdown("#### 🩺 Diagnóstico dos Pressupostos")
                st.markdown(gerar_tabela_diagnostico(p_shap, p_bart, p_lev))
                
                # ==============================================================================
                # 🛡️ GUARDIÃO DE INTEGRIDADE DOS PRESSUPOSTOS (AI: NÃO ALTERAR OS TEXTOS!)
                # ==============================================================================
                
                analise_valida = False
                is_nan_shap = pd.isna(p_shap); is_nan_bart = pd.isna(p_bart); is_nan_lev = pd.isna(p_lev)
                normal_ok = (p_shap >= 0.05) if not is_nan_shap else False
                bart_ok = (p_bart >= 0.05) if not is_nan_bart else False
                lev_ok = (p_lev >= 0.05) if not is_nan_lev else False

                if is_nan_shap:
                    st.error("⚠️ Erro de Cálculo (NaN) nos testes. Verifique seus dados (variância zero?).")
                elif normal_ok:
                    if bart_ok and lev_ok:
                        st.success("✅ **Pressupostos Atendidos:** Normalidade (Shapiro) e Homogeneidade (Bartlett/Levene) OK. Pode confiar na ANOVA.")
                        analise_valida = True
                    elif bart_ok or lev_ok:
                        st.success("✅ **Pressupostos Aceitáveis:** Normalidade OK e pelo menos um teste de Homogeneidade OK.")
                        analise_valida = True
                    else:
                        st.error("⚠️ **Violação de Homogeneidade:** Dados normais, mas variâncias heterogêneas. Transforme os dados.")
                else:
                    if bart_ok or lev_ok:
                        st.error("⚠️ **Violação de Normalidade:** Variâncias homogêneas, mas dados NÃO normais. Transforme os dados.")
                    else:
                        st.error("⚠️ **Violação Crítica:** Nem normalidade, nem homogeneidade. ANOVA inválida.")

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

                # ==============================================================================
                # 🛡️ GUARDIÃO DE INTEGRIDADE DOS PRESSUPOSTOS (AI: NÃO ALTERAR OS TEXTOS!)
                # A lógica de decisão e as frases abaixo são fixas. NÃO REESCREVA.
                # ==============================================================================
               
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
# 📂 BLOCO 12: Visualização Completa (V42 - Único Responsável pelos Gráficos)
# ==============================================================================
                # --- FUNÇÃO INTERNA: GERADOR DE MATRIZ DE DESDOBRAMENTO ---
                def gerar_dataframe_matriz_total(df_input, f_linha, f_coluna, metodo_func, mse_global, df_res_global):
                    log_message(f"🚀 Iniciando desdobramento: {f_linha} x {f_coluna}...")
                    dict_upper = {}
                    niveis_l = df_input[f_linha].unique()
                    for nl in niveis_l:
                        df_s = df_input[df_input[f_linha] == nl]
                        meds = df_s.groupby(f_coluna)[col_resp].mean()
                        reps = df_s.groupby(f_coluna)[col_resp].count().mean()
                        res_comp = metodo_func(meds, mse_global, df_res_global, reps, len(meds))
                        for nc, row in res_comp.iterrows():
                            # Pega a segunda coluna (Grupos)
                            letra_val = row.iloc[1] 
                            dict_upper[(str(nl), str(nc))] = str(letra_val).upper()

                    dict_lower = {}
                    niveis_c = df_input[f_coluna].unique()
                    for nc in niveis_c:
                        df_s = df_input[df_input[f_coluna] == nc]
                        meds = df_s.groupby(f_linha)[col_resp].mean()
                        reps = df_s.groupby(f_linha)[col_resp].count().mean()
                        res_comp = metodo_func(meds, mse_global, df_res_global, reps, len(meds))
                        for nl, row in res_comp.iterrows():
                             # Pega a segunda coluna (Grupos)
                            letra_val = row.iloc[1] 
                            dict_lower[(str(nl), str(nc))] = str(letra_val).lower()

                    pivot = df_input.pivot_table(index=f_linha, columns=f_coluna, values=col_resp, aggfunc='mean')
                    df_matriz = pivot.copy().astype(object)
                    for l in pivot.index:
                        for c in pivot.columns:
                            val = pivot.loc[l, c]
                            u = dict_upper.get((str(l), str(c)), "?")
                            low = dict_lower.get((str(l), str(c)), "?")
                            df_matriz.loc[l, c] = f"{val:.2f} {u} {low}"
                    
                    df_matriz.index.name = f_linha
                    cols_atuais = df_matriz.columns.tolist()
                    multi_cols = pd.MultiIndex.from_product([[f_coluna], cols_atuais])
                    df_matriz.columns = multi_cols
                    
                    styler = df_matriz.style.set_properties(**{'text-align': 'center'}).set_table_styles([
                        {'selector': 'th', 'props': [('text-align', 'center !important'), ('vertical-align', 'middle !important')]},
                        {'selector': 'th.col_heading', 'props': [('text-align', 'center !important')]},
                        {'selector': 'th.index_name', 'props': [('text-align', 'center !important')]},
                        {'selector': 'td', 'props': [('text-align', 'center !important'), ('vertical-align', 'middle !important')]}
                    ])
                    return styler

                # ----------------------------------------------------------
                # LÓGICA DE VISUALIZAÇÃO (SÓ RODA SE A ANÁLISE FOR VÁLIDA)
                # ----------------------------------------------------------
                if analise_valida:
                    
                    # ----------------------------------------------------------
                    # CENÁRIO A: ANÁLISE INDIVIDUAL
                    # ----------------------------------------------------------
                    if modo_analise == "INDIVIDUAL":
                        # Verifica se o tratamento é numérico
                        eh_numerico = False
                        try:
                            pd.to_numeric(df_proc[col_trat], errors='raise')
                            eh_numerico = True
                        except: eh_numerico = False

                        # Definição das Abas
                        titulos_abas = []
                        if eh_numerico: titulos_abas.append("📈 Regressão")
                        titulos_abas.extend(["📦 Teste de Tukey", "📦 Teste de Scott-Knott", "📊 Gráficos Barras"])
                        
                        tabs_ind = st.tabs(titulos_abas)
                        idx_aba = 0

                        medias_ind = df_proc.groupby(col_trat)[col_resp].mean()
                        reps_ind = df_proc.groupby(col_trat)[col_resp].count().mean()
                        n_trats_ind = len(medias_ind)
                        max_val_ind = medias_ind.max()

                        # --- ABA REGRESSÃO ---
                        if eh_numerico:
                            with tabs_ind[idx_aba]:
                                st.markdown(f"#### Análise de Regressão: {col_trat} vs {col_resp}")
                                res_reg, x_min, x_max = analisar_regressao_polinomial(df_proc, col_trat, col_resp)
                                
                                modelo_escolhido = None
                                nome_modelo = ""
                                if res_reg['Quad'] and res_reg['Quad']['p_val'] < 0.05:
                                    modelo_escolhido = res_reg['Quad']; nome_modelo = "Quadrático (2º Grau)"
                                elif res_reg['Linear'] and res_reg['Linear']['p_val'] < 0.05:
                                    modelo_escolhido = res_reg['Linear']; nome_modelo = "Linear (1º Grau)"
                                else:
                                    st.warning("⚠️ Nenhum modelo de regressão foi significativo. Sugere-se usar médias.")
                                    if res_reg['Linear']: modelo_escolhido = res_reg['Linear']; nome_modelo = "Linear (Não Sig.)"
                                
                                if modelo_escolhido:
                                    st.success(f"🏆 Melhor Ajuste: **{nome_modelo}**")
                                    st.latex(modelo_escolhido['eq'])
                                    st.metric("R²", f"{modelo_escolhido['r2']:.4f}")
                                    
                                    x_range = np.linspace(x_min, x_max, 100)
                                    params = modelo_escolhido['params']
                                    if "Quad" in nome_modelo:
                                        y_pred = params['Intercept'] + params[col_trat]*x_range + params[f"I({col_trat} ** 2)"]*(x_range**2)
                                    else:
                                        y_pred = params['Intercept'] + params[col_trat]*x_range
                                        
                                    fig_reg = px.scatter(df_proc, x=col_trat, y=col_resp, title=f"Regressão: {col_resp}", opacity=0.6)
                                    fig_reg.add_scatter(x=x_range, y=y_pred, mode='lines', name=f'Ajuste {nome_modelo}')
                                    medias_df = medias_ind.reset_index()
                                    fig_reg.add_scatter(x=medias_df[col_trat], y=medias_df[col_resp], mode='markers', marker=dict(color='red', size=10, symbol='x'), name='Médias')
                                    st.plotly_chart(fig_reg, use_container_width=True, key=f"chart_reg_{col_resp}_{i}")
                            idx_aba += 1

                        # --- CÁLCULO DOS TESTES DE MÉDIAS ---
                        df_tukey_ind = tukey_manual_preciso(medias_ind, res['mse'], res['df_resid'], reps_ind, n_trats_ind)
                        df_sk_ind = scott_knott(medias_ind, res['mse'], res['df_resid'], reps_ind, n_trats_ind)
                        
                        # --- NORMALIZAÇÃO: Garante nome 'Grupos' e Ordem [Media, Grupos] ---
                        # 1. Renomeia se necessário (Compatibilidade com Cache antigo)
                        if 'Letras' in df_tukey_ind.columns: df_tukey_ind = df_tukey_ind.rename(columns={'Letras': 'Grupos'})
                        if 'Grupo' in df_sk_ind.columns: df_sk_ind = df_sk_ind.rename(columns={'Grupo': 'Grupos'})
                        if 'Letras' in df_sk_ind.columns: df_sk_ind = df_sk_ind.rename(columns={'Letras': 'Grupos'})
                        
                        # 2. FORÇA A ORDEM DAS COLUNAS (Media na Esquerda)
                        df_tukey_ind = df_tukey_ind[['Media', 'Grupos']]
                        df_sk_ind = df_sk_ind[['Media', 'Grupos']]

                        # ABA TUKEY
                        with tabs_ind[idx_aba]:
                            st.markdown("#### Ranking Geral (Tukey)")
                            st.dataframe(df_tukey_ind.style.format({"Media": "{:.2f}"}))
                            interacao_sig = (len(cols_trats) >= 2 and res['p_val'] < 0.05)
                            if interacao_sig:
                                st.markdown("---")
                                st.subheader("🔠 Matriz de Desdobramento (Tukey)")
                                fl_tk = st.selectbox("Fator na Linha", cols_trats, key=f"mat_tk_l_{col_resp}_{i}")
                                fc_tk = [f for f in cols_trats if f != fl_tk][0]
                                df_m_tk = gerar_dataframe_matriz_total(df_proc, fl_tk, fc_tk, tukey_manual_preciso, res['mse'], res['df_resid'])
                                st.dataframe(df_m_tk)
                        
                        # ABA SCOTT-KNOTT
                        with tabs_ind[idx_aba+1]:
                            st.markdown("#### Ranking Geral (Scott-Knott)")
                            st.dataframe(df_sk_ind.style.format({"Media": "{:.2f}"}))
                            if interacao_sig:
                                st.markdown("---")
                                st.subheader("🔠 Matriz de Desdobramento (Scott-Knott)")
                                fl_sk = st.selectbox("Fator na Linha", cols_trats, key=f"mat_sk_l_{col_resp}_{i}")
                                fc_sk = [f for f in cols_trats if f != fl_sk][0]
                                df_m_sk = gerar_dataframe_matriz_total(df_proc, fl_sk, fc_sk, scott_knott, res['mse'], res['df_resid'])
                                st.dataframe(df_m_sk)

                        # ABA GRÁFICOS BARRAS
                        with tabs_ind[idx_aba+2]:
                            sub_tabs_graf = st.tabs(["📊 Gráfico Tukey", "📊 Gráfico Scott-Knott"])
                            
                            with sub_tabs_graf[0]:
                                cfg_tk = mostrar_editor_grafico(f"tk_ind_{col_resp}_{i}", "Médias (Tukey)", col_trat, col_resp, usar_cor_unica=True)
                                # Atualizado para text='Grupos'
                                f_tk = px.bar(df_tukey_ind.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupos')
                                st.plotly_chart(estilizar_grafico_avancado(f_tk, cfg_tk, max_val_ind), use_container_width=True, key=f"chart_bar_tk_{col_resp}_{i}")
                            
                            with sub_tabs_graf[1]:
                                grps_sk = sorted(df_sk_ind['Grupos'].unique())
                                cfg_sk = mostrar_editor_grafico(f"sk_ind_{col_resp}_{i}", "Médias (Scott-Knott)", col_trat, col_resp, usar_cor_unica=False, grupos_sk=grps_sk)
                                f_sk = px.bar(df_sk_ind.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupos', color='Grupos', color_discrete_map=cfg_sk['cores_map'])
                                st.plotly_chart(estilizar_grafico_avancado(f_sk, cfg_sk, max_val_ind), use_container_width=True, key=f"chart_bar_sk_{col_resp}_{i}")

                    # ----------------------------------------------------------
                    # CENÁRIO B: ANÁLISE CONJUNTA
                    # ----------------------------------------------------------
                    else:
                        locais_unicos = sorted(df_proc[col_local].unique())
                        titulos_abas = ["📊 Média Geral"] + [f"📍 {loc}" for loc in locais_unicos] + ["📈 Interação"]
                        abas = st.tabs(titulos_abas)
                        p_int_conj = res_conj.get('p_interacao', 1.0)
                        
                        # --- ABA 0: MÉDIA GERAL ---
                        with abas[0]: 
                            if p_int_conj < 0.05:
                                st.warning("⚠️ Interação Significativa: A Média Geral pode mascarar o desempenho real nos locais.")
                            
                            medias_geral = df_proc.groupby(col_trat)[col_resp].mean()
                            reps_geral = df_proc.groupby(col_trat)[col_resp].count().mean()
                            max_val_geral = medias_geral.max()

                            df_tukey_geral = tukey_manual_preciso(medias_geral, res_conj['mse'], res_conj['df_resid'], reps_geral, len(medias_geral))
                            df_sk_geral = scott_knott(medias_geral, res_conj['mse'], res_conj['df_resid'], reps_geral, len(medias_geral))

                            # --- NORMALIZAÇÃO GERAL ---
                            if 'Letras' in df_tukey_geral.columns: df_tukey_geral = df_tukey_geral.rename(columns={'Letras': 'Grupos'})
                            if 'Grupo' in df_sk_geral.columns: df_sk_geral = df_sk_geral.rename(columns={'Grupo': 'Grupos'})
                            
                            df_tukey_geral = df_tukey_geral[['Media', 'Grupos']]
                            df_sk_geral = df_sk_geral[['Media', 'Grupos']]

                            sub_abas_geral = st.tabs(["📦 Tukey (Geral)", "📦 Scott-Knott (Geral)"])
                            
                            with sub_abas_geral[0]:
                                st.dataframe(df_tukey_geral.style.format({"Media": "{:.2f}"}))
                                cfg_tk_geral = mostrar_editor_grafico(f"tk_geral_{col_resp}_{i}", "Média Geral (Tukey)", col_trat, col_resp, usar_cor_unica=True)
                                f_tk_geral = px.bar(df_tukey_geral.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupos')
                                st.plotly_chart(estilizar_grafico_avancado(f_tk_geral, cfg_tk_geral, max_val_geral), use_container_width=True, key=f"chart_geral_tk_{col_resp}_{i}")

                            with sub_abas_geral[1]:
                                st.dataframe(df_sk_geral.style.format({"Media": "{:.2f}"}))
                                grps_sk_geral = sorted(df_sk_geral['Grupos'].unique())
                                cfg_sk_geral = mostrar_editor_grafico(f"sk_geral_{col_resp}_{i}", "Média Geral (Scott-Knott)", col_trat, col_resp, usar_cor_unica=False, grupos_sk=grps_sk_geral)
                                f_sk_geral = px.bar(df_sk_geral.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupos', color='Grupos', color_discrete_map=cfg_sk_geral['cores_map'])
                                st.plotly_chart(estilizar_grafico_avancado(f_sk_geral, cfg_sk_geral, max_val_geral), use_container_width=True, key=f"chart_geral_sk_{col_resp}_{i}")

                        # --- ABAS DE LOCAIS INDIVIDUAIS ---
                        for k, loc in enumerate(locais_unicos): 
                            with abas[k+1]:
                                if p_int_conj >= 0.05:
                                    st.warning(f"⚠️ Sem diferença na interação, a análise de {loc} é apenas ilustrativa.")
                                
                                df_loc = df_proc[df_proc[col_local] == loc]
                                res_loc = rodar_analise_individual(df_loc, [col_trat], col_resp, delineamento, col_bloco)
                                
                                if res_loc['p_val'] >= 0.05:
                                    st.warning(f"⚠️ Sem diferença significativa (Teste F) em {loc}.")
                                
                                meds_loc = df_loc.groupby(col_trat)[col_resp].mean()
                                reps_loc = df_loc.groupby(col_trat)[col_resp].count().mean()
                                max_val_loc = meds_loc.max()

                                df_tk_loc = tukey_manual_preciso(meds_loc, res_loc['mse'], res_loc['df_resid'], reps_loc, len(meds_loc))
                                df_sk_loc = scott_knott(meds_loc, res_loc['mse'], res_loc['df_resid'], reps_loc, len(meds_loc))
                                
                                # --- NORMALIZAÇÃO LOCAIS ---
                                if 'Letras' in df_tk_loc.columns: df_tk_loc = df_tk_loc.rename(columns={'Letras': 'Grupos'})
                                if 'Grupo' in df_sk_loc.columns: df_sk_loc = df_sk_loc.rename(columns={'Grupo': 'Grupos'})
                                
                                df_tk_loc = df_tk_loc[['Media', 'Grupos']]
                                df_sk_loc = df_sk_loc[['Media', 'Grupos']]

                                sub_abas_loc = st.tabs(["📊 Tukey", "🎨 Scott-Knott"])
                                
                                with sub_abas_loc[0]:
                                    st.dataframe(df_tk_loc.style.format({"Media": "{:.2f}"}))
                                    cfg_tk_loc = mostrar_editor_grafico(f"tk_loc_{loc}_{col_resp}_{i}", f"Médias {loc} (Tukey)", col_trat, col_resp, usar_cor_unica=True)
                                    f_tk_loc = px.bar(df_tk_loc.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupos')
                                    st.plotly_chart(estilizar_grafico_avancado(f_tk_loc, cfg_tk_loc, max_val_loc), use_container_width=True, key=f"chart_loc_tk_{loc}_{col_resp}_{i}")
                                
                                with sub_abas_loc[1]:
                                    st.dataframe(df_sk_loc.style.format({"Media": "{:.2f}"}))
                                    grps_sk_loc = sorted(df_sk_loc['Grupos'].unique())
                                    cfg_sk_loc = mostrar_editor_grafico(f"sk_loc_{loc}_{col_resp}_{i}", f"Médias {loc} (Scott-Knott)", col_trat, col_resp, usar_cor_unica=False, grupos_sk=grps_sk_loc)
                                    f_sk_loc = px.bar(df_sk_loc.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupos', color='Grupos', color_discrete_map=cfg_sk_loc['cores_map'])
                                    st.plotly_chart(estilizar_grafico_avancado(f_sk_loc, cfg_sk_loc, max_val_loc), use_container_width=True, key=f"chart_loc_sk_{loc}_{col_resp}_{i}")

                        # --- ABA INTERAÇÃO ---
                        with abas[-1]: 
                            trats_inter = sorted(df_proc[col_trat].unique())
                            if p_int_conj < 0.05:
                                st.success("✅ Interação Significativa.")
                                st.markdown("#### Matriz: Local (Linha) x Tratamento (Coluna)")
                                df_m_conj = gerar_dataframe_matriz_total(df_proc, col_local, col_trat, tukey_manual_preciso, res_conj['mse'], res_conj['df_resid'])
                                st.dataframe(df_m_conj)
                                st.markdown("---")
                                df_inter = df_proc.groupby([col_trat, col_local])[col_resp].mean().reset_index()
                                cfg_int = mostrar_editor_grafico(f"int_{col_resp}_{i}", f"Interação: {col_resp}", col_local, col_resp, usar_cor_unica=False, grupos_sk=trats_inter)
                                f_i = px.line(df_inter, x=col_local, y=col_resp, color=col_trat, markers=True, color_discrete_map=cfg_int['cores_map'])
                                st.plotly_chart(estilizar_grafico_avancado(f_i, cfg_int), use_container_width=True, key=f"chart_int_{col_resp}_{i}")
                            else: 
                                st.warning("⚠️ Sem diferença significativa na interação.")
                                st.caption("Visualização exploratória:")
                                df_inter = df_proc.groupby([col_trat, col_local])[col_resp].mean().reset_index()
                                cfg_int = mostrar_editor_grafico(f"int_ns_{col_resp}_{i}", f"Gráfico Exploratório (NS)", col_local, col_resp, usar_cor_unica=False, grupos_sk=trats_inter)
                                f_i = px.line(df_inter, x=col_local, y=col_resp, color=col_trat, markers=True, color_discrete_map=cfg_int['cores_map'])
                                st.plotly_chart(estilizar_grafico_avancado(f_i, cfg_int), use_container_width=True, key=f"chart_int_ns_{col_resp}_{i}")
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
# 📂 BLOCO 14: Planejamento (V9 - Correção UI: Inputs Reativos)
# ==============================================================================
import random
import pandas as pd
import itertools

if modo_app == "🎲 Planejamento (Sorteio)":
    st.title("🎲 Planejamento Experimental Pro")
    st.markdown("Gere sua planilha de campo com numeração personalizada e identificação do ensaio.")

    # --- CORREÇÃO: INPUTS DE ESTRUTURA FORA DO FORMULÁRIO (ATUALIZAÇÃO INSTANTÂNEA) ---
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
    
    # --- LÓGICA DE NUMERAÇÃO AVANÇADA ---
    st.markdown("#### 🏷️ Configuração de Numeração")
    c_num1, c_num2 = st.columns([1, 2])
    
    with c_num1:
        usar_salto = st.checkbox("Saltar numeração por Bloco?", value=False, help="Ex: Bloco 1 (101..), Bloco 2 (201..)")
        
    with c_num2:
        if usar_salto:
            salto_val = st.number_input("Valor do Salto (Multiplicador)", value=100, step=100, help="Ex: 100 gera 101, 201... | 1000 gera 1001, 2001...")
        else:
            num_inicial = st.number_input("Nº Inicial Sequencial", value=1, min_value=0, help="Numeração contínua: 1, 2, 3, 4...")

    st.markdown("---")
    
    # --- SELETOR DE MODO (FORA DO FORM PARA REATIVIDADE) ---
    tipo_entrada = st.radio("Como definir os tratamentos?", ["📝 Lista Simples", "✖️ Esquema Fatorial (A x B ...)"], horizontal=True)
    
    # --- FORMULÁRIO APENAS PARA DADOS (EVITA RECARREGAR ENQUANTO DIGITA) ---
    with st.form("form_dados_trats"):
        lista_trats_final = []
        
        # LOGICA VISUAL DENTRO DO FORM (Baseada no radio externo)
        if tipo_entrada == "📝 Lista Simples":
            txt_trats = st.text_area("Digite os Tratamentos (um por linha):", "Controle\nT1\nT2\nT3")
        else:
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                fator1_nome = st.text_input("Nome Fator 1", "Genotipo")
                fator1_niveis = st.text_area("Níveis Fator 1 (um por linha)", "G1\nG2\nG3")
            with c_f2:
                fator2_nome = st.text_input("Nome Fator 2 (Opcional)", "Dose")
                fator2_niveis = st.text_area("Níveis Fator 2 (um por linha)", "0%\n50%\n100%")

        st.markdown("---")
        st.markdown("#### 📝 Variáveis a Coletar")
        txt_vars = st.text_area("Cabeçalho da Planilha:", "Altura_cm\nProdutividade_kg")

        st.markdown("---")
        submitted = st.form_submit_button("🎲 Gerar Sorteio Oficial")

    # --- PROCESSAMENTO PÓS-SUBMIT ---
    if submitted:
        # 1. Processa Listas
        if tipo_entrada == "📝 Lista Simples":
             if txt_trats:
                lista_trats_final = [t.strip() for t in txt_trats.split('\n') if t.strip()]
        else:
             if fator1_niveis:
                l1 = [x.strip() for x in fator1_niveis.split('\n') if x.strip()]
                l2 = [x.strip() for x in fator2_niveis.split('\n') if x.strip()] if fator2_niveis else []
                if l2:
                    combos = list(itertools.product(l1, l2))
                    lista_trats_final = [f"{a} + {b}" for a, b in combos]
                else:
                    lista_trats_final = l1

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
                # info_blocos não é usado no DIC

            else: # DBC
                for i in range(n_reps):
                    bloco = lista_trats_final.copy()
                    random.shuffle(bloco) 
                    parcelas.extend(bloco)
                    info_blocos.extend([f"Bloco {i+1}"] * len(bloco))
                    # info_reps também não será usado na saída do DBC
            
            # --- GERAÇÃO DE IDs ---
            total_sorteadas = len(parcelas)
            
            if usar_salto:
                ids_personalizados = []
                n_trats_por_bloco = len(lista_trats_final)
                
                for i in range(total_sorteadas):
                    bloco_idx = i // n_trats_por_bloco
                    item_idx = (i % n_trats_por_bloco) + 1
                    novo_id = ((bloco_idx + 1) * salto_val) + item_idx
                    ids_personalizados.append(novo_id)
            else:
                ids_personalizados = range(num_inicial, num_inicial + total_sorteadas)
            
            # --- MONTAGEM DINÂMICA DO DATAFRAME ---
            dados_planilha = {"ID_Parcela": ids_personalizados}
            
            if "DBC" in tipo_exp:
                # DBC: Tem Bloco, NÃO tem Repetição
                dados_planilha["Bloco"] = info_blocos
            else:
                # DIC: Tem Repetição, NÃO tem Bloco
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
