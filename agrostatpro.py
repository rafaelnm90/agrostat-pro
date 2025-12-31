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
# 📂 BLOCO 02: Utilitários Básicos (Limpeza e Conversão)
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
# ==============================================================================
# 🏁 FIM DO BLOCO 02
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 03: Formatação de Tabelas e Classificações
# ==============================================================================
def formatar_tabela_anova(anova_df):
    cols_map = {'sum_sq': 'SQ', 'df': 'GL', 'F': 'Fcalc', 'PR(>F)': 'P-valor'}
    df = anova_df.rename(columns=cols_map)
    
    # Cálculo do QM (Quadrado Médio)
    if 'SQ' in df.columns and 'GL' in df.columns:
        df['QM'] = df['SQ'] / df['GL']
    
    if 'Intercept' in df.index: df = df.drop('Intercept')
        
    new_index = []
    for idx in df.index:
        nome = str(idx)
        nome = nome.replace('C(', '').replace(', Sum)', '').replace(')', '')
        nome = nome.replace(':', ' x ')
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
    
    ordem_desejada = ['GL', 'SQ', 'QM', 'Fcalc', 'P-valor', 'Sig.']
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

# --- NOVAS FUNÇÕES SIMPLIFICADAS (MENU + DADOS DE TEXTO) ---

def mostrar_editor_tabela(key_prefix):
    """Menu para incluir CV e Média no rodapé."""
    with st.expander("✏️ Personalizar Tabela (Rodapé e Dados)"):
        c1, c2 = st.columns(2)
        show_media = c1.checkbox("Incluir Média Geral no Rodapé", value=True, key=f"show_mean_{key_prefix}")
        show_cv = c2.checkbox("Incluir CV (%) no Rodapé", value=True, key=f"show_cv_{key_prefix}")
        return show_media, show_cv

def calcular_texto_rodape(media_geral, mse_resid, show_media, show_cv):
    """
    Retorna apenas os textos formatados para serem inseridos na nota.
    """
    txt_media = ""
    txt_cv = ""
    
    if show_media:
        txt_media = f"Média Geral: {media_geral:.2f}"
        
    if show_cv:
        if mse_resid > 0:
            cv_val = (np.sqrt(mse_resid) / media_geral) * 100
            txt_cv = f"CV (%): {cv_val:.2f}"
        else:
            txt_cv = "CV (%): -"
            
    return txt_media, txt_cv
# ==============================================================================
# 🏁 FIM DO BLOCO 03
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 04: Cálculo de Métricas e Relatórios de Texto
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
# 🏁 FIM DO BLOCO 04
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 05: Diagnóstico Visual e Transformações
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
# ==============================================================================
# 🏁 FIM DO BLOCO 05
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 06: Estilos Gráficos e Editor Visual
# ==============================================================================
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
# 🏁 FIM DO BLOCO 06
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 07: Motores Estatísticos (Tukey/Scott-Knott - Cache)
# ==============================================================================

# OTIMIZAÇÃO: O cálculo do Tukey é pesado. Salvamos o resultado em cache.
@st.cache_data(show_spinner=False)
def tukey_manual_preciso(medias, mse, df_resid, r, n_trats, decrescente=True):
    """Calcula Tukey (HSD) e retorna DataFrame pronto com letras."""
    # 1. Cálculo do Delta (DMS)
    q_critico = studentized_range.ppf(1 - 0.05, n_trats, df_resid)
    dms = q_critico * np.sqrt(mse / r)
    
    # 2. Ordenação (Respeita o critério do usuário)
    # Se decrescente=True (Maior é melhor), ascending=False
    medias_ord = medias.sort_values(ascending=not decrescente)
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
            diff = abs(referencia - vals[i]) # Diferença absoluta
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
def scott_knott(medias, mse, df_resid, r, n_trats, decrescente=True):
    """Algoritmo de Scott-Knott de agrupamento (Clusterização)."""
    # Ordenação respeitando critério do usuário
    medias_ord = medias.sort_values(ascending=not decrescente)
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
    # Re-mapeia as letras baseadas na média do grupo, respeitando a ordem decrescente/crescente
    df_temp = pd.DataFrame({'Media': valores, 'LetraRaw': [resultados[n] for n in nomes]}, index=nomes)
    
    # Agrupa por letra e tira a média para saber quem é "a", "b"...
    media_grupos = df_temp.groupby('LetraRaw')['Media'].mean().sort_values(ascending=not decrescente)
    
    mapa_final = {}
    for i, (letra_velha, _) in enumerate(media_grupos.items()):
        mapa_final[letra_velha] = get_letra_segura(i)
        
    df_temp['Grupos'] = df_temp['LetraRaw'].map(mapa_final)
    # GARANTIA FINAL DE ORDEM: Media na esquerda, Grupos na direita
    return df_temp[['Media', 'Grupos']]

def explaining_ranking(df, method="Tukey"):
    return f"Nota: Médias seguidas pela mesma letra/grupo não diferem estatisticamente ({method} 5%)."
# ==============================================================================
# 🏁 FIM DO BLOCO 07
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 08: Motores Estatísticos (Regressão)
# ==============================================================================
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
# 🏁 FIM DO BLOCO 08
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 09: Motores Estatísticos (Não-Paramétricos & Dunn com Correção de Empates)
# ==============================================================================
def calcular_nao_parametrico(df, col_trat, col_resp, delineamento, col_bloco=None):
    """
    Executa Kruskal-Wallis (DIC) ou Friedman (DBC).
    """
    import scipy.stats as stats
    
    trats = sorted(df[col_trat].unique())
    
    try:
        if delineamento == "DIC":
            grupos = [df[df[col_trat] == t][col_resp].values for t in trats]
            stat, p_val = stats.kruskal(*grupos)
            nome_teste = "Kruskal-Wallis"
            
        else: # DBC
            df_pivot = df.pivot(index=col_bloco, columns=col_trat, values=col_resp).dropna()
            if df_pivot.empty: return "Erro (Dados Incompletos)", 0, 1.0
            args = [df_pivot[col].values for col in df_pivot.columns]
            stat, p_val = stats.friedmanchisquare(*args)
            nome_teste = "Friedman"
            
        return nome_teste, stat, p_val

    except Exception:
        return "Erro de Cálculo", 0, 1.0

def calcular_posthoc_dunn(df, col_trat, col_resp):
    """
    Teste de Dunn OTIMIZADO:
    1. Usa correção de Empates (Tie Correction) - Crucial para dados repetidos.
    2. Usa ajuste de HOLM-BONFERRONI (Mais poderoso).
    """
    import scipy.stats as stats
    from itertools import combinations
    
    df_rank = df.copy()
    # Calcula ranks globais (com média para empates)
    df_rank['posto'] = df[col_resp].rank(method='average')
    
    trats = sorted(df[col_trat].unique())
    
    # Estatísticas dos Postos
    R_means = df_rank.groupby(col_trat)['posto'].mean()
    ns = df_rank.groupby(col_trat)['posto'].count()
    N = len(df)
    
    # --- CÁLCULO DO FATOR DE CORREÇÃO DE EMPATES (TIE CORRECTION) ---
    # Fórmula: 1 - (Sum(t^3 - t) / (N^3 - N))
    from collections import Counter
    valores = df[col_resp].values
    counts = Counter(valores)
    
    # Soma de (t^3 - t) para todos os grupos de empates
    T_sum = sum([cnt**3 - cnt for cnt in counts.values() if cnt > 1])
    
    if T_sum == 0:
        tie_correction = 1.0
    else:
        tie_correction = 1 - (T_sum / (N**3 - N))
    
    # Se tie_correction for 0 (impossível teoricamente se N>1, mas previne div/0)
    if tie_correction == 0: tie_correction = 1.0
        
    # Variância corrigida
    var_base = (N * (N + 1)) / 12.0
    
    # 1. Calcula P-valores brutos
    comparacoes = []
    for t1, t2 in combinations(trats, 2):
        n1, n2 = ns[t1], ns[t2]
        r1, r2 = R_means[t1], R_means[t2]
        diff = abs(r1 - r2)
        
        # Erro Padrão COM correção de empates
        # SE = sqrt( (N(N+1)/12 - T_fator) * (1/n1 + 1/n2) ) -> Aproximação robusta
        # Fórmula exata Dunn com ties:
        # Sigma = sqrt( ( (N(N+1)/12) - (Sum(t^3-t)/(12(N-1))) ) * (1/n1 + 1/n2) )
        
        term_ties = T_sum / (12.0 * (N - 1))
        var_corrected = (N * (N + 1) / 12.0) - term_ties
        
        se = np.sqrt( var_corrected * (1/n1 + 1/n2) )
        
        if se == 0: 
            z_val = 0
            p_raw = 1.0
        else:
            z_val = diff / se
            p_raw = 2 * (1 - stats.norm.cdf(z_val)) # Two-tailed
        
        comparacoes.append({'A': t1, 'B': t2, 'p_raw': p_raw})
        
    df_res = pd.DataFrame(comparacoes)
    if df_res.empty: return df_res
    
    # 2. Aplica Correção de HOLM (Step-Down)
    df_res = df_res.sort_values('p_raw')
    m = len(df_res)
    df_res['p_adj'] = 1.0
    
    for idx, row in enumerate(df_res.itertuples()):
        rank = idx + 1
        fator = m - rank + 1
        p_corrigido = row.p_raw * fator
        
        if idx > 0:
            p_corrigido = max(p_corrigido, df_res.iloc[idx-1]['p_adj'])
            
        df_res.at[row.Index, 'p_adj'] = min(p_corrigido, 1.0)
        
    return df_res

def gerar_letras_dunn(trats, df_comparacoes):
    """
    Algoritmo de Atribuição de Letras (Grafo de Conectividade).
    """
    if df_comparacoes.empty: return {t: "a" for t in trats}

    # Mapeia quem é igual a quem (p_adj > 0.05)
    iguais = {t: {t} for t in trats}
    
    for _, row in df_comparacoes.iterrows():
        if row['p_adj'] > 0.05: 
            iguais[row['A']].add(row['B'])
            iguais[row['B']].add(row['A'])
            
    # Algoritmo Greedy Clique Cover
    trats_ordenados = sorted(trats, key=lambda x: len(iguais[x]), reverse=True)
    
    candidatos = []
    for t in trats_ordenados:
        clique = {t}
        grupo = iguais[t]
        # Tenta expandir o clique
        for cand in grupo:
            if cand == t: continue
            # Só entra se for amigo de todos que já estão no clique
            if all(cand in iguais[membro] for membro in clique):
                clique.add(cand)
        if clique not in candidatos:
            candidatos.append(clique)

    # Limpa subconjuntos (ex: se tenho {A,B,C}, removo {A,B})
    candidatos.sort(key=len, reverse=True)
    cliques_finais = []
    for c in candidatos:
        eh_sub = False
        for aceito in cliques_finais:
            if c.issubset(aceito):
                eh_sub = True; break
        if not eh_sub: cliques_finais.append(c)

    # Atribui letras
    letras_map = {t: "" for t in trats}
    for i, clique in enumerate(cliques_finais):
        l = get_letra_segura(i)
        for t in clique: letras_map[t] += l
            
    # Ordena as letras
    for t in letras_map:
        letras_map[t] = "".join(sorted(letras_map[t]))
        
    return letras_map
# ==============================================================================
# 🏁 FIM DO BLOCO 09
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 10: Motores Estatísticos IV (Implementações Auxiliares)
# ==============================================================================
# Nota: Implementações alternativas de Tukey/Scott-Knott (Sem Cache)
# e outras funções auxiliares mantidas conforme script original.

def tukey_manual_preciso(medias, mse, df_resid, r, n_trats, decrescente=True):
    """
    Calcula o teste de Tukey e retorna DataFrame com letras.
    Usa algoritmo de 'Maximal Contiguous Subsequences' para evitar redundância.
    """
    from scipy.stats import studentized_range
    
    # 1. Ordenação Robusta
    medias_sorted = medias.sort_values(ascending=not decrescente)
    nomes = medias_sorted.index.tolist()
    vals = medias_sorted.values
    n = len(vals)
    
    # 2. DMS (Diferença Mínima Significativa)
    alpha = 0.05
    q_val = studentized_range.ppf(1 - alpha, n_trats, df_resid)
    dms = q_val * np.sqrt(mse / r)
    
    # 3. Identificação de Grupos (Cliques)
    grupos_indices = []
    ultimo_fim = -1
    
    for i in range(n):
        fim_atual = i
        for j in range(i + 1, n):
            if abs(vals[i] - vals[j]) < dms:
                fim_atual = j
            else:
                break 
        
        if fim_atual > ultimo_fim:
            grupos_indices.append(range(i, fim_atual + 1))
            ultimo_fim = fim_atual

    # 4. Atribuição de Letras
    letras_map = {idx: "" for idx in range(n)}
    
    for i, grp in enumerate(grupos_indices):
        letra = get_letra_segura(i)
        for idx in grp:
            letras_map[idx] += letra
            
    # 5. Montagem do DataFrame (GARANTIA DE ORDEM: MEDIA, GRUPOS)
    res_df = pd.DataFrame({
        'Media': vals,
        'Grupos': [letras_map[i] for i in range(n)]
    }, index=nomes)
    
    return res_df[['Media', 'Grupos']].sort_index()

def scott_knott(medias, mse, df_resid, r, n_trats, decrescente=True):
    """
    Implementação simplificada e robusta do Scott-Knott.
    Agrupa médias minimizando a soma de quadrados dentro dos grupos.
    """
    from scipy.stats import f
    
    medias_sorted = medias.sort_values(ascending=not decrescente)
    vals = medias_sorted.values
    nomes = medias_sorted.index
    
    n = len(vals)
    
    # Função para calcular BO (Soma de Quadrados Entre Grupos)
    def calcular_bo(grupo_idx):
        if len(grupo_idx) < 2: return 0, -1
        
        melhor_bo = -1
        melhor_corte = -1
        
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
        
        sigma2 = mse / r
        lambda_val = (np.pi / (2 * (np.pi - 2))) * (bo / sigma2)
        v0 = n_trats / (np.pi - 2)
        p_val = 1 - f.cdf(lambda_val, v0, df_resid)
        
        if p_val < 0.05: # Há diferença, divide
            g1 = grupo_atual[:corte]
            g2 = grupo_atual[corte:]
            fila.insert(0, g2)
            fila.insert(0, g1)
        else:
            grupos_finais.append(grupo_atual)
    
    # Atribui letras aos grupos
    dic_res = {}
    grupos_finais.sort(key=lambda idxs: np.mean(vals[idxs]), reverse=decrescente)
    
    for i, grp in enumerate(grupos_finais):
        letra = get_letra_segura(i)
        for idx in grp:
            nome_trat = nomes[idx]
            dic_res[nome_trat] = letra
        
    df_res = pd.DataFrame.from_dict(dic_res, orient='index', columns=['Grupos'])
    df_res['Media'] = medias
    df_res = df_res.sort_values('Media', ascending=not decrescente)
    
    # Sincronização de Empates
    medias_unicas = df_res['Media'].round(6).unique()
    for m in medias_unicas:
        mask = df_res['Media'].round(6) == m
        if mask.sum() > 1:
            primeira_letra = df_res.loc[mask, 'Grupos'].iloc[0]
            df_res.loc[mask, 'Grupos'] = primeira_letra

    # --- CORREÇÃO CRÍTICA: FORÇA A ORDEM DAS COLUNAS [MEDIA, GRUPOS] ---
    # Isso impede que o gerador de matriz pegue a média no lugar da letra
    return df_res[['Media', 'Grupos']]

def rodar_analise_individual(df_input, col_trat, col_resp, delineamento, col_bloco=None):
    """Roda ANOVA Individual (DIC ou DBC) usando OLS."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from scipy import stats
    
    df_f = df_input.dropna(subset=[col_resp]).copy()
    
    if len(col_trat) > 1:
        formula_trat = " * ".join([f"C({c})" for c in col_trat])
    else:
        formula_trat = f"C({col_trat[0]})"
        
    if delineamento == "DBC":
        formula = f"{col_resp} ~ {formula_trat} + C({col_bloco})"
    else: # DIC
        formula = f"{col_resp} ~ {formula_trat}"
        
    modelo = ols(formula, data=df_f).fit()
    anova_table = sm.stats.anova_lm(modelo, typ=2)
    
    try:
        idx_p = [x for x in anova_table.index if ':' in x or 'C(' in x][0]
        p_val = anova_table.loc[idx_p, "PR(>F)"]
    except:
        p_val = 1.0

    resid = modelo.resid
    w_stat, p_shapiro = stats.shapiro(resid)
    
    if len(col_trat) > 1:
        grupos = df_f[col_trat].astype(str).agg('_'.join, axis=1)
    else:
        grupos = df_f[col_trat[0]]
        
    vals_grupos = [df_f[col_resp][grupos == g].values for g in grupos.unique()]
    
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
    
    form_base = f"{col_resp} ~ C({col_trat}) * C({col_local})"
    
    if delineamento == "DBC":
        form_base += f" + C({col_local}):C({col_bloco})"
        
    modelo = ols(form_base, data=df_f).fit()
    anova_table = sm.stats.anova_lm(modelo, typ=2)
    
    try: p_interacao = anova_table.loc[f"C({col_trat}):C({col_local})", "PR(>F)"]
    except: p_interacao = 1.0
    
    try: p_trat = anova_table.loc[f"C({col_trat})", "PR(>F)"]
    except: p_trat = 1.0
    
    resid = modelo.resid
    w_stat, p_shapiro = stats.shapiro(resid)
    
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
# 🏁 FIM DO BLOCO 10
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 11: Motores Estatísticos V (OLS Principais - Com Cache)
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

# --- CÁLCULO DE HOMOGENEIDADE PARA CONJUNTA ---
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
# 🏁 FIM DO BLOCO 11
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 12: Interface - Setup e CSS
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

# Configura apenas o título da ABA do navegador, não o texto da página
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
            [data-testid="stFileUploader"] section {
                padding: 1rem !important;
                align-items: stretch !important;
            }
            [data-testid="stFileUploader"] button {
                width: 100% !important;
                max-width: 100% !important;
                display: block !important;
                margin-left: 0 !important;
                margin-right: 0 !important;
            }
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

# ⚠️ REMOVIDO: st.title("🌱 AgroStat Pro: Análises Estatísticas") foi deletado daqui!
# ==============================================================================
# 🏁 FIM DO BLOCO 12
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 13: Interface - Sidebar e Menu
# ==============================================================================
# Substituí a imagem externa (quebrava) por um título nativo robusto
st.sidebar.markdown("# 🌾 AgroStat Pro") 

# --- MENU PRINCIPAL ---
modo_app = st.sidebar.radio(
    "Navegação:",
    ("📊 Análise Estatística", "🎲 Sorteio Experimental"),
    index=0
)

# ALTERAÇÃO: O título aparece APENAS se estivermos neste modo
if modo_app == "📊 Análise Estatística":
    st.title("🌱 AgroStat Pro: Análises Estatísticas")

st.sidebar.markdown("---")

# ==============================================================================
# LÓGICA CONDICIONAL DA SIDEBAR
# ==============================================================================

# --- MODO 1: ANÁLISE ESTATÍSTICA ---
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
        
        OPCAO_PADRAO = "Local Único (Análise Individual)" 
        col_local = st.sidebar.selectbox("Coluna de Local/Ambiente", [OPCAO_PADRAO] + [c for c in colunas if c not in cols_trats], on_change=reset_analise)
        
        col_bloco = None
        cols_ocupadas = cols_trats + [col_local]
        
        if delineamento == "DBC":
            col_bloco = st.sidebar.selectbox("Blocos (Repetições)", [c for c in colunas if c not in cols_ocupadas], on_change=reset_analise)
            cols_ocupadas.append(col_bloco)
        else:
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

        if st.sidebar.button("🚀 Rodar Dados!", type="primary", use_container_width=True):
            st.session_state['processando'] = True

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

# --- MODO 2: SORTEIO (Novo) ---
elif modo_app == "🎲 Sorteio Experimental":
    st.sidebar.info("🛠️ Você está no modo de Sorteio. Configure os tratamentos e gere o croqui na tela principal.")
    st.session_state['processando'] = False 
# ==============================================================================
# 🏁 FIM DO BLOCO 13
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 14: Execução Principal - Setup e Inicialização (DIAGNÓSTICO RIGOROSO)
# ==============================================================================
# TRAVA DE SEGURANÇA: Só roda se o botão foi clicado E se estivermos no modo Análise
if st.session_state['processando'] and modo_app == "📊 Análise Estatística":
    
    # --- VERIFICAÇÕES DE SEGURANÇA (Evita NameError) ---
    erro_vars = False
    
    if 'lista_resps' not in locals() or not lista_resps:
        st.error("⚠️ Nenhuma Variável Resposta detectada. Por favor, carregue um arquivo e selecione as variáveis.")
        erro_vars = True
        
    elif 'cols_trats' not in locals() or not cols_trats:
         st.error("⚠️ Nenhum Tratamento selecionado. Por favor, selecione os fatores.")
         erro_vars = True

    # Só prossegue se não houve erro nas variáveis
    if not erro_vars:
        # --- 0. APLICAÇÃO INTELIGENTE DE RENOMEAÇÃO ---
        if 'df_analise' not in locals():
            if 'df' in locals():
                df_analise = df.copy()
            else:
                st.error("Erro crítico: DataFrame não encontrado na memória.")
                st.stop()

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

        # --- AVISO DE MODO CONJUNTA ---
        if modo_analise == "CONJUNTA":
            n_locais_detectados = len(df[col_local].unique())
            st.info(f"🌍 **Modo Conjunta Ativado!** ({n_locais_detectados} locais)")

        st.markdown(f"### 📋 Resultados: {len(lista_resps)} variáveis processadas")
        
        # --- 0.1 ANÁLISE DE DIMENSÕES ---
        dimensoes = []
        for f in cols_trats:
            n_niveis = df[f].nunique()
            dimensoes.append(str(n_niveis))
        
        esquema_txt = "x".join(dimensoes)
        eh_fatorial = len(cols_trats) > 1
        
        if eh_fatorial:
            st.info(f"🔬 **Esquema Fatorial Detectado:** {esquema_txt} ({' x '.join(cols_trats)})")
        else:
            log_message(f"✅ Experimento Unifatorial [{esquema_txt}] identificado.")

        # --- MEMÓRIA DO RELATÓRIO (CRUCIAL) ---
        dados_para_relatorio_final = [] 
        # ---------------------------------------

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
                
                if modo_analise == "INDIVIDUAL":
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
                        st.error(f"⚠️ **Violação de Homogeneidade (MSE):** Razão {razao:.2f} > 7. A variância entre os locais é muito discrepante.")

                # ==============================================================================
                # 🚨 PAINEL DE ALERTAS (PADRÃO VERMELHO/ERRO PARA TUDO QUE FOR RUIM)
                # ==============================================================================
                cv_val = (np.sqrt(res_model.mse_resid)/df_proc[col_resp].mean())*100
                
                # Container para agrupar avisos (Visualmente mais limpo)
                with st.container():
                    # 1. Alerta de CV (Agora sempre Vermelho se > 20)
                    if cv_val > 30:
                        st.error(f"⚠️ **CV Crítico ({cv_val:.2f}%):** Precisão experimental muito baixa. Dados inconsistentes.")
                    elif cv_val > 20:
                        st.error(f"⚠️ **CV Alto ({cv_val:.2f}%):** Precisão experimental reduzida. Atenção na interpretação.")
                    
                    # 2. Alerta de ANOVA (P-valor)
                    if p_final_trat > 0.05:
                        st.error(f"⚠️ **ANOVA Não Significativa (P={p_final_trat:.4f}):** Não houve diferença estatística entre os tratamentos.")

                    # 3. Alerta de R² (Agora sempre Vermelho se < 0.70)
                    r2_val = extras.get('r2', 0)
                    if r2_val < 0.50:
                        st.error(f"⚠️ **R² Crítico ({r2_val:.2f}):** O modelo não se ajustou aos dados (Explica < 50%).")
                    elif r2_val < 0.70:
                        st.error(f"⚠️ **R² Regular ({r2_val:.2f}):** O ajuste do modelo está abaixo do ideal (< 0.70).")

                    # 4. Alerta de Acurácia Seletiva (Agora sempre Vermelho se < 0.70)
                    ac_val = extras.get('acuracia', 0)
                    if ac_val > 0 and ac_val < 0.70:
                        st.error(f"⚠️ **Acurácia Baixa ({ac_val:.2f}):** Baixa confiabilidade para seleção de genótipos.")

                    # 5. Alerta de Herdabilidade (Agora sempre Vermelho se < 0.50)
                    h2_val = extras.get('h2', 0)
                    if h2_val > 0 and h2_val < 0.50:
                        st.error(f"⚠️ **Herdabilidade Baixa ({h2_val:.2f}):** Forte influência ambiental sobre a característica.")
                
                # Feedback Positivo Geral (Só aparece se estiver tudo "verde")
                if cv_val <= 20 and p_final_trat < 0.05 and r2_val >= 0.70:
                    st.success("✅ **Excelente:** Dados com alta precisão e modelo bem ajustado.")
# ==============================================================================
# 🏁 FIM DO BLOCO 14
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 15: Execução Principal - ANOVA e Diagnósticos
# ==============================================================================
                # --- EXIBIÇÃO FINAL DO RESULTADO ANOVA (TOPO) ---
                st.markdown("---")
                st.markdown("### 📊 Análise de Variância (ANOVA)")
                if anova_tab is not None:
                    st.dataframe(anova_tab)

                # ==========================================================
                # LÓGICA DE ALERTAS: DIAGNÓSTICO FATORIAL COMPLETO
                # ==========================================================
                if eh_fatorial and 'res_analysis' in locals() and res_analysis:
                    # Pega a tabela ANOVA bruta (numérica) para verificação
                    raw_anova = res_analysis['anova']
                    
                    # Filtra apenas as linhas de interesse (Fatores e Interações)
                    ignorar = ['Residual', 'Resíduo', 'Intercept', 'Total']
                    if 'Bloco' in df.columns: ignorar.append('Bloco') 
                    
                    linhas_fatores = [idx for idx in raw_anova.index if not any(x in str(idx) for x in ignorar) and 'Bloco' not in str(idx)]
                    
                    for fator in linhas_fatores:
                        try:
                            p_val = raw_anova.loc[fator, 'PR(>F)']
                            nome_display = str(fator).replace("C(", "").replace(")", "").replace(":", " x ").replace(", Sum", "")
                            
                            # Lógica para INTERAÇÃO
                            if ":" in str(fator):
                                if p_val < 0.05:
                                    st.warning(f"⚠️ **Interação ({nome_display}): Significativa (P={p_val:.4e}).** O efeito dos fatores é dependente. Analise o desdobramento.")
                                else:
                                    st.success(f"✅ **Interação ({nome_display}): Não Significativa (P={p_val:.4f}).** Atuação independente.")
                            
                            # Lógica para FATORES PRINCIPAIS
                            else:
                                if p_val < 0.05:
                                    st.success(f"✅ **Fator Principal ({nome_display}): Significativo (P={p_val:.4e}).**")
                                else:
                                    st.error(f"⚠️ **Fator Principal ({nome_display}): Não Significativo (P={p_val:.4f}).**")
                        except: pass

                else:
                    # --- CENÁRIO UNI-FATORIAL SIMPLES ---
                    if p_final_trat < 0.05: 
                        st.success(f"✅ **Diferença Significativa (P = {p_final_trat:.4e}).** Rejeita-se a Hipótese Nula (H0).")
                    else: 
                        st.error(f"⚠️ **Não Significativo (P = {p_final_trat:.4f}).** Aceita-se H0 (Médias estatisticamente iguais).")
# ==============================================================================
# 🏁 FIM DO BLOCO 15
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 16: Árvore de Decisão Universal (Lógica de Pressupostos)
# ==============================================================================
                # Esta lógica agora se aplica tanto para DIC, DBC Individual quanto Conjunta.
                # As variáveis p_shap, p_bart, p_lev foram definidas no Bloco 10.

                st.markdown("---")
                st.markdown("#### 🩺 Diagnóstico dos Pressupostos")
                st.markdown(gerar_tabela_diagnostico(p_shap, p_bart, p_lev))

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
                            
                # ==========================================================
                # 🪄 PREPARAÇÃO DE VARIÁVEIS PARA O PRÓXIMO BLOCO
                # ==========================================================
                col_trat_original_lista = cols_trats # Backup da lista (se precisar)
                col_trat = col_combo 
                # ==========================================================
# ==============================================================================
# 🏁 FIM DO BLOCO 16
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 17: Visualização Completa - Funções Auxiliares
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
# ==============================================================================
# 🏁 FIM DO BLOCO 17
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 18: Visualização - Análise Individual (Rodapé Dinâmico)
# ==============================================================================
                # --- PATCH DE SEGURANÇA (CORREÇÃO DO ERRO) ---
                # Garante que a variável exista, mesmo se o Bloco 16 falhar ou não existir
                if 'analise_valida' not in locals():
                    analise_valida = False

                if analise_valida:
                    
                    if modo_analise == "INDIVIDUAL":
                        # --- 1. GESTÃO DE ESTADO (MEMÓRIA) PARA ESTA VARIÁVEL ---
                        # Chaves únicas para salvar as configs desta variável
                        key_desc = f"final_desc_{col_resp}_{i}"
                        key_show_mean = f"final_show_mean_{col_resp}_{i}"
                        key_show_cv = f"final_show_cv_{col_resp}_{i}"

                        # Se não existir na memória, define o padrão
                        if key_desc not in st.session_state: st.session_state[key_desc] = True # Padrão: Maior (True)
                        if key_show_mean not in st.session_state: st.session_state[key_show_mean] = True
                        if key_show_cv not in st.session_state: st.session_state[key_show_cv] = True

                        # Recupera os valores ATUAIS para usar nos cálculos (Estado Salvo)
                        is_decrescente_final = st.session_state[key_desc]
                        show_mean_final = st.session_state[key_show_mean]
                        show_cv_final = st.session_state[key_show_cv]

                        # Define o index do Radio para a interface refletir o estado salvo
                        # 0 = Maior (True), 1 = Menor (False)
                        idx_radio_padrao = 0 if is_decrescente_final else 1

                        # --- 2. CÁLCULOS GLOBAIS (VALIDOS PARA TODAS AS ABAS) ---
                        eh_numerico = False
                        try:
                            pd.to_numeric(df_proc[col_trat], errors='raise')
                            eh_numerico = True
                        except: eh_numerico = False

                        medias_ind = df_proc.groupby(col_trat)[col_resp].mean()
                        media_geral_valor = df_proc[col_resp].mean()
                        reps_ind = df_proc.groupby(col_trat)[col_resp].count().mean()
                        n_trats_ind = len(medias_ind)
                        max_val_ind = medias_ind.max()

                        # Cálculos Estatísticos usando a configuração FINAL (Salva)
                        # Isso garante que Tukey, SK e Gráficos usem sempre a mesma lógica
                        df_tukey_ind = tukey_manual_preciso(medias_ind, res['mse'], res['df_resid'], reps_ind, n_trats_ind, decrescente=is_decrescente_final)
                        df_sk_ind = scott_knott(medias_ind, res['mse'], res['df_resid'], reps_ind, n_trats_ind, decrescente=is_decrescente_final)
                        
                        if 'Letras' in df_tukey_ind.columns: df_tukey_ind = df_tukey_ind.rename(columns={'Letras': 'Grupos'})
                        if 'Grupo' in df_sk_ind.columns: df_sk_ind = df_sk_ind.rename(columns={'Grupo': 'Grupos'})
                        if 'Letras' in df_sk_ind.columns: df_sk_ind = df_sk_ind.rename(columns={'Letras': 'Grupos'})
                        
                        df_tukey_ind = df_tukey_ind[['Media', 'Grupos']]
                        df_sk_ind = df_sk_ind[['Media', 'Grupos']]

                        eh_fatorial = len(cols_trats) > 1
                        interacao_sig = False
                        if eh_fatorial:
                            idx_int = [x for x in res['anova'].index if ":" in str(x) or " x " in str(x)]
                            if idx_int:
                                try:
                                    p_int_val = res['anova'].loc[idx_int[-1], 'PR(>F)']
                                    if p_int_val < 0.05: interacao_sig = True
                                except: pass

                        # --- 3. RENDERIZAÇÃO DAS ABAS ---
                        titulos_abas = []
                        if eh_numerico: titulos_abas.append("📈 Regressão")
                        titulos_abas.extend(["📦 Teste de Tukey", "📦 Teste de Scott-Knott", "📊 Gráficos Barras"])
                        
                        tabs_ind = st.tabs(titulos_abas)
                        idx_aba = 0

                        # ABA REGRESSÃO
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

                        # ABA TUKEY
                        with tabs_ind[idx_aba]:
                            if res['p_val'] > 0.05:
                                st.warning("⚠️ **ANOVA Não Significativa:** Médias estatisticamente iguais.")
                                container_tk = st.expander("Visualizar mesmo assim")
                            else: container_tk = st.container()

                            with container_tk:
                                if eh_fatorial and interacao_sig:
                                    st.error("🚨 **Interação Significativa.** Use a Matriz de Desdobramento abaixo.")
                                
                                st.markdown("#### Ranking Geral (Tukey)")

                                # ==============================================================================
                                # ⚙️ MENU DE CONFIGURAÇÃO (TUKEY)
                                # ==============================================================================
                                with st.expander("✏️ Personalizar Tabela (Rodapé e Dados)"):
                                    col_cfg1, col_cfg2 = st.columns(2)
                                    with col_cfg1:
                                        st.markdown("##### 🎯 Objetivo Agronômico")
                                        c_rank_temp_tk = st.radio(
                                            "O que define o melhor tratamento?",
                                            (
                                                "⬆️ Quanto MAIOR os valores melhor é a variável (Ex: Produtividade, Peso)", 
                                                "⬇️ Quanto MENOR os valores melhor é a variável (Ex: Doença, Ciclo, Acamamento)"
                                            ),
                                            index=idx_radio_padrao, # Usa o estado salvo
                                            key=f"temp_rank_tk_{col_resp}_{i}"
                                        )
                                    with col_cfg2:
                                        st.markdown("##### 📝 Opções de Exibição")
                                        show_m_temp_tk = st.checkbox("Incluir Média Geral no Rodapé", value=show_mean_final, key=f"temp_mean_tk_{col_resp}_{i}")
                                        show_cv_temp_tk = st.checkbox("Incluir CV (%) no Rodapé", value=show_cv_final, key=f"temp_cv_tk_{col_resp}_{i}")
                                    
                                    st.markdown("---")
                                    # BOTÃO DE AÇÃO TUKEY - ATUALIZA TUDO
                                    if st.button("🔄 Atualizar Tabela", key=f"btn_upd_tk_{col_resp}_{i}", use_container_width=True):
                                        st.session_state[key_desc] = True if "MAIOR" in c_rank_temp_tk else False
                                        st.session_state[key_show_mean] = show_m_temp_tk
                                        st.session_state[key_show_cv] = show_cv_temp_tk
                                        st.rerun()
                                # ==============================================================================

                                # TABELA
                                st.dataframe(df_tukey_ind.style.format({"Media": "{:.2f}"}), use_container_width=True)
                                
                                # RODAPÉ
                                txt_m_tk, txt_cv_tk = calcular_texto_rodape(media_geral_valor, res['mse'], show_mean_final, show_cv_final)
                                infos_extras = []
                                if txt_cv_tk: infos_extras.append(txt_cv_tk)
                                if txt_m_tk: infos_extras.append(txt_m_tk)
                                texto_final_extras = f". {'. '.join(infos_extras)}." if infos_extras else "."
                                
                                st.markdown(f"> **Nota de Rodapé da Tabela:** Médias seguidas pela mesma letra na coluna não diferem estatisticamente entre si pelo teste de Tukey a 5% de probabilidade{texto_final_extras}")
                                
                                if interacao_sig:
                                    st.markdown("---")
                                    st.subheader("🔠 Matriz de Desdobramento (Tukey)")
                                    fl_tk = st.selectbox("Fator na Linha", cols_trats, key=f"mat_tk_l_{col_resp}_{i}")
                                    fc_tk = [f for f in cols_trats if f != fl_tk][0]
                                    df_m_tk = gerar_dataframe_matriz_total(df_proc, fl_tk, fc_tk, tukey_manual_preciso, res['mse'], res['df_resid'])
                                    st.dataframe(df_m_tk)
                                    st.markdown("> **Nota:** Médias seguidas pela mesma letra **maiúscula na linha** não diferem estatisticamente entre si (comparação horizontal). Médias seguidas pela mesma letra **minúscula na coluna** não diferem estatisticamente entre si (comparação vertical), pelo teste de Tukey a 5%.")
                        
                        # ABA SCOTT-KNOTT
                        with tabs_ind[idx_aba+1]:
                            if res['p_val'] > 0.05:
                                st.warning("⚠️ **ANOVA Não Significativa:** Médias estatisticamente iguais.")
                                container_sk = st.expander("Visualizar mesmo assim")
                            else: container_sk = st.container()

                            with container_sk:
                                if eh_fatorial and interacao_sig:
                                    st.error("🚨 **Interação Significativa.** Use a Matriz de Desdobramento abaixo.")

                                st.markdown("#### Ranking Geral (Scott-Knott)")
                                
                                # ==============================================================================
                                # ⚙️ MENU DE CONFIGURAÇÃO (SCOTT-KNOTT) - TOTALMENTE SINCRONIZADO
                                # ==============================================================================
                                with st.expander("✏️ Personalizar Tabela (Rodapé e Dados)"):
                                    col_cfg1, col_cfg2 = st.columns(2)
                                    with col_cfg1:
                                        st.markdown("##### 🎯 Objetivo Agronômico")
                                        c_rank_temp_sk = st.radio(
                                            "O que define o melhor tratamento?",
                                            (
                                                "⬆️ Quanto MAIOR os valores melhor é a variável (Ex: Produtividade, Peso)", 
                                                "⬇️ Quanto MENOR os valores melhor é a variável (Ex: Doença, Ciclo, Acamamento)"
                                            ),
                                            index=idx_radio_padrao, # Usa o MESMO estado salvo
                                            key=f"temp_rank_sk_{col_resp}_{i}"
                                        )
                                    with col_cfg2:
                                        st.markdown("##### 📝 Opções de Exibição")
                                        show_m_temp_sk = st.checkbox("Incluir Média Geral no Rodapé", value=show_mean_final, key=f"temp_mean_sk_{col_resp}_{i}")
                                        show_cv_temp_sk = st.checkbox("Incluir CV (%) no Rodapé", value=show_cv_final, key=f"temp_cv_sk_{col_resp}_{i}")
                                    
                                    st.markdown("---")
                                    # BOTÃO DE AÇÃO SCOTT-KNOTT - ATUALIZA TUDO
                                    if st.button("🔄 Atualizar Tabela", key=f"btn_upd_sk_{col_resp}_{i}", use_container_width=True):
                                        # Atualiza as MESMAS chaves de sessão
                                        st.session_state[key_desc] = True if "MAIOR" in c_rank_temp_sk else False
                                        st.session_state[key_show_mean] = show_m_temp_sk
                                        st.session_state[key_show_cv] = show_cv_temp_sk
                                        st.rerun()
                                # ==============================================================================

                                # TABELA
                                st.dataframe(df_sk_ind.style.format({"Media": "{:.2f}"}), use_container_width=True)
                                
                                # RODAPÉ
                                txt_m_sk, txt_cv_sk = calcular_texto_rodape(media_geral_valor, res['mse'], show_mean_final, show_cv_final)
                                infos_extras_sk = []
                                if txt_cv_sk: infos_extras_sk.append(txt_cv_sk)
                                if txt_m_sk: infos_extras_sk.append(txt_m_sk)
                                texto_final_extras_sk = f". {'. '.join(infos_extras_sk)}." if infos_extras_sk else "."
                                
                                st.markdown(f"> **Nota de Rodapé da Tabela:** Médias seguidas pela mesma letra na coluna não diferem estatisticamente entre si pelo teste de Scott-Knott a 5% de probabilidade{texto_final_extras_sk}")
                                
                                if interacao_sig:
                                    st.markdown("---")
                                    st.subheader("🔠 Matriz de Desdobramento (Scott-Knott)")
                                    fl_sk = st.selectbox("Fator na Linha", cols_trats, key=f"mat_sk_l_{col_resp}_{i}")
                                    fc_sk = [f for f in cols_trats if f != fl_sk][0]
                                    df_m_sk = gerar_dataframe_matriz_total(df_proc, fl_sk, fc_sk, scott_knott, res['mse'], res['df_resid'])
                                    st.dataframe(df_m_sk)
                                    st.markdown("> **Nota:** Médias seguidas pela mesma letra **maiúscula na linha** não diferem estatisticamente entre si (comparação horizontal). Médias seguidas pela mesma letra **minúscula na coluna** não diferem estatisticamente entre si (comparação vertical), pelo teste de Scott-Knott a 5%.")

                        # --- ABA GRÁFICOS ---
                        with tabs_ind[idx_aba+2]:
                            if res['p_val'] > 0.05:
                                st.warning("⚠️ **ANOVA Não Significativa.**")
                                container_graf = st.expander("Visualizar mesmo assim")
                            else: container_graf = st.container()

                            with container_graf:
                                sub_tabs_graf = st.tabs(["📊 Gráfico Tukey", "📊 Gráfico Scott-Knott"])
                                
                                # Recalcula/Reordena para gráficos usando a variável Global is_decrescente_final
                                df_tk_g = tukey_manual_preciso(medias_ind, res['mse'], res['df_resid'], reps_ind, n_trats_ind, decrescente=is_decrescente_final)
                                if 'Letras' in df_tk_g.columns: df_tk_g = df_tk_g.rename(columns={'Letras': 'Grupos'})
                                
                                df_sk_g = scott_knott(medias_ind, res['mse'], res['df_resid'], reps_ind, n_trats_ind, decrescente=is_decrescente_final)
                                if 'Grupo' in df_sk_g.columns: df_sk_g = df_sk_g.rename(columns={'Grupo': 'Grupos'})
                                
                                with sub_tabs_graf[0]:
                                    cfg_tk = mostrar_editor_grafico(f"tk_ind_{col_resp}_{i}", "Médias (Tukey)", col_trat, col_resp, usar_cor_unica=True)
                                    if eh_fatorial:
                                        df_plot_tk = df_tk_g.reset_index().rename(columns={'index': col_trat})
                                        try:
                                            split_data = df_plot_tk[col_trat].astype(str).str.split(' + ', expand=True)
                                            if split_data.shape[1] >= 2:
                                                df_plot_tk[cols_trats[0]] = split_data[0]
                                                df_plot_tk[cols_trats[1]] = split_data[1]
                                                f_tk = px.bar(df_plot_tk, x=cols_trats[0], y='Media', color=cols_trats[1], text='Grupos', barmode='group')
                                            else: f_tk = px.bar(df_plot_tk, x=col_trat, y='Media', text='Grupos')
                                        except: f_tk = px.bar(df_plot_tk, x=col_trat, y='Media', text='Grupos')
                                    else:
                                        f_tk = px.bar(df_tk_g.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupos')
                                    
                                    st.plotly_chart(estilizar_grafico_avancado(f_tk, cfg_tk, max_val_ind), use_container_width=True, key=f"chart_bar_tk_{col_resp}_{i}")
                                
                                with sub_tabs_graf[1]:
                                    grps_sk = sorted(df_sk_g['Grupos'].dropna().unique())
                                    cfg_sk = mostrar_editor_grafico(f"sk_ind_{col_resp}_{i}", "Médias (Scott-Knott)", col_trat, col_resp, usar_cor_unica=False, grupos_sk=grps_sk)
                                    if eh_fatorial:
                                        df_plot_sk = df_sk_g.reset_index().rename(columns={'index': col_trat})
                                        try:
                                            split_data = df_plot_sk[col_trat].astype(str).str.split(' + ', expand=True)
                                            if split_data.shape[1] >= 2:
                                                df_plot_sk[cols_trats[0]] = split_data[0]
                                                df_plot_sk[cols_trats[1]] = split_data[1]
                                                f_sk = px.bar(df_plot_sk, x=cols_trats[0], y='Media', color=cols_trats[1], text='Grupos', barmode='group')
                                            else: f_sk = px.bar(df_plot_sk, x=col_trat, y='Media', text='Grupos', color='Grupos', color_discrete_map=cfg_sk['cores_map'])
                                        except: f_sk = px.bar(df_plot_sk, x=col_trat, y='Media', text='Grupos', color='Grupos', color_discrete_map=cfg_sk['cores_map'])
                                    else:
                                        f_sk = px.bar(df_sk_g.reset_index().rename(columns={'index':col_trat}), x=col_trat, y='Media', text='Grupos', color='Grupos', color_discrete_map=cfg_sk['cores_map'])
                                    
                                    st.plotly_chart(estilizar_grafico_avancado(f_sk, cfg_sk, max_val_ind), use_container_width=True, key=f"chart_bar_sk_{col_resp}_{i}")
# ==============================================================================
# 🏁 FIM DO BLOCO 18
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 19: Visualização - Análise Conjunta (BLINDADO)
# ==============================================================================
                    # ----------------------------------------------------------
                    # CENÁRIO B: ANÁLISE CONJUNTA
                    # ----------------------------------------------------------
                    else:
                        # --- TRAVA DE SEGURANÇA CONTRA KEYERROR ---
                        # Se o usuário mudou o dropdown para 'Local Único' mas o estado ainda acha que é Conjunta
                        coluna_local_valida = col_local in df_proc.columns

                        if not coluna_local_valida:
                            st.warning("⚠️ **Aviso de Navegação:** Você alterou o 'Local' para 'Local Único' na barra lateral.") 
                            st.info("👉 Por favor, clique novamente no botão **🚀 Rodar Dados!** para atualizar os resultados para o modo Individual.")
                        
                        else:
                            # SÓ EXECUTA SE A COLUNA DE LOCAL EXISTIR DE FATO NO DATAFRAME
                            locais_unicos = sorted(df_proc[col_local].unique())
                            titulos_abas = ["📊 Média Geral"] + [f"📍 {loc}" for loc in locais_unicos] + ["📈 Interação"]
                            abas = st.tabs(titulos_abas)
                            p_int_conj = res_conj.get('p_interacao', 1.0)
                            interacao_significativa = p_int_conj < 0.05
                            
                            # --- ABA 0: MÉDIA GERAL ---
                            with abas[0]: 
                                if interacao_significativa:
                                    st.error("🚨 **INTERDIÇÃO:** Interação Significativa Detectada.")
                                    st.markdown("👉 Como o desempenho muda conforme o ambiente, esta **Média Geral não representa a realidade**. Não utilize esta aba para conclusões técnicas. Vá para a aba 'Interação'.")
                                else:
                                    st.success("✅ **APROVADO:** Interação Não Significativa.")
                                    st.markdown("👉 O comportamento é estável. Você **pode e deve** usar esta aba de Média Geral para suas conclusões.")
                                
                                # --- TRAVA DE SEGURANÇA (P-Valor Tratamento Geral) ---
                                p_trat_geral = res_conj.get('p_trat', 1.0)
                                
                                exibir_conteudo_geral = True
                                if p_trat_geral > 0.05:
                                    st.warning(f"⚠️ **ANOVA Não Significativa (P={p_trat_geral:.4f}):** As médias gerais são estatisticamente iguais.")
                                    container_geral = st.expander("✏️ Visualizar Média Geral mesmo assim")
                                else:
                                    container_geral = st.container()

                                with container_geral:
                                    medias_geral = df_proc.groupby(col_trat)[col_resp].mean()
                                    reps_geral = df_proc.groupby(col_trat)[col_resp].count().mean()
                                    max_val_geral = medias_geral.max()

                                    df_tukey_geral = tukey_manual_preciso(medias_geral, res_conj['mse'], res_conj['df_resid'], reps_geral, len(medias_geral))
                                    df_sk_geral = scott_knott(medias_geral, res_conj['mse'], res_conj['df_resid'], reps_geral, len(medias_geral))

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
                                    if interacao_significativa:
                                        st.success(f"✅ **ANÁLISE RECOMENDADA:** Focando em {loc}.")
                                        st.caption("Como houve interação, é correto analisar o que aconteceu especificamente neste local.")
                                    else:
                                        st.warning(f"⚠️ **CUIDADO:** Interação Não Significativa.")
                                        st.caption(f"As diferenças vistas aqui em {loc} podem ser apenas ruído estatístico. A recomendação segura é olhar a Média Geral.")
                                    
                                    df_loc = df_proc[df_proc[col_local] == loc]
                                    res_loc = rodar_analise_individual(df_loc, [col_trat], col_resp, delineamento, col_bloco)
                                    
                                    if res_loc['p_val'] > 0.05:
                                        st.warning(f"⚠️ **ANOVA Não Significativa em {loc} (P={res_loc['p_val']:.4f}).** Médias estatisticamente iguais.")
                                        container_loc = st.expander(f"✏️ Visualizar Dados de {loc} mesmo assim")
                                    else:
                                        container_loc = st.container()
                                    
                                    with container_loc:
                                        meds_loc = df_loc.groupby(col_trat)[col_resp].mean()
                                        reps_loc = df_loc.groupby(col_trat)[col_resp].count().mean()
                                        max_val_loc = meds_loc.max()

                                        df_tk_loc = tukey_manual_preciso(meds_loc, res_loc['mse'], res_loc['df_resid'], reps_loc, len(meds_loc))
                                        df_sk_loc = scott_knott(meds_loc, res_loc['mse'], res_loc['df_resid'], reps_loc, len(meds_loc))
                                        
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
                                if interacao_significativa:
                                    st.success("✅ **INTERAÇÃO CONFIRMADA:** O ambiente altera o resultado dos tratamentos.")
                                    st.info("💡 **DICA:** Utilize a matriz abaixo para identificar qual tratamento venceu em cada cenário.")
                                    st.success("✅ **ANÁLISE RECOMENDADA:** Foque sua interpretação na matriz abaixo.")
                                    
                                    st.markdown("#### Matriz: Local (Linha) x Tratamento (Coluna)")
                                    df_m_conj = gerar_dataframe_matriz_total(df_proc, col_local, col_trat, tukey_manual_preciso, res_conj['mse'], res_conj['df_resid'])
                                    st.dataframe(df_m_conj)
                                    st.markdown("> **Nota:** Médias seguidas pela mesma letra **maiúscula na linha** não diferem estatisticamente entre si (comparação horizontal). Médias seguidas pela mesma letra **minúscula na coluna** não diferem estatisticamente entre si (comparação vertical), pelo teste de Tukey a 5%.")

                                    st.markdown("---")
                                    df_inter = df_proc.groupby([col_trat, col_local])[col_resp].mean().reset_index()
                                    cfg_int = mostrar_editor_grafico(f"int_{col_resp}_{i}", f"Interação: {col_resp}", col_local, col_resp, usar_cor_unica=False, grupos_sk=trats_inter)
                                    f_i = px.line(df_inter, x=col_local, y=col_resp, color=col_trat, markers=True, color_discrete_map=cfg_int['cores_map'])
                                    st.plotly_chart(estilizar_grafico_avancado(f_i, cfg_int), use_container_width=True, key=f"chart_int_{col_resp}_{i}")
                                else: 
                                    st.warning("⚠️ **ATENÇÃO:** Não houve interação significativa.")
                                    st.markdown("A análise de desdobramento abaixo é meramente ilustrativa. **As conclusões devem ser tomadas na aba 'Média Geral'.**")
                                    
                                    st.caption("Visualização exploratória:")
                                    df_inter = df_proc.groupby([col_trat, col_local])[col_resp].mean().reset_index()
                                    cfg_int = mostrar_editor_grafico(f"int_ns_{col_resp}_{i}", f"Gráfico Exploratório (NS)", col_local, col_resp, usar_cor_unica=False, grupos_sk=trats_inter)
                                    f_i = px.line(df_inter, x=col_local, y=col_resp, color=col_trat, markers=True, color_discrete_map=cfg_int['cores_map'])
                                    st.plotly_chart(estilizar_grafico_avancado(f_i, cfg_int), use_container_width=True, key=f"chart_int_ns_{col_resp}_{i}")
# ==============================================================================
# 🏁 FIM DO BLOCO 19
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 20: Visualização Final e Captura de Dados (FLUXO ESTRITO)
# ==============================================================================
                # --- LÓGICA DE VISUALIZAÇÃO E FALLBACK ---
                grafico_final_obj = None 
                tabela_final_obj = None

                # Tenta recuperar o critério de ordenação do estado GLOBAL
                key_desc_glob = f"final_desc_{col_resp}_{i}"
                if key_desc_glob in st.session_state:
                    sort_desc = st.session_state[key_desc_glob]
                else:
                    sort_desc = True # Padrão: Maior é melhor

                if analise_valida:
                    # ==========================================================
                    # CENÁRIO: PARAMÉTRICA (Gera apenas objetos PDF, Silencioso)
                    # ==========================================================
                    if transf_atual != "Nenhuma":
                        st.markdown("---"); st.markdown("### 🛡️ Solução Final: Análise Paramétrica (ANOVA)")
                        st.success(f"✅ **Transformação Eficaz!** Com **{transf_atual}**, os pressupostos foram atendidos.")
                        if st.button("Voltar ao Original", key=f"reset_success_{col_resp_original}"):
                            set_transformacao(col_resp_original, "Nenhuma"); st.rerun()
                    
                    try:
                        medias_rep = df_proc.groupby(col_trat)[col_resp].mean()
                        reps_rep = df_proc.groupby(col_trat)[col_resp].count().mean()
                        
                        # Usa a configuração global definida no Bloco 18
                        df_tk_rep = tukey_manual_preciso(medias_rep, res_model.mse_resid, res_model.df_resid, reps_rep, len(medias_rep), decrescente=sort_desc)
                        
                        if 'Letras' in df_tk_rep.columns: df_tk_rep = df_tk_rep.rename(columns={'Letras': 'Grupos'})
                        tabela_final_obj = df_tk_rep[['Media', 'Grupos']]
                        
                        import plotly.graph_objects as go
                        fig_rep = go.Figure()
                        fig_rep.add_trace(go.Bar(
                            x=tabela_final_obj.index, y=tabela_final_obj['Media'],
                            text=tabela_final_obj['Grupos'], textposition='outside',
                            marker_color='#2E86C1'
                        ))
                        fig_rep.update_layout(title=f"Médias: {col_resp} (Tukey)", template="plotly_white")
                        grafico_final_obj = fig_rep
                    except: pass

                else:
                    # ==========================================================
                    # CENÁRIO: PRESSUPOSTOS FALHARAM (FLUXO GUIADO)
                    # ==========================================================
                    st.markdown("---"); st.error("🚨 ALERTA ESTATÍSTICO: A validade da Análise de Variância (ANOVA) foi comprometida, pois os pressupostos estatísticos necessários não foram satisfeitos.")
                    
                    # --- ESTÁGIO 1: SEM TRANSFORMAÇÃO ---
                    if transf_atual == "Nenhuma":
                        st.info("💡 **Passo 1:** Tente transformar os dados para corrigir a normalidade/homogeneidade.")
                        if st.button("🧪 Tentar Transformação Log10", key=f"btn_log_{col_resp_original}", use_container_width=True):
                            set_transformacao(col_resp_original, "Log10"); st.rerun()

                    # --- ESTÁGIO 2: LOG10 FALHOU ---
                    elif transf_atual == "Log10":
                        st.warning("⚠️ A transformação **Log10** não resolveu o problema.")
                        st.info("💡 **Passo 2:** Tente a transformação por Raiz Quadrada.")
                        
                        col_b1, col_b2 = st.columns(2)
                        with col_b1:
                            if st.button("🌱 Tentar Raiz Quadrada (SQRT)", key=f"btn_sqrt_{col_resp_original}", use_container_width=True):
                                set_transformacao(col_resp_original, "Raiz Quadrada (SQRT)"); st.rerun()
                        with col_b2:
                            if st.button("↩️ Voltar ao Original", key=f"reset_log_{col_resp_original}", use_container_width=True):
                                set_transformacao(col_resp_original, "Nenhuma"); st.rerun()

                    # --- ESTÁGIO 3: SQRT FALHOU (FIM DA LINHA) ---
                    elif transf_atual == "Raiz Quadrada (SQRT)":
                        st.error("❌ As transformações (Log10 e SQRT) **não funcionaram**.")
                        
                        if st.button("↩️ Voltar ao Original (Reiniciar)", key=f"reset_final_np_{col_resp_original}"):
                            set_transformacao(col_resp_original, "Nenhuma"); st.rerun()

                        # --- AGORA SIM: TRAVA E AVISO EDUCATIVO ---
                        st.markdown("### 🛑 Decisão Necessária")
                        st.warning(f"""
                        ⚠️ **ATENÇÃO: Por que a Análise Não-Paramétrica é necessária?**
                        
                        Os seus dados **não atenderam** aos pressupostos obrigatórios para a realização da ANOVA (Normalidade dos Resíduos e/ou Homogeneidade de Variâncias), mesmo após todas as tentativas de transformação disponíveis. Insistir na ANOVA aqui geraria conclusões científicas falsas.

                        **Consequências da mudança:**
                        1.  **Foco:** A análise deixará de comparar **Médias** e passará a comparar **Medianas/Postos**.
                        2.  **Teste:** Será usado Kruskal-Wallis (DIC) ou Friedman (DBC).
                        3.  **Comparação:** O teste de médias (Tukey) será substituído pelo teste de **Dunn**.
                        """)

                        # Estado do botão de execução da não-paramétrica
                        key_run_np = f"run_np_manual_{col_resp}_{i}"
                        if key_run_np not in st.session_state: st.session_state[key_run_np] = False

                        if st.button("🚀 Rodar Análise Não-Paramétrica", key=f"btn_trigger_np_{col_resp}_{i}", type="primary", use_container_width=True):
                            st.session_state[key_run_np] = True
                            st.rerun()

                        # --- EXECUÇÃO APÓS CLIQUE ---
                        if st.session_state[key_run_np]:
                            
                            st.markdown("---")
                            # --- CÁLCULO NÃO-PARAMÉTRICO ---
                            nome_np, stat_np, p_np = calcular_nao_parametrico(df_proc, col_trat, col_resp, delineamento, col_bloco)
                            
                            # --- EXIBIÇÃO DA INTERPRETAÇÃO DO TESTE ---
                            c1, c2 = st.columns([1, 3])
                            with c1: st.metric(f"Teste: {nome_np}", f"{stat_np:.2f}")
                            with c2: 
                                sig_label = "Significativo" if p_np < 0.05 else "Não Significativo"
                                st.metric("P-valor", f"{p_np:.4f}", sig_label, delta_color="normal" if p_np < 0.05 else "inverse")
                            
                            if p_np < 0.05:
                                st.success(f"✅ **Diferença Detectada (P < 0.05):** O teste de {nome_np} indica que pelo menos um tratamento difere estatisticamente dos demais.")
                            else:
                                st.warning(f"⚠️ **Não Significativo (P >= 0.05):** Não há evidências estatísticas de diferença entre as medianas dos tratamentos.")
                            
                            st.markdown("#### 🏆 Ranking de Medianas")

                            # ==============================================================================
                            # ⚙️ MENU DE CONFIGURAÇÃO (NÃO PARAMÉTRICO)
                            # ==============================================================================
                            # Inicializa Estado para NP se não existir
                            if key_desc_glob not in st.session_state: st.session_state[key_desc_glob] = True
                            sort_desc_np_atual = st.session_state[key_desc_glob]

                            with st.expander("✏️ Personalizar Tabela (Rodapé e Dados)"):
                                st.markdown("##### 🎯 Objetivo Agronômico")
                                idx_padrao_np = 0 if sort_desc_np_atual else 1
                                c_rank_np = st.radio(
                                    "O que define o melhor tratamento?",
                                    (
                                        "⬆️ Quanto MAIOR os valores melhor é a variável (Ex: Produtividade, Peso)", 
                                        "⬇️ Quanto MENOR os valores melhor é a variável (Ex: Doença, Ciclo, Acamamento)"
                                    ),
                                    index=idx_padrao_np, 
                                    key=f"temp_rank_np_{col_resp}_{i}"
                                )
                                st.markdown("---")
                                
                                # BOTÃO DE ATUALIZAÇÃO
                                if st.button("🔄 Atualizar Tabela", key=f"btn_upd_np_{col_resp}_{i}", use_container_width=True):
                                    st.session_state[key_desc_glob] = True if "MAIOR" in c_rank_np else False
                                    st.rerun()
                            
                            # Usa o valor do Estado Global
                            sort_desc_np = st.session_state[key_desc_glob]

                            # --- TABELA DE MEDIANAS ---
                            df_meds = df_proc.groupby(col_trat)[col_resp].median().reset_index(name='Mediana')
                            df_iqr = df_proc.groupby(col_trat)[col_resp].apply(lambda x: f"{x.min():.2f} – {x.max():.2f}").reset_index(name='Amplitude')
                            df_final = pd.merge(df_meds, df_iqr, on=col_trat)
                            
                            if p_np < 0.05:
                                df_dunn = calcular_posthoc_dunn(df_proc, col_trat, col_resp)
                                trats_np = sorted(df_proc[col_trat].unique())
                                letras_dunn = gerar_letras_dunn(trats_np, df_dunn)
                                df_final['Grupo'] = df_final[col_trat].map(letras_dunn)
                            else:
                                df_final['Grupo'] = "a"
                                
                            # 1. ORDENAÇÃO DINÂMICA ROBUSTA
                            df_final = df_final.sort_values('Mediana', ascending=not sort_desc_np)
                            
                            # 2. RE-MAPEAMENTO VISUAL DAS LETRAS
                            if p_np < 0.05 and 'Grupo' in df_final.columns:
                                letras_na_ordem = []
                                seen = set()
                                for l in df_final['Grupo']:
                                    if l not in seen:
                                        letras_na_ordem.append(l)
                                        seen.add(l)
                                
                                mapa_letras = {}
                                ascii_char = 97 # 'a'
                                for l_antiga in letras_na_ordem:
                                    mapa_letras[l_antiga] = get_letra_segura(ascii_char - 97)
                                    ascii_char += 1
                                
                                df_final['Grupo'] = df_final['Grupo'].map(mapa_letras)

                            ordem_trats = df_final[col_trat].tolist()
                            tabela_final_obj = df_final # Guarda para o relatório
                            
                            st.dataframe(df_final.style.format({"Mediana": "{:.2f}"}), hide_index=True)
                            
                            # --- NOTA DE RODAPÉ (PADRÃO CIENTÍFICO) ---
                            if p_np < 0.05:
                                st.markdown("> **Nota:** Medianas seguidas pela mesma letra na coluna não diferem estatisticamente entre si pelo teste de Dunn a 5% de probabilidade.")
                            else:
                                st.markdown("> **Nota:** Pela ausência de significância no teste de Kruskal-Wallis (p ≥ 0.05), as medianas não diferem estatisticamente entre si.")
                                
                            st.markdown("---")

                            # --- VISUALIZAÇÃO INTELIGENTE ---
                            st.markdown("#### 📉 Visualização")
                            min_reps = df_proc.groupby(col_trat)[col_resp].count().min()
                            tem_empates = (df_proc.groupby(col_trat)[col_resp].apply(lambda x: x.max() - x.min()) == 0).any()
                            
                            idx_padrao = 0
                            if min_reps < 5: idx_padrao = 2 if tem_empates else 1
                            
                            opcoes = ["📦 Boxplot (Tradicional)", "📍 Strip Plot (Pontos)", "🎯 Dot Plot (Mediana Única)", "📊 Barras + Erro", "🎻 Violin Plot"]
                            tipo_grafico = st.selectbox("Estilo:", opcoes, index=idx_padrao, key=f"sel_graf_np_{col_resp_original}")
                            
                            cfg = mostrar_editor_grafico(f"edit_np_{col_resp}_{i}", f"Medianas: {col_resp}", col_trat, col_resp, usar_cor_unica=True)
                            
                            import plotly.graph_objects as go
                            fig_viz = go.Figure()
                            
                            cor_princ = cfg['cor_barras'] or '#5D6D7E'
                            cor_txt = cfg['cor_texto']
                            cor_borda = 'black'

                            # --- CONFIGURAÇÕES VISUAIS APLICADAS ---
                            pos_txt_final = 'top center' # Default para 'outside' ou 'auto'
                            if cfg.get('posicao_texto') == 'inside': 
                                pos_txt_final = 'bottom center'
                            
                            def aplicar_estilo_texto(texto):
                                if cfg.get('letras_negrito', False):
                                    return f"<b>{texto}</b>"
                                return texto

                            if "Dot Plot" in tipo_grafico:
                                labels_estilizados = [aplicar_estilo_texto(t) for t in df_final['Grupo']]
                                fig_viz.add_trace(go.Scatter(x=df_final[col_trat], y=df_final['Mediana'], mode='markers+text', marker=dict(size=18, color=cor_princ, symbol='circle', line=dict(width=2, color='white')), text=labels_estilizados, textposition=pos_txt_final, textfont=dict(size=cfg['font_size']+2, color=cor_txt), name='Mediana'))
                                fig_viz.update_traces(showlegend=False)
                            
                            elif "Barras" in tipo_grafico:
                                df_min = df_proc.groupby(col_trat)[col_resp].min(); df_max = df_proc.groupby(col_trat)[col_resp].max()
                                erros_sup = []; erros_inf = []
                                text_pos_y = []
                                text_labels = []
                                
                                for t in ordem_trats:
                                    m = df_final[df_final[col_trat]==t]['Mediana'].values[0]
                                    val_max = df_max[t]
                                    val_min = df_min[t]
                                    erros_sup.append(val_max - m)
                                    erros_inf.append(m - val_min)
                                    text_pos_y.append(val_max)
                                    raw_label = df_final[df_final[col_trat]==t]['Grupo'].values[0]
                                    text_labels.append(aplicar_estilo_texto(raw_label))

                                fig_viz.add_trace(go.Bar(
                                    x=df_final[col_trat], y=df_final['Mediana'], 
                                    marker_color=cor_princ, 
                                    error_y=dict(type='data', symmetric=False, array=erros_sup, arrayminus=erros_inf, visible=True, color=cor_txt, thickness=1.5, width=5)
                                ))
                                
                                fig_viz.add_trace(go.Scatter(
                                    x=df_final[col_trat], y=text_pos_y, text=text_labels, 
                                    mode='text', textposition=pos_txt_final, 
                                    textfont=dict(size=cfg['font_size'], color=cor_txt),
                                    hoverinfo='skip', showlegend=False
                                ))

                            elif "Boxplot" in tipo_grafico:
                                fig_viz.add_trace(go.Box(
                                    x=df_proc[col_trat], y=df_proc[col_resp], name="Dados", 
                                    marker_color=cor_princ, boxpoints=False, 
                                    line=dict(color=cor_borda, width=1.5), fillcolor=cor_princ
                                ))
                                y_pos = []; txts = []; margin = (df_proc[col_resp].max() - df_proc[col_resp].min()) * 0.1
                                for t in ordem_trats:
                                    y_pos.append(df_proc[df_proc[col_trat]==t][col_resp].max() + margin)
                                    raw_l = df_final[df_final[col_trat]==t]['Grupo'].values[0]
                                    txts.append(aplicar_estilo_texto(raw_l))
                                    
                                fig_viz.add_trace(go.Scatter(x=ordem_trats, y=y_pos, text=txts, mode='text', textposition=pos_txt_final, showlegend=False, textfont=dict(size=cfg['font_size'], color=cor_txt)))
                            
                            elif "Strip Plot" in tipo_grafico:
                                fig_viz.add_trace(go.Box(x=df_proc[col_trat], y=df_proc[col_resp], name="Dados", boxpoints='all', jitter=0.3, pointpos=0, fillcolor='rgba(0,0,0,0)', line=dict(width=0), marker=dict(color=cor_princ, size=10, opacity=0.8, line=dict(width=1, color=cor_borda)), showlegend=False))
                                fig_viz.add_trace(go.Scatter(x=df_final[col_trat], y=df_final['Mediana'], mode='markers', marker=dict(symbol='line-ew', size=40, color=cor_txt, line=dict(width=3)), name='Mediana', hoverinfo='y'))
                            
                            elif "Violin" in tipo_grafico:
                                fig_viz.add_trace(go.Violin(x=df_proc[col_trat], y=df_proc[col_resp], name="Dados", line_color=cor_princ, box_visible=True, points='all', fillcolor=cor_princ, opacity=0.6))

                            # Layout Global
                            show_line = True if cfg['estilo_borda'] != "Sem Bordas" else False
                            mirror_bool = True if cfg['estilo_borda'] == "Caixa (Espelhado)" else False
                            mostrar_ticks = cfg.get('mostrar_ticks', True)

                            fig_viz.update_layout(
                                title=dict(text=f"<b>{cfg['titulo_custom']}</b>", x=0.5, font=dict(size=cfg['font_size']+4, color=cor_txt)),
                                paper_bgcolor=cfg['cor_fundo'], plot_bgcolor=cfg['cor_fundo'], height=cfg['altura'],
                                font=dict(family=cfg['font_family'], size=cfg['font_size'], color=cor_txt), showlegend=False,
                                yaxis=dict(title=cfg['label_y'], showgrid=cfg['mostrar_grid'], gridcolor=cfg['cor_grade'], showline=show_line, linewidth=1, linecolor=cor_borda, mirror=mirror_bool, tickfont=dict(color=cor_txt, size=cfg['font_size']), showticklabels=True, ticks='outside' if mostrar_ticks else ''),
                                xaxis=dict(title=cfg['label_x'], showgrid=False, showline=show_line, linewidth=1, linecolor=cor_borda, mirror=mirror_bool, tickfont=dict(color=cor_txt, size=cfg['font_size']), categoryorder='array', categoryarray=ordem_trats, showticklabels=True, ticks='outside' if mostrar_ticks else '')
                            )
                            if cfg['mostrar_subgrade']: fig_viz.update_yaxes(minor=dict(showgrid=True, gridcolor=cfg['cor_subgrade'], gridwidth=0.5))
                            
                            st.plotly_chart(fig_viz, use_container_width=True, key=f"chart_np_final_{i}")
                            grafico_final_obj = fig_viz # Salva para o relatório

                            if st.button("Ocultar Resultado", key=f"hide_np_{col_resp_original}"):
                                st.session_state[key_run_np] = False; st.rerun()

                # --- CAPTURA DE DADOS PARA O RELATÓRIO (CRUCIAL) ---
                if 'dados_para_relatorio_final' in locals():
                    p_shap_val = res_analysis.get('shapiro', (0,0))[1] if res_analysis else 0
                    p_lev_val = res_analysis.get('levene', (0,0))[1] if res_analysis else 0
                    
                    conclusao_txt = "Não houve diferença estatística."
                    if 'p_final_trat' in locals() and p_final_trat < 0.05: conclusao_txt = "Diferença significativa detectada (ANOVA)."
                    elif not analise_valida and 'p_np' in locals() and p_np < 0.05: conclusao_txt = "Diferença significativa detectada (Não-Paramétrica)."

                    dados_para_relatorio_final.append({
                        "var": col_resp,
                        "metodo": "ANOVA (Paramétrica)" if analise_valida else "Não-Paramétrica",
                        "transf": transf_atual,
                        "p_norm": p_shap_val,
                        "p_homog": p_lev_val,
                        "cv": f"{cv_val:.2f}%" if 'cv_val' in locals() else "-",
                        "conclusao": conclusao_txt,
                        "tabela_medias": tabela_final_obj,
                        "grafico": grafico_final_obj,
                        "data_hora": pd.Timestamp.now().strftime('%d/%m/%Y')
                    })
# ==============================================================================
# 🏁 FIM DO BLOCO 20
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 21: Análise de Correlação (Multivariada)
# ==============================================================================

# TRAVA DE SEGURANÇA PRINCIPAL: O bloco só é lido se a análise principal já tiver rodado
if st.session_state.get('processando', False):

    # --- 1. FUNÇÃO AUXILIAR DE PERSONALIZAÇÃO ---
    def mostrar_editor_heatmap(key_prefix):
        """
        Menu para personalizar cores, textos e bordas do Heatmap.
        Encapsulado em formulário.
        """
        with st.expander("✏️ Personalizar Gráfico de Correlação", expanded=False):
            with st.form(key=f"form_{key_prefix}"):
                st.markdown("##### 🎨 Aparência Geral")
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown("**Cores do Mapa (Gradiente)**")
                    cor_neg = st.color_picker("Valor -1 (Negativo)", "#D73027", key=f"{key_prefix}_cneg")
                    cor_zero = st.color_picker("Valor 0 (Neutro)", "#FFFFFF", key=f"{key_prefix}_czero")
                    cor_pos = st.color_picker("Valor 1 (Positivo)", "#4575B4", key=f"{key_prefix}_cpos")
                    
                with c2:
                    st.markdown("**Fundo e Eixos**")
                    cor_fundo = st.color_picker("Fundo do Gráfico", "#FFFFFF", key=f"{key_prefix}_cbg")
                    cor_eixos = st.color_picker("Cor Eixos/Título/Legenda", "#000000", key=f"{key_prefix}_ceixos")
                    fam_fonte = st.selectbox("Fonte", ["Arial", "Verdana", "Times New Roman", "Courier New"], key=f"{key_prefix}_font")
                    
                with c3:
                    st.markdown("**Estrutura**")
                    titulo_custom = st.text_input("Título", "Matriz de Correlação", key=f"{key_prefix}_tit")
                    estilo_borda = st.selectbox("Bordas", ["Caixa (Espelhado)", "Apenas L (Eixos)", "Sem Bordas"], key=f"{key_prefix}_borda")
                    c3a, c3b = st.columns(2)
                    with c3a: mostrar_ticks = st.checkbox("Ticks", False, key=f"{key_prefix}_ticks")
                    with c3b: eixos_negrito = st.checkbox("Negrito", False, key=f"{key_prefix}_boldax")

                st.markdown("---")
                st.markdown("##### 🔢 Valores Internos (Texto nas Células)")
                
                c4, c5, c6 = st.columns(3)
                with c4:
                    modo_cor_txt = st.selectbox("Modo de Cor", 
                                                ["Cor Única", "Condicional (Pos/Neg/Zero)"], 
                                                key=f"{key_prefix}_modotxt")
                    tamanho_fonte_val = st.number_input("Tamanho Fonte", 8, 24, 12, key=f"{key_prefix}_fsize")
                    
                with c5:
                    val_negrito = st.checkbox("Valores em Negrito", False, key=f"{key_prefix}_boldval")
                
                cores_texto = {}
                with c6:
                    if modo_cor_txt == "Cor Única":
                        cores_texto['unica'] = st.color_picker("Cor Texto Única", "#000000", key=f"{key_prefix}_ctxtuni")
                    else:
                        c6a, c6b = st.columns(2)
                        with c6a:
                            cores_texto['pos'] = st.color_picker("Txt Positivo", "#0000FF", key=f"{key_prefix}_ctxtpos")
                            cores_texto['neg'] = st.color_picker("Txt Negativo", "#FF0000", key=f"{key_prefix}_ctxtneg")
                        with c6b:
                            cores_texto['zero'] = st.color_picker("Txt Zero", "#AAAAAA", key=f"{key_prefix}_ctxtzero")

                st.markdown("---")
                submit_btn = st.form_submit_button("🔄 Atualizar Gráfico de Correlação")

            return {
                "cor_mapa": [cor_neg, cor_zero, cor_pos],
                "cor_fundo": cor_fundo,
                "titulo": titulo_custom,
                "fonte": fam_fonte,
                "cor_eixos": cor_eixos,
                "eixos_negrito": eixos_negrito,
                "estilo_borda": estilo_borda,
                "ticks": mostrar_ticks,
                "modo_cor_txt": modo_cor_txt,
                "tamanho_fonte_val": tamanho_fonte_val,
                "val_negrito": val_negrito,
                "cores_texto": cores_texto
            }

    # --- 2. PREPARAÇÃO E CORREÇÃO DOS DADOS ---
    df_corr_input = None
    if 'df_analise' in locals():
        df_corr_input = df_analise.copy()
    elif 'df' in locals() and df is not None:
        df_corr_input = df.copy()

    # Só executa se tivermos dados e variáveis selecionadas na lista
    if df_corr_input is not None and 'lista_resps' in locals() and lista_resps:
        
        # --- CONVERSÃO FORÇADA (Trata vírgulas como pontos antes de checar numéricos) ---
        for col in lista_resps:
            try:
                df_corr_input[col] = limpar_e_converter_dados(df_corr_input, col)
            except:
                pass 

        # --- FILTRAGEM (Agora detectará corretamente os numéricos) ---
        cols_numericas_corr = df_corr_input.select_dtypes(include=[np.number]).columns.tolist()
        vars_corr = [v for v in lista_resps if v in cols_numericas_corr]

        # --- LÓGICA ORIGINAL (ESTÉTICA PRESERVADA) ---
        if len(vars_corr) > 1:
            st.markdown("---")
            st.header("🔗 Análise de Correlação entre Variáveis")
            
            # 1. Menu de Configuração (Original)
            cfg = mostrar_editor_heatmap("corr_main")
            
            # Lógica de Seleção do Método
            metodo_corr = st.radio(
                "Método de Correlação:", 
                ["Pearson (Paramétrico)", "Spearman (Não-Paramétrico)"], 
                horizontal=True,
                index=1 # Spearman selecionado por padrão para segurança
            )
            metodo = "pearson" if "Pearson" in metodo_corr else "spearman"

            # --- AVISO EDUCATIVO (ORIENTAÇÃO AO USUÁRIO) ---
            if metodo == "pearson":
                st.warning("""
                ⚠️ **ATENÇÃO:** O método de **Pearson** é sensível a dados que não seguem distribuição normal. 
                Se o seu conjunto de dados contiver variáveis **Não-Paramétricas** (ou uma mistura de Paramétricas e Não-Paramétricas), 
                o uso de Pearson pode gerar correlações imprecisas. Na dúvida ou em dados mistos, prefira **Spearman**.
                """)
            else:
                st.success("✅ **Ótima escolha:** O método de **Spearman** (correlação de postos) é robusto e adequado tanto para dados normais quanto para dados não-paramétricos.")

            # VAR DE CONTROLE INTERNO (BOTÃO DE AÇÃO)
            # O gráfico só roda se o usuário clicar no botão do formulário (submit_btn) ou se já tiver rodado antes
            if 'executar_corr_btn' not in st.session_state: st.session_state['executar_corr_btn'] = False
            
            # Se o botão do formulário foi clicado, ativa a visualização
            # Nota: cfg é o retorno do form, que contém o submit_btn implícito pelo fluxo do streamlit form, 
            # mas aqui usamos o retorno do form.submit no final da função acima? 
            # Ajuste: A função mostrar_editor_heatmap retorna o dicionário configs. O botão está dentro dela.
            # O Streamlit rerun acontece quando o botão do form é clicado.
            
            # Para garantir que o gráfico apareça após clicar, verificamos se o form foi submetido.
            # Como a função retorna configs, assumimos que se estamos aqui, o script rodou.
            # Para economizar recursos, podemos usar um st.button EXTERNO ao form se quisermos travar o cálculo,
            # mas você pediu um botão "Atualizar". O botão dentro do form já faz isso!
            
            # Então, vamos rodar o cálculo (que é pesado) SEMPRE QUE O SCRIPT PASSAR AQUI?
            # Não, você quer travar. Vamos adicionar uma trava extra visual.
            
            container_grafico = st.container()
            
            # Cálculo da Matriz (Usando o dataframe corrigido)
            try:
                # O cálculo pesado acontece aqui. 
                # Se quiser evitar que rode automaticamente na primeira vez que abre a seção (após Rodar Dados),
                # podemos colocar um botão inicial "Gerar Matriz".
                
                if 'matriz_gerada' not in st.session_state: st.session_state['matriz_gerada'] = False
                
                if not st.session_state['matriz_gerada']:
                    if st.button("🔄 Gerar Gráfico de Correlação", type="primary"):
                        st.session_state['matriz_gerada'] = True
                        st.rerun()
                
                if st.session_state['matriz_gerada']:
                    df_corr = df_corr_input[vars_corr].corr(method=metodo)
                    
                    # 3. Definição da Escala de Cores do Fundo
                    colorscale_custom = [
                        [0.0, cfg['cor_mapa'][0]], # -1
                        [0.5, cfg['cor_mapa'][1]], # 0
                        [1.0, cfg['cor_mapa'][2]]  # 1
                    ]
                    
                    # 4. PREPARAÇÃO DO TEXTO CUSTOMIZADO (HTML)
                    custom_text = []
                    vals = df_corr.values
                    
                    for i in range(len(vals)):
                        row_text = []
                        for val in vals[i]:
                            # Define a cor
                            c_code = "#000000"
                            if cfg['modo_cor_txt'] == "Cor Única":
                                c_code = cfg['cores_texto']['unica']
                            else:
                                if val > 0.001: c_code = cfg['cores_texto']['pos']
                                elif val < -0.001: c_code = cfg['cores_texto']['neg']
                                else: c_code = cfg['cores_texto']['zero']
                            
                            # Define Negrito
                            val_fmt = f"{val:.2f}"
                            if cfg['val_negrito']:
                                val_fmt = f"<b>{val_fmt}</b>"
                            
                            # Cria o HTML final para a célula
                            cell_html = f"<span style='color:{c_code}'>{val_fmt}</span>"
                            row_text.append(cell_html)
                        custom_text.append(row_text)

                    # 5. Geração do Gráfico (Sem text_auto)
                    fig_corr = px.imshow(
                        df_corr,
                        text_auto=False, # Desligamos o auto para usar nosso custom_text
                        aspect="auto",
                        color_continuous_scale=colorscale_custom,
                        zmin=-1, zmax=1
                    )
                    
                    # 6. Personalização Avançada (Layout)
                    mirror_bool = True if cfg['estilo_borda'] == "Caixa (Espelhado)" else False
                    show_line = False if cfg['estilo_borda'] == "Sem Bordas" else True
                    tick_mode = "outside" if cfg['ticks'] else ""
                    weight_eixos = "bold" if cfg['eixos_negrito'] else "normal"
                    title_text = f"<b>{cfg['titulo']}</b>" if cfg['eixos_negrito'] else cfg['titulo']
                    
                    fig_corr.update_layout(
                        title=dict(
                            text=title_text,
                            x=0.5,
                            font=dict(family=cfg['fonte'], size=18, color=cfg['cor_eixos'])
                        ),
                        height=500,
                        paper_bgcolor=cfg['cor_fundo'], 
                        plot_bgcolor=cfg['cor_fundo'],
                        font=dict(family=cfg['fonte'], color=cfg['cor_eixos']),
                        xaxis=dict(
                            showline=show_line, mirror=mirror_bool, linecolor=cfg['cor_eixos'], linewidth=1,
                            ticks=tick_mode, tickcolor=cfg['cor_eixos'],
                            tickfont=dict(family=cfg['fonte'], color=cfg['cor_eixos'], weight=weight_eixos)
                        ),
                        yaxis=dict(
                            showline=show_line, mirror=mirror_bool, linecolor=cfg['cor_eixos'], linewidth=1,
                            ticks=tick_mode, tickcolor=cfg['cor_eixos'],
                            tickfont=dict(family=cfg['fonte'], color=cfg['cor_eixos'], weight=weight_eixos)
                        )
                    )
                    
                    # Atualização da Legenda Lateral
                    fig_corr.update_coloraxes(
                        colorbar=dict(
                            tickfont=dict(
                                family=cfg['fonte'],
                                color=cfg['cor_eixos'], 
                                size=cfg['tamanho_fonte_val'], 
                                weight=weight_eixos
                            ),
                            title=dict(text="")
                        )
                    )

                    # 7. Injeção do Texto HTML
                    fig_corr.update_traces(
                        text=custom_text, 
                        texttemplate="%{text}",
                        textfont=dict(
                            family=cfg['fonte'],
                            size=cfg['tamanho_fonte_val']
                        )
                    )

                    # --- EXIBIÇÃO FINAL ---
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.dataframe(df_corr.style.format("{:.2f}"), use_container_width=True)

                    # --- NOTA DE RODAPÉ ADAPTATIVA ---
                    st.markdown(f"""
                    <div style="
                        font-family: 'Times New Roman', Times, serif; 
                        font-size: 0.9em; 
                        border-top: 1px solid rgba(128, 128, 128, 0.5); 
                        margin-top: 10px; 
                        padding-top: 8px; 
                        text-align: justify;">
                        <b>Nota:</b> A matriz acima apresenta os coeficientes de correlação ({'<i>r</i> de Pearson' if metodo == 'pearson' else '<i>ρ</i> de Spearman'}) 
                        entre as variáveis analisadas. O coeficiente varia no intervalo <b>[-1, +1]</b>. 
                        Valores próximos a <b>+1</b> indicam forte associação linear positiva (proporcionalidade direta), 
                        enquanto valores próximos a <b>-1</b> indicam forte associação linear negativa (proporcionalidade inversa). 
                        Coeficientes próximos a <b>0</b> sugerem ausência de correlação linear significativa.
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Não foi possível calcular a correlação: {e}")
# ==============================================================================
# 🏁 FIM DO BLOCO 21
# ==============================================================================
   
   
# ==============================================================================
# 📂 BLOCO 22: Gerador de Relatório Completo (HTML Download)
# ==============================================================================
    # ATENÇÃO: Esta parte fica FORA do loop (alinhada à esquerda do IF principal)
    
    if 'dados_para_relatorio_final' in locals() and dados_para_relatorio_final:
        st.markdown("---")
        st.header("📑 Central de Relatórios")
        st.success(f"✅ Processamento concluído de {len(dados_para_relatorio_final)} variáveis.")
        st.info("O botão abaixo gera um relatório completo com **Gráficos, Tabelas e Laudos** que você pode salvar como PDF.")
        
        def fig_to_html(fig):
            if fig: return fig.to_html(full_html=False, include_plotlyjs='cdn', default_height='450px')
            return "<div style='color:#999;'>Gráfico não gerado automaticamente (verifique abas).</div>"

        def df_to_html(df):
            if df is not None: return df.to_html(classes='table table-striped', float_format="%.2f", justify='center')
            return "<div style='color:#999;'>Tabela não disponível.</div>"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório AgroStat Pro</title>
            <meta charset="utf-8">
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; padding: 40px; background: #f4f4f4; }}
                .container {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 0 15px rgba(0,0,0,0.1); }}
                h1 {{ color: #2E86C1; text-align: center; margin-bottom: 10px; }}
                .var-section {{ margin-bottom: 60px; border-bottom: 2px solid #eee; padding-bottom: 40px; }}
                h2 {{ color: #28B463; border-left: 5px solid #28B463; padding-left: 15px; }}
                .metric-box {{ background: #f8f9fa; border: 1px solid #ddd; padding: 15px; margin-top: 20px; }}
                .table-container {{ margin-top: 20px; overflow-x: auto; }}
                table {{ width: 100%; text-align: center; }}
                th {{ background-color: #2E86C1; color: white; }}
                @media print {{ .no-print {{ display: none; }} .var-section {{ page-break-inside: avoid; }} }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="no-print" style="text-align:right; margin-bottom:20px;">
                    <button class="btn btn-primary btn-lg" onclick="window.print()">🖨️ Imprimir / Salvar como PDF</button>
                </div>
                <h1>🌱 AgroStat Pro - Relatório de Análise</h1>
                <p style="text-align:center;">Data: {dados_para_relatorio_final[0]['data_hora']}</p>
                <hr>
        """

        for item in dados_para_relatorio_final:
            html_content += f"""
            <div class="var-section">
                <h2>Variável: {item['var']}</h2>
                <div class="row metric-box">
                    <div class="col-md-3"><strong>Método:</strong><br>{item['metodo']}</div>
                    <div class="col-md-3"><strong>Transformação:</strong><br>{item['transf']}</div>
                    <div class="col-md-3"><strong>Normalidade (P):</strong><br>{item['p_norm']:.4f}</div>
                    <div class="col-md-3"><strong>CV (%):</strong><br>{item['cv']}</div>
                </div>
                <div class="alert alert-info" style="margin-top: 15px;"><strong>Conclusão:</strong> {item['conclusao']}</div>
                <div class="row">
                    <div class="col-md-5">
                        <h5 style="margin-top:20px;">📋 Resultados</h5>
                        <div class="table-container">{df_to_html(item['tabela_medias'])}</div>
                    </div>
                    <div class="col-md-7">
                        <h5 style="margin-top:20px;">📊 Gráfico</h5>
                        <div style="border:1px solid #eee; padding:10px;">{fig_to_html(item['grafico'])}</div>
                    </div>
                </div>
            </div>
            """

        html_content += "</div></body></html>"

        st.download_button(
            label="📥 Baixar Relatório Completo (HTML com Gráficos)",
            data=html_content,
            file_name="Relatorio_AgroStat.html",
            mime="text/html"
        )
# ==============================================================================
# 🏁 FIM DO BLOCO 22
# ==============================================================================


# ==============================================================================
# 📂 BLOCO 23: Planejamento (Sorteio Experimental)
# ==============================================================================
import random
import pandas as pd
import itertools

if modo_app == "🎲 Sorteio Experimental":
    st.title("🎲 Sorteio Experimental")
    st.markdown("Gere sua planilha de campo com numeração personalizada e identificação do ensaio.")

    # --- INPUTS DE ESTRUTURA ---
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
    
    # --- CONFIGURAÇÃO DE RÓTULOS (NOVA FUNCIONALIDADE) ---
    st.markdown("#### 🎨 Personalização de Rótulos")
    st.caption("Defina como os nomes aparecerão na planilha final.")
    
    c_rot1, c_rot2 = st.columns(2)
    with c_rot1:
        # Define o padrão sugerido baseado no delineamento
        padrao_prefixo = "Bloco " if "DBC" in tipo_exp else ""
        label_desc = "Blocos" if "DBC" in tipo_exp else "Repetições"
        
        prefixo_grupo = st.text_input(
            f"Prefixo para {label_desc}", 
            value=padrao_prefixo, 
            help="Ex: 'Bloco ' gera 'Bloco 1'. Deixe vazio para gerar apenas números (1, 2, 3)."
        )
    
    with c_rot2:
        prefixo_id = st.text_input(
            "Prefixo para ID da Parcela (Opcional)", 
            value="", 
            help="Ex: Digite 'P-' para gerar IDs como 'P-101', 'P-102'. Útil para etiquetas."
        )

    st.markdown("---")
    
    # --- LÓGICA DE NUMERAÇÃO AVANÇADA ---
    st.markdown("#### 🏷️ Configuração de Numeração")
    
    usar_salto = False
    salto_val = 100
    
    if "DBC" in tipo_exp:
        c_num1, c_num2 = st.columns([1, 2])
        with c_num1:
            usar_salto = st.checkbox("Saltar numeração por Bloco?", value=False, help="Ex: Bloco 1 (101..), Bloco 2 (201..)")
        
        with c_num2:
            if usar_salto:
                col_s1, col_s2 = st.columns(2)
                with col_s1: 
                    num_inicial = st.number_input("Início (1º Bloco)", value=101, step=1)
                with col_s2: 
                    salto_val = st.number_input("Salto (Entre Blocos)", value=100, step=100)
            else:
                num_inicial = st.number_input("Nº Inicial Sequencial", value=1, min_value=0)
    else:
        usar_salto = False 
        num_inicial = st.number_input("Nº Inicial Sequencial", value=1, min_value=0)

    st.markdown("---")
    
    # --- SELETOR DE MODO ---
    tipo_entrada = st.radio("Como definir os tratamentos?", ["📝 Lista Simples", "✖️ Esquema Fatorial (A x B ...)"], horizontal=True)
    
    with st.form("form_dados_trats"):
        lista_trats_final = []
        
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

    if submitted:
        # Processamento dos Tratamentos
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
            parcelas = []
            info_blocos = []
            info_reps = [] 
            
            # Sorteio e Geração de Rótulos Personalizados
            if "DIC" in tipo_exp:
                base_trats = lista_trats_final * n_reps
                random.shuffle(base_trats)
                parcelas = base_trats
                contadores = {t: 0 for t in lista_trats_final}
                for t in parcelas:
                    contadores[t] += 1
                    # APLICAÇÃO DO PREFIXO PERSONALIZADO (DIC)
                    info_reps.append(f"{prefixo_grupo}{contadores[t]}")
            else: # DBC
                for i in range(n_reps):
                    bloco = lista_trats_final.copy()
                    random.shuffle(bloco) 
                    parcelas.extend(bloco)
                    # APLICAÇÃO DO PREFIXO PERSONALIZADO (DBC)
                    info_blocos.extend([f"{prefixo_grupo}{i+1}"] * len(bloco))
            
            # Geração de IDs
            total_sorteadas = len(parcelas)
            ids_personalizados = []
            
            if usar_salto:
                n_trats_por_bloco = len(lista_trats_final)
                for i in range(total_sorteadas):
                    bloco_idx = i // n_trats_por_bloco 
                    item_idx = (i % n_trats_por_bloco) + 1 
                    val_num = num_inicial + (bloco_idx * salto_val) + (item_idx - 1)
                    ids_personalizados.append(f"{prefixo_id}{val_num}")
            else:
                for i in range(total_sorteadas):
                    val_num = num_inicial + i
                    ids_personalizados.append(f"{prefixo_id}{val_num}")
            
            # Montagem do DataFrame
            dados_planilha = {"ID_Parcela": ids_personalizados}
            
            if "DBC" in tipo_exp:
                dados_planilha["Bloco"] = info_blocos
            else:
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
# 🏁 FIM DO BLOCO 23
# ==============================================================================

# ==============================================================================
# 📂 BLOCO 24: Rodapé e Créditos (GLOBAL)
# ==============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 20px;">
        <hr style="margin-bottom: 15px; border-top: 1px solid #eee;">
        Developed by <b>Rafael Novais de Miranda</b><br>
        📧 rafaelnovaismiranda@gmail.com | 📱 +55 (34) 9.99777-9966
    </div>
    """,
    unsafe_allow_html=True
)
# ==============================================================================
# 🏁 FIM DO BLOCO 24
# ==============================================================================
