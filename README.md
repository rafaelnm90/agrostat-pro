# 🌱 AgroStat Pro: Plataforma de Análise Estatística Experimental

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Stable%20v6.3-success)
![Science](https://img.shields.io/badge/Science-Data%20Analysis-lightgrey)

**AgroStat Pro** é uma aplicação web desenvolvida para automatizar e democratizar a análise estatística de experimentos agronômicos. Focada em precisão científica e usabilidade, a ferramenta processa Delineamentos Inteiramente Casualizados (DIC) e Blocos Casualizados (DBC), realizando desde a ANOVA até testes de comparação múltipla complexos e Análise Conjunta de Experimentos (MET).

---

## 🎯 Funcionalidades Principais

### 1. Processamento em Lote (Batch Processing)
- Capacidade de analisar **múltiplas variáveis resposta** simultaneamente.
- Geração automática de relatórios individuais para cada variável (Produtividade, Altura, etc.) em containers expansíveis.

### 2. Estatística Experimental Robusta
- **ANOVA (Análise de Variância):** Quadro completo com SQ, GL, QM, F-calc e P-valor.
- **Teste de Tukey (HSD):** Implementação personalizada do algoritmo *Studentized Range* para máxima precisão em delineamentos DBC.
- **Teste de Scott-Knott:** Algoritmo de agrupamento de médias ideal para grande número de tratamentos.
- **Interpretação Automática:** Geração de textos que explicam os rankings ("Líder Numérico", "Empate Estatístico").

### 3. Análise Conjunta (Multi-Environment Trials - MET)
- **Detecção Automática:** O sistema identifica se há múltiplos locais/ambientes no dataset.
- **Homogeneidade de Variâncias:** Cálculo automático da razão entre o maior e menor QM Resíduo, com alertas baseados no critério de Pimentel-Gomes (< 7:1).
- **Interação GxE:** Diagnóstico automático da interação Genótipo x Ambiente.
- **Desdobramento:** Em caso de interação significativa, o software realiza o desdobramento da interação automaticamente, gerando rankings por local.

### 4. Métricas e Diagnósticos
- **Classificação de CV:** O Coeficiente de Variação é classificado automaticamente (Baixo, Médio, Alto, Muito Alto) seguindo as normas de **Pimentel-Gomes (2009)**.
- **Pressupostos:** Testes de Normalidade (Shapiro-Wilk) e Homocedasticidade (Bartlett).

---

## 🧠 Diferenciais Técnicos (Engine Matemático)

Diferente de scripts básicos, o AgroStat Pro possui um **motor estatístico customizado** para garantir paridade com softwares de referência (R/Sisvar):

* **Algoritmo de Letras (Graph Theory):** Utiliza o algoritmo de **Bron-Kerbosch** para encontrar cliques máximos em grafos de adjacência, garantindo que as letras de agrupamento (ex: "ab", "bc") sejam geradas sem erros de lógica, mesmo em casos de alta sobreposição.
* **Distribuição Studentized Range:** Substituição da biblioteca padrão `statsmodels` pela `scipy.stats.studentized_range` para cálculo exato do valor crítico $q$, corrigindo distorções em delineamentos de blocos.

---

## 📸 Screenshots

*(Espaço reservado para inserir prints da tela do Dashboard, Quadro da ANOVA e Gráficos)*

---

## 🚀 Como Rodar Localmente

Pré-requisitos: Python 3.9+ instalado.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/rafaelnm90/agrostat-pro.git](https://github.com/rafaelnm90/agrostat-pro.git)
    cd agrostat-pro
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute a aplicação:**
    ```bash
    python -m streamlit run agrostatpro.py
    ```

---

## 📊 Formato dos Dados (Input)

O sistema aceita arquivos `.csv` ou `.xlsx` no formato "tidy data" (formato longo):

| Tratamento | Local    | Bloco | Produtividade | Altura |
|------------|----------|-------|---------------|--------|
| Genotipo_A | Lavras   | 1     | 4500          | 2.5    |
| Genotipo_A | Lavras   | 2     | 4600          | 2.6    |
| ...        | ...      | ...   | ...           | ...    |

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **Interface:** Streamlit
* **Cálculo Científico:** NumPy, Scipy, Statsmodels
* **Visualização:** Plotly Express
* **Manipulação de Dados:** Pandas

---

## 👨‍🔬 Sobre o Desenvolvedor

Desenvolvido por:
**Rafael Novais de Miranda**, Doutor em Genética e Melhoramento de Plantas (UFLA) e Cientista de Dados em formação.

Este projeto une o rigor da estatística acadêmica com a agilidade da engenharia de software moderna, visando solucionar gargalos reais na análise de dados agronômicos.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/rafaelnovais/)
