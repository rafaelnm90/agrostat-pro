import pandas as pd
import numpy as np

# Configuração para reprodutibilidade
np.random.seed(42)

print("🌽 Gerando dados complexos de milho...")

# --- PARÂMETROS DO EXPERIMENTO ---
n_tratamentos = 50  # 50 Cultivares diferentes
n_blocos = 4        # 4 Repetições
media_base = 6000   # Produtividade média base (kg/ha)

# --- 1. CRIANDO OS TRATAMENTOS (COM DIFERENÇAS REAIS) ---
# Vamos dividir os 50 tratamentos em 3 grupos genéticos:
# Grupo A (Elite): Produzem muito (+1500 kg)
# Grupo B (Comercial): Produzem médio (+0 kg)
# Grupo C (Antigo): Produzem pouco (-1500 kg)

trats = [f"GEN_{i:02d}" for i in range(1, n_tratamentos + 1)]
efeitos_trat = []

for i in range(n_tratamentos):
    if i < 10:   # Top 10 Elite
        efeito = 1500 + np.random.normal(0, 100) # Variação dentro do grupo
    elif i < 30: # Intermediários
        efeito = 0 + np.random.normal(0, 100)
    else:        # Inferiores
        efeito = -1500 + np.random.normal(0, 100)
    efeitos_trat.append(efeito)

dict_efeitos_trat = dict(zip(trats, efeitos_trat))

# --- 2. CRIANDO OS BLOCOS (EFEITO AMBIENTAL) ---
# O solo não é uniforme. O Bloco 1 é melhor que o 4.
blocos = [1, 2, 3, 4]
efeitos_bloco = {
    1: 500,  # Bloco muito fértil
    2: 200,  # Bloco bom
    3: -100, # Bloco regular
    4: -600  # Bloco com mancha de cascalho (ruim)
}

# --- 3. GERANDO AS PARCELAS ---
dados = []

for b in blocos:
    for t in trats:
        # Modelo Estatístico: Y = Média + Trat + Bloco + Erro
        erro_experimental = np.random.normal(0, 300) # Desvio padrão de 300kg
        
        produtividade = (
            media_base + 
            dict_efeitos_trat[t] + 
            efeitos_bloco[b] + 
            erro_experimental
        )
        
        # Arredondar para ficar bonito
        produtividade = round(produtividade, 2)
        
        dados.append({
            "Cultivar": t,
            "Bloco": b,
            "Produtividade_kg_ha": produtividade
        })

# --- 4. SALVANDO ---
df = pd.DataFrame(dados)
nome_arquivo = "dados_milho_vcu.csv"
df.to_csv(nome_arquivo, index=False)

print(f"✅ Sucesso! Arquivo '{nome_arquivo}' gerado com {len(df)} linhas.")
print(f"   Média Geral Simulada: {df['Produtividade_kg_ha'].mean():.2f} kg/ha")
print(f"   Amplitude: {df['Produtividade_kg_ha'].min()} a {df['Produtividade_kg_ha'].max()} kg/ha")