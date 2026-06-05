import streamlit as st
import pandas as pd
import quantstats as qs
import base64
import os
import tempfile
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# CSS customizado
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def apply_pdf_chart_theme(fig, ax):
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8fafc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.tick_params(colors='#475569', labelsize=8)
    ax.yaxis.label.set_color('#475569')
    ax.xaxis.label.set_color('#475569')
    ax.title.set_color('#0f172a')
    ax.grid(True, color='#e2e8f0', linestyle=':', alpha=0.8)
# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
    <div class="page-hero-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 60 60" fill="none">
          <!-- Document body -->
          <rect x="10" y="4" width="40" height="52" rx="5" fill="#0e1524" stroke="#1e293b" stroke-width="1.5"/>
          <!-- Fold corner -->
          <path d="M38 4 L50 16 L38 16 Z" fill="#080c14" stroke="#1e293b" stroke-width="1"/>
          <!-- Header line (green accent) -->
          <line x1="16" y1="24" x2="44" y2="24" stroke="#00ff87" stroke-width="2.5" stroke-linecap="round"/>
          <!-- Body text lines -->
          <line x1="16" y1="31" x2="40" y2="31" stroke="#334155" stroke-width="1.5" stroke-linecap="round"/>
          <line x1="16" y1="36" x2="36" y2="36" stroke="#334155" stroke-width="1.5" stroke-linecap="round"/>
          <!-- Mini bar chart (bottom of doc) -->
          <rect x="16" y="47" width="5" height="7"  rx="1.5" fill="#00ff87"/>
          <rect x="24" y="43" width="5" height="11" rx="1.5" fill="#00d2ff"/>
          <rect x="32" y="45" width="5" height="9"  rx="1.5" fill="#ffd600"/>
          <rect x="40" y="40" width="5" height="14" rx="1.5" fill="#a855f7"/>
        </svg>
    </div>
    <div class="page-hero-content">
        <h1 class="page-hero-title">Relatório Completo do Portfólio</h1>
        <p class="page-hero-subtitle">Exporte uma análise quantitativa completa via QuantStats — com Sharpe, Drawdowns, Retornos Anuais e muito mais.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Verifica se as variáveis necessárias já estão no session_state
required_keys = ["modo", "returns", "pesos_manuais", "peso_manual_df"]
for key in required_keys:
    if key not in st.session_state:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.warning("Configure primeiro seu portfólio na aba Portfolio para liberar a geração do relatório.")
        st.stop()

modo = st.session_state["modo"]
returns = st.session_state["returns"]
pesos_manuais = st.session_state["pesos_manuais"]
peso_manual_df = st.session_state["peso_manual_df"]
portfolio_returns = st.session_state["portfolio_returns"]
retorno_bench = st.session_state["retorno_bench"]

          
# Converte para formato aceito pelo QuantStats
portfolio_returns.index = pd.to_datetime(portfolio_returns.index, errors='coerce')
portfolio_returns = portfolio_returns.tz_localize(None)  # Remove timezone

# Cria arquivo temporário
with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
    tmp_path = tmpfile.name

# Carregamento página
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("""
    <div class="loading-container">
        <div class="loading-spinner"></div>
        <div class="loading-text">Gerando relatório estatístico detalhado (QuantStats)... Isso pode levar alguns segundos.</div>
    </div>
    """, unsafe_allow_html=True)

qs.reports.html(
    portfolio_returns,
    benchmark=retorno_bench,
    output=tmp_path,
    title="Relatório Completo do Portfólio",
    download_filename="relatorio_portfolio.html")

loading_placeholder.empty()
    
# Botão para download HTML
with open(tmp_path, "rb") as f:
    html_data = f.read()

# Gera PDF resumo
from fpdf import FPDF
import datetime
import fundamentus
import re

def clean_txt(text):
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        'ã': 'a', 'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a',
        'Õ': 'O', 'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
        'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'Ç': 'C',
        'Ã': 'A', 'Á': 'A', 'À': 'A', 'Â': 'A', 'Ä': 'A',
        'Õ': 'O', 'Ó': 'O', 'Ò': 'O', 'Ô': 'O', 'Ö': 'O',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'Í': 'I', 'Ì': 'I', 'Î': 'I', 'Ï': 'I',
        'Ú': 'U', 'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'º': 'o', 'ª': 'a', '﹪': '%', '—': '-', '–': '-',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('latin-1', 'replace').decode('latin-1')

def clean_numeric_column(col):
    col = col.astype(str).str.strip()
    col = col.str.replace(r'[^0-9,.\-]', '', regex=True)
    col = col.str.replace(',', '.')
    return pd.to_numeric(col, errors='coerce')

@st.cache_data(ttl=3600)
def get_fundamentus_report_data(tickers_list):
    data_list = []
    for t in tickers_list:
        try:
            p = fundamentus.get_papel(t)
            data_list.append(p)
        except Exception:
            pass
    if data_list:
        return pd.concat(data_list)
    return pd.DataFrame()

# Obter tickers
tickers_yf = list(returns.columns)
tickers_clean = [t.replace(".SA", "") for t in tickers_yf]
df_fund = get_fundamentus_report_data(tickers_clean)

class PDF(FPDF):
    def header(self):
        # Accent top bar
        self.set_fill_color(14, 21, 36)
        self.rect(0, 0, 210, 4, 'F')
        
        # Title
        self.set_text_color(15, 23, 42)
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, clean_txt('B3 EXPLORER - RELATORIO DO PORTFOLIO'), 0, 1, 'L')
        
        # Subtitle
        self.set_text_color(71, 85, 105)
        self.set_font('Arial', 'I', 9)
        self.cell(0, 4, clean_txt('Analise Quantitativa, Simulacao Monte Carlo e Indicadores Fundamentalistas'), 0, 1, 'L')
        
        # Divider line
        self.set_draw_color(226, 232, 240)
        self.line(15, 24, 195, 24)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, clean_txt(f'Pagina {self.page_no()}'), 0, 0, 'C')
        self.cell(0, 10, clean_txt(datetime.date.today().strftime('%d/%m/%Y')), 0, 0, 'R')

def draw_section_header(pdf, title):
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, clean_txt(f"  {title}"), 0, 1, 'L', fill=True)
    pdf.ln(3)

def get_weight_val(row):
    for col in ["Pesos Otimizados", "Pesos", "Peso"]:
        if col in row.index:
            return row[col]
    return row.iloc[0]

pdf = PDF()
pdf.set_margins(15, 15, 15)
pdf.add_page()

# ─── SECÇÃO 1: CONFIGURAÇÃO E ALOCAÇÃO ───
draw_section_header(pdf, '1. CONFIGURACAO E ALOCACAO DO PORTFOLIO')
pdf.set_font('Arial', '', 9)
pdf.cell(0, 5, clean_txt(f"Modo de Alocacao: {st.session_state.get('modo', 'Nao disponivel')}"), 0, 1)
pdf.cell(0, 5, clean_txt(f"Periodo Historico (Lookback): {st.session_state.get('lookback', 'Nao disponivel')}"), 0, 1)
pdf.ln(3)

pdf.set_font('Arial', 'B', 9)
pdf.set_fill_color(226, 232, 240)
pdf.cell(30, 7, clean_txt('Ticker'), 1, 0, 'C', fill=True)
pdf.cell(110, 7, clean_txt('Empresa'), 1, 0, 'L', fill=True)
pdf.cell(40, 7, clean_txt('Peso (%)'), 1, 1, 'R', fill=True)

pdf.set_font('Arial', '', 9)
for idx, row in peso_manual_df.iterrows():
    t_clean = str(idx).replace('.SA', '')
    val = get_weight_val(row)
    
    emp_name = "-"
    if not df_fund.empty and t_clean in df_fund.index:
        emp_name = str(df_fund.loc[t_clean, 'Empresa'])
        
    pdf.cell(30, 6, clean_txt(t_clean), 1, 0, 'C')
    pdf.cell(110, 6, clean_txt(emp_name), 1, 0, 'L')
    pdf.cell(40, 6, clean_txt(f"{val*100:.2f}%"), 1, 1, 'R')
pdf.ln(6)

# ─── SECÇÃO 2: MÉTRICAS DE PERFORMANCE DA CARTEIRA ───
draw_section_header(pdf, '2. METRICAS DE PERFORMANCE DA CARTEIRA')
metrics_list = [
    ("Retorno Total do Periodo", f"{st.session_state.get('total_return', 0.0):.2f}%"),
    ("Volatilidade Anualizada", f"{st.session_state.get('vol_anual', 0.0):.2f}%"),
    ("Indice Sharpe", f"{st.session_state.get('sharpe_val', 0.0):.2f}"),
    ("Indice Sortino", f"{st.session_state.get('sortino_val', 0.0):.2f}"),
    ("Max Drawdown", f"{st.session_state.get('max_dd', 0.0):.2f}%"),
    ("Beta (em relacao ao Benchmark)", f"{st.session_state.get('beta', 0.0):.2f}"),
    ("Alfa de Jensen Anualizado", f"{st.session_state.get('alfa_val', 0.0)*100:.2f}%" if st.session_state.get('alfa_val') is not None else "-"),
    ("R-Quadrado (R2)", f"{st.session_state.get('r_quadrado', 0.0):.2f}"),
    ("Information Ratio", f"{st.session_state.get('information_ratio', 0.0):.2f}"),
]

pdf.set_font('Arial', 'B', 9)
pdf.set_fill_color(226, 232, 240)
pdf.cell(100, 7, clean_txt('Metrica de Performance'), 1, 0, 'L', fill=True)
pdf.cell(80, 7, clean_txt('Valor da Carteira'), 1, 1, 'R', fill=True)

pdf.set_font('Arial', '', 9)
for label, val_str in metrics_list:
    pdf.cell(100, 6, clean_txt(label), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(val_str), 1, 1, 'R')
pdf.ln(6)

# ─── SECÇÃO 3: ANÁLISE FUNDAMENTALISTA (se disponível) ───
if not df_fund.empty:
    draw_section_header(pdf, '3. INDICADORES FUNDAMENTALISTAS DAS EMPRESAS')
    
    pdf.set_font('Arial', 'B', 8)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(20, 7, clean_txt('Ticker'), 1, 0, 'C', fill=True)
    pdf.cell(50, 7, clean_txt('Setor'), 1, 0, 'L', fill=True)
    pdf.cell(22, 7, clean_txt('P/L'), 1, 0, 'R', fill=True)
    pdf.cell(22, 7, clean_txt('P/VP'), 1, 0, 'R', fill=True)
    pdf.cell(22, 7, clean_txt('DY (%)'), 1, 0, 'R', fill=True)
    pdf.cell(22, 7, clean_txt('ROE (%)'), 1, 0, 'R', fill=True)
    pdf.cell(22, 7, clean_txt('ROIC (%)'), 1, 1, 'R', fill=True)
    
    pdf.set_font('Arial', '', 8)
    for ticker_clean, row in df_fund.iterrows():
        pl_val = clean_numeric_column(pd.Series([row.get('PL', 0.0)])).fillna(0).iloc[0]
        pvp_val = clean_numeric_column(pd.Series([row.get('PVP', 0.0)])).fillna(0).iloc[0]
        dy_val = clean_numeric_column(pd.Series([row.get('Div_Yield', 0.0)])).fillna(0).iloc[0]
        roe_val = clean_numeric_column(pd.Series([row.get('ROE', 0.0)])).fillna(0).iloc[0]
        roic_val = clean_numeric_column(pd.Series([row.get('ROIC', 0.0)])).fillna(0).iloc[0]
        setor_name = str(row.get('Setor', '-'))
        
        pdf.cell(20, 6, clean_txt(ticker_clean), 1, 0, 'C')
        pdf.cell(50, 6, clean_txt(setor_name), 1, 0, 'L')
        pdf.cell(22, 6, clean_txt(f"{pl_val:.2f}"), 1, 0, 'R')
        pdf.cell(22, 6, clean_txt(f"{pvp_val:.2f}"), 1, 0, 'R')
        pdf.cell(22, 6, clean_txt(f"{dy_val*100:.2f}%" if dy_val < 1.0 else f"{dy_val:.2f}%"), 1, 0, 'R')
        pdf.cell(22, 6, clean_txt(f"{roe_val*100:.2f}%" if roe_val < 1.0 else f"{roe_val:.2f}%"), 1, 0, 'R')
        pdf.cell(22, 6, clean_txt(f"{roic_val*100:.2f}%" if roic_val < 1.0 else f"{roic_val:.2f}%"), 1, 1, 'R')
    pdf.ln(6)

# ─── SECÇÃO 4: SIMULAÇÃO MONTE CARLO (sempre incluída) ───
try:
    # Parâmetros da simulação (do session_state ou defaults)
    sim_run = st.session_state.get("simulation_run", False)
    if not sim_run:
        n_simulations = 200
        valor_inicial_sim = 10000.0
        years_sim = 3
    else:
        n_simulations = st.session_state["sim_n_simulations"]
        valor_inicial_sim = st.session_state["sim_valor_inicial"]
        years_sim = st.session_state["sim_years"]

    n_dias = years_sim * 252
    
    if modo == "Alocação Manual":
        pesos_dict = pesos_manuais
    else:
        pesos_dict = dict(zip(peso_manual_df.index + ".SA", peso_manual_df["Peso"].values))

    # Remove pesos nulos
    pesos_dict = {k: v for k, v in pesos_dict.items() if v > 1e-6}
    aligned_returns = returns.loc[:, pesos_dict.keys()].dropna()
    pesos_sim = np.array(list(pesos_dict.values()))

    mu_sim = aligned_returns.mean().values
    cov_sim = aligned_returns.cov().values

    np.random.seed(42)
    retornos_simulados = np.random.multivariate_normal(mu_sim, cov_sim, size=(n_dias, n_simulations))
    precos_simulados = np.exp(retornos_simulados.cumsum(axis=0))
    valor_portfolio = (precos_simulados * pesos_sim).sum(axis=2) * valor_inicial_sim

    datas_sim = pd.date_range(start=datetime.date.today(), periods=n_dias+1, freq='B')
    valor_portfolio = np.vstack([np.ones(n_simulations)*valor_inicial_sim, valor_portfolio])
    sim_df = pd.DataFrame(valor_portfolio, index=datas_sim)

    valores_finais = sim_df.iloc[-1]
    valor_esperado = valores_finais.mean()
    var_5 = np.percentile(valores_finais, 5)
    cvar_5 = valores_finais[valores_finais <= var_5].mean()
    pior_cenario = valores_finais.min()
    melhor_cenario = valores_finais.max()

    q1 = valores_finais.quantile(0.25)
    q2 = valores_finais.quantile(0.50)
    q3 = valores_finais.quantile(0.75)

    # Plot 1: Trajetórias
    fig_ind, ax_ind = plt.subplots(figsize=(8, 3.5))
    n_plot = min(20, n_simulations)
    for idx_sim in range(n_plot):
        ax_ind.plot(sim_df.index, sim_df.iloc[:, idx_sim], color='#0ea5e9', alpha=0.35, linewidth=0.8)
    ax_ind.set_title("Exemplos de Trajetorias Simuladas", fontsize=11, fontweight='bold', pad=10)
    ax_ind.set_ylabel("Valor do Portfolio (R$)")
    ax_ind.set_xlabel("Data")
    apply_pdf_chart_theme(fig_ind, ax_ind)
    fig_ind.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_ind:
        path_ind = f_ind.name
        fig_ind.savefig(path_ind, dpi=150)
        plt.close(fig_ind)

    # Plot 2: Fan Chart
    percentis = [5, 25, 50, 75, 95]
    fan_chart = sim_df.quantile(q=np.array(percentis) / 100, axis=1).T
    fan_chart.columns = [f"P{p}" for p in percentis]
    fig_fan, ax_fan = plt.subplots(figsize=(8, 3.5))
    ax_fan.fill_between(fan_chart.index, fan_chart["P5"], fan_chart["P95"], color='#bae6fd', alpha=0.4, label='Faixa 5%-95%')
    ax_fan.fill_between(fan_chart.index, fan_chart["P25"], fan_chart["P75"], color='#7dd3fc', alpha=0.5, label='Faixa 25%-75%')
    ax_fan.plot(fan_chart.index, fan_chart["P50"], color='#0284c7', linewidth=2, label='Mediana')
    ax_fan.set_title("Fan Chart com Faixas de Confianca", fontsize=11, fontweight='bold', pad=10)
    ax_fan.set_ylabel("Valor do Portfolio (R$)")
    ax_fan.set_xlabel("Data")
    ax_fan.legend(loc='upper left', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    apply_pdf_chart_theme(fig_fan, ax_fan)
    fig_fan.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_fan:
        path_fan = f_fan.name
        fig_fan.savefig(path_fan, dpi=150)
        plt.close(fig_fan)

    # Plot 3: Histograma
    fig_hist, ax_hist = plt.subplots(figsize=(8, 3.5))
    sns.histplot(valores_finais, bins=30, kde=True, color='#0284c7', edgecolor='#ffffff', alpha=0.6, ax=ax_hist)
    ax_hist.axvline(q1, color='#ef4444', linestyle='--', linewidth=1.2, label='Q1 (25%)')
    ax_hist.axvline(q2, color='#22c55e', linestyle='-', linewidth=1.5, label='Mediana (50%)')
    ax_hist.axvline(q3, color='#eab308', linestyle='--', linewidth=1.2, label='Q3 (75%)')
    ax_hist.set_title("Distribuicao dos Valores Finais", fontsize=11, fontweight='bold', pad=10)
    ax_hist.set_xlabel("Valor Final do Portfolio (R$)")
    ax_hist.set_ylabel("Frequencia")
    ax_hist.legend(loc='upper right', fontsize=8, frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
    apply_pdf_chart_theme(fig_hist, ax_hist)
    fig_hist.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_hist:
        path_hist = f_hist.name
        fig_hist.savefig(path_hist, dpi=150)
        plt.close(fig_hist)

    # Adiciona página e conteúdo ao PDF
    pdf.add_page()
    draw_section_header(pdf, '4. PROJECAO E RISCO ESTOCASTICO (MONTE CARLO)')
    
    pdf.set_font('Arial', '', 9)
    pdf.cell(0, 5, clean_txt(f"Capital Inicial: R$ {valor_inicial_sim:,.2f}  |  Horizonte: {years_sim} Anos  |  Trajetorias: {n_simulations}"), 0, 1)
    pdf.ln(3)
    
    pdf.set_font('Arial', 'B', 9)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(100, 7, clean_txt('Metrica de Retorno ou Risco Estocastico'), 1, 0, 'L', fill=True)
    pdf.cell(80, 7, clean_txt('Valor Projetado (R$)'), 1, 1, 'R', fill=True)
    
    pdf.set_font('Arial', '', 9)
    pdf.cell(100, 6, clean_txt('Valor Esperado Final (Media)'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {valor_esperado:,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('VaR 5% (Valor em Risco Extremo)'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {var_5:,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('CVaR 5% (Perda Media nos 5% Piores Casos)'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {cvar_5:,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('Pior Cenario Simulado'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {pior_cenario:,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('Melhor Cenario Simulado'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {melhor_cenario:,.2f}"), 1, 1, 'R')
    pdf.ln(6)

    # Tabela de estatísticas complementares
    pdf.set_font('Arial', 'B', 9)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(100, 7, clean_txt('Percentil / Estatistica Final'), 1, 0, 'L', fill=True)
    pdf.cell(80, 7, clean_txt('Valor Projetado (R$)'), 1, 1, 'R', fill=True)
    
    pdf.set_font('Arial', '', 9)
    pdf.cell(100, 6, clean_txt('Minimo'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {valores_finais.min():,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('Q1 (25%)'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {q1:,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('Mediana (50%)'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {q2:,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('Q3 (75%)'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {q3:,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('Maximo'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {valores_finais.max():,.2f}"), 1, 1, 'R')
    pdf.cell(100, 6, clean_txt('Desvio Padrao'), 1, 0, 'L')
    pdf.cell(80, 6, clean_txt(f"R$ {valores_finais.std():,.2f}"), 1, 1, 'R')
    pdf.ln(10)

    # Adiciona gráficos ao PDF
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, clean_txt('Graficos de Projecao Estocastica (Monte Carlo)'), 0, 1, 'L')
    pdf.ln(5)
    pdf.image(path_ind, x=15, y=pdf.get_y(), w=180)
    pdf.ln(70)
    pdf.image(path_fan, x=15, y=pdf.get_y(), w=180)

    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, clean_txt('Graficos de Projecao Estocastica (Monte Carlo) - Continuacao'), 0, 1, 'L')
    pdf.ln(5)
    pdf.image(path_hist, x=15, y=pdf.get_y(), w=180)
    pdf.ln(70)

    # Limpeza dos arquivos temporários
    for p in [path_ind, path_fan, path_hist]:
        try:
            os.remove(p)
        except Exception:
            pass
except Exception as sim_err:
    # Fallback silencioso ou log de erro se a simulação falhar por dados insuficientes
    pdf.add_page()
    draw_section_header(pdf, '4. PROJECAO E RISCO ESTOCASTICO (MONTE CARLO)')
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 10, clean_txt(f"Nao foi possivel gerar a projecao: {str(sim_err)}"), 0, 1)

# ─── SECÇÃO 5: DETALHAMENTO QUANTSTATS ───
pdf.add_page()
draw_section_header(pdf, 'DETALHAMENTO QUANTSTATS (ESTATISTICAS ADICIONAIS)')
pdf.set_font('Arial', '', 8)

metrics_df = qs.reports.metrics(portfolio_returns, benchmark=retorno_bench, display=False)
if isinstance(metrics_df, pd.DataFrame):
    pdf.set_font('Arial', 'B', 8)
    pdf.set_fill_color(226, 232, 240)
    pdf.cell(70, 6, clean_txt("Metrica"), 1, 0, 'L', fill=True)
    if "Benchmark" in metrics_df.columns:
        pdf.cell(55, 6, clean_txt("Benchmark"), 1, 0, 'C', fill=True)
    if "Strategy" in metrics_df.columns:
        pdf.cell(55, 6, clean_txt("Strategy (Carteira)"), 1, 1, 'C', fill=True)
    else:
        pdf.ln()
        
    pdf.set_font('Arial', '', 8)
    for idx, row in metrics_df.iterrows():
        idx_name = str(idx)
        pdf.cell(70, 5.5, clean_txt(idx_name), 1, 0, 'L')
        if "Benchmark" in metrics_df.columns:
            bench_val = str(row["Benchmark"])
            pdf.cell(55, 5.5, clean_txt(bench_val), 1, 0, 'C')
        if "Strategy" in metrics_df.columns:
            strat_val = str(row["Strategy"])
            pdf.cell(55, 5.5, clean_txt(strat_val), 1, 1, 'C')
        else:
            pdf.ln()

with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
    pdf_path = tmp_pdf.name
pdf.output(pdf_path)

with open(pdf_path, "rb") as f:
    pdf_data = f.read()

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="Baixar Relatório Completo (HTML)",
        data=html_data,
        file_name="relatorio_portfolio.html",
        mime="text/html"
    )
with col2:
    st.download_button(
        label="Baixar Resumo do Portfolio (PDF)",
        data=pdf_data,
        file_name="resumo_portfolio.pdf",
        mime="application/pdf"
    )
