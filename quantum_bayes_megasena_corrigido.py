"""
MEGA-SENA STATISTICAL LAB — versão revisada

Aplicativo Streamlit para análise estatística, backtesting e geração responsável
 de combinações da Mega-Sena.

IMPORTANTE:
- Cada combinação de seis dezenas possui a mesma probabilidade matemática.
- O histórico não garante vantagem preditiva em um sorteio independente.
- O objetivo deste aplicativo é avaliar modelos contra uma referência uniforme,
  evitando apresentar padrões históricos como certeza de previsão.
"""

from __future__ import annotations

import io
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy import stats
from sklearn.metrics import mutual_info_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

N_NUMEROS = 60
DEZENAS_POR_JOGO = 6
PROB_MARGINAL_UNIFORME = DEZENAS_POR_JOGO / N_NUMEROS
COLS_DEZENAS = [f"D{i}" for i in range(1, 7)]
CACHE_FILE = Path("megasena_cache.parquet")
CACHE_TTL_HORAS = 12

CAIXA_URL = (
    "https://servicebus2.caixa.gov.br/portaldeloterias/api/resultados/"
    "download?modalidade=Mega-Sena"
)
CAIXA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "Referer": "https://loterias.caixa.gov.br/",
}


# =============================================================================
# DADOS
# =============================================================================

def _validar_dezenas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza, valida e ordena as seis dezenas de cada concurso."""
    out = df.copy()

    for col in COLS_DEZENAS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=COLS_DEZENAS).copy()
    out[COLS_DEZENAS] = out[COLS_DEZENAS].astype(int)

    matriz = out[COLS_DEZENAS].to_numpy()
    faixa_valida = ((matriz >= 1) & (matriz <= N_NUMEROS)).all(axis=1)
    sem_repeticao = np.array([len(set(row)) == DEZENAS_POR_JOGO for row in matriz])
    out = out.loc[faixa_valida & sem_repeticao].copy()

    out[COLS_DEZENAS] = np.sort(out[COLS_DEZENAS].to_numpy(), axis=1)

    if "Concurso" in out.columns:
        out["Concurso"] = pd.to_numeric(out["Concurso"], errors="coerce")
        out = out.dropna(subset=["Concurso"])
        out["Concurso"] = out["Concurso"].astype(int)
        out = out.sort_values("Concurso")
    elif "Data" in out.columns:
        out = out.sort_values("Data")

    return out.drop_duplicates(subset=COLS_DEZENAS + (["Concurso"] if "Concurso" in out else []))\
              .reset_index(drop=True)


def _detectar_colunas_dezenas(df: pd.DataFrame) -> list[str]:
    encontradas: list[str] = []
    for col in df.columns:
        texto = str(col).strip().lower()
        if "dezena" in texto and any(
            marcador in texto
            for i in range(1, 7)
            for marcador in (f"{i}ª", f"{i}a", f"{i}º")
        ):
            encontradas.append(col)

    if len(encontradas) >= 6:
        return encontradas[:6]

    candidatas = []
    for col in df.columns:
        serie = pd.to_numeric(df[col], errors="coerce")
        validos = serie.dropna()
        if len(validos) and validos.between(1, 60).mean() >= 0.90:
            candidatas.append(col)

    return candidatas[:6]


def parse_caixa_excel(content: bytes) -> pd.DataFrame:
    bruto = pd.read_excel(io.BytesIO(content), header=0)
    bruto.columns = [str(c).strip() for c in bruto.columns]

    dezenas = _detectar_colunas_dezenas(bruto)
    if len(dezenas) != 6:
        raise ValueError("Não foi possível identificar as seis colunas de dezenas.")

    concurso_col = next(
        (c for c in bruto.columns if "concurso" in str(c).lower()),
        None,
    )
    data_col = next((c for c in bruto.columns if "data" in str(c).lower()), None)

    out = pd.DataFrame()
    out["Concurso"] = (
        pd.to_numeric(bruto[concurso_col], errors="coerce")
        if concurso_col
        else np.arange(1, len(bruto) + 1)
    )
    if data_col:
        out["Data"] = pd.to_datetime(bruto[data_col], dayfirst=True, errors="coerce")

    for destino, origem in zip(COLS_DEZENAS, dezenas):
        out[destino] = bruto[origem]

    return _validar_dezenas(out)


def parse_upload(content: bytes, filename: str) -> pd.DataFrame:
    nome = filename.lower()
    if nome.endswith((".xlsx", ".xls")):
        return parse_caixa_excel(content)

    erros: list[str] = []
    for sep in (";", ",", "\t"):
        try:
            bruto = pd.read_csv(io.BytesIO(content), sep=sep)
            dezenas = _detectar_colunas_dezenas(bruto)
            if len(dezenas) != 6:
                raise ValueError("seis colunas de dezenas não encontradas")

            out = pd.DataFrame({"Concurso": np.arange(1, len(bruto) + 1)})
            for destino, origem in zip(COLS_DEZENAS, dezenas):
                out[destino] = bruto[origem]
            validado = _validar_dezenas(out)
            if not validado.empty:
                return validado
        except Exception as exc:
            erros.append(str(exc))

    raise ValueError(
        "Não foi possível interpretar o arquivo. Use o Excel oficial da CAIXA "
        "ou CSV contendo seis colunas numéricas entre 1 e 60."
    )


@st.cache_data(ttl=CACHE_TTL_HORAS * 3600, show_spinner=False)
def baixar_dados_caixa() -> tuple[pd.DataFrame | None, str]:
    if CACHE_FILE.exists():
        idade = time.time() - CACHE_FILE.stat().st_mtime
        if idade < CACHE_TTL_HORAS * 3600:
            try:
                df = pd.read_parquet(CACHE_FILE)
                return _validar_dezenas(df), f"Cache local: {len(df)} concursos"
            except Exception:
                pass

    try:
        resposta = requests.get(CAIXA_URL, headers=CAIXA_HEADERS, timeout=30)
        resposta.raise_for_status()
        df = parse_caixa_excel(resposta.content)
        try:
            df.to_parquet(CACHE_FILE, index=False)
        except Exception:
            pass
        return df, f"Dados oficiais carregados: {len(df)} concursos"
    except Exception as exc:
        return None, f"Falha no download automático: {exc}"


@st.cache_data
def gerar_historico_sintetico(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    jogos = [np.sort(rng.choice(np.arange(1, 61), 6, replace=False)) for _ in range(n)]
    df = pd.DataFrame(jogos, columns=COLS_DEZENAS)
    df.insert(0, "Concurso", np.arange(1, n + 1))
    return df


def para_array(df: pd.DataFrame) -> np.ndarray:
    return df[COLS_DEZENAS].to_numpy(dtype=int)


def matriz_binaria(historico: np.ndarray) -> np.ndarray:
    x = np.zeros((len(historico), N_NUMEROS), dtype=np.int8)
    linhas = np.repeat(np.arange(len(historico)), DEZENAS_POR_JOGO)
    colunas = historico.reshape(-1) - 1
    x[linhas, colunas] = 1
    return x


# =============================================================================
# MODELOS
# Todos retornam distribuição de inclusão que soma 1. Para a probabilidade
# marginal de presença no concurso, multiplica-se por 6.
# =============================================================================

def normalizar(score: np.ndarray, piso: float = 1e-12) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    score = np.clip(score, piso, None)
    return score / score.sum()


def modelo_uniforme(_: np.ndarray) -> np.ndarray:
    return np.ones(N_NUMEROS) / N_NUMEROS


def modelo_bayes(historico: np.ndarray, alpha: float = 20.0) -> np.ndarray:
    contagens = matriz_binaria(historico).sum(axis=0)
    posterior = alpha / N_NUMEROS + contagens
    return normalizar(posterior)


def modelo_bayes_decay(
    historico: np.ndarray,
    alpha: float = 20.0,
    meia_vida: float = 250.0,
) -> np.ndarray:
    x = matriz_binaria(historico)
    idade = np.arange(len(x) - 1, -1, -1)
    pesos = 0.5 ** (idade / max(meia_vida, 1.0))
    contagens = (x * pesos[:, None]).sum(axis=0)
    posterior = alpha / N_NUMEROS + contagens
    return normalizar(posterior)


def modelo_tendencia_regularizada(
    historico: np.ndarray,
    janela_curta: int = 100,
    janela_longa: int = 600,
    shrinkage: float = 0.25,
) -> np.ndarray:
    x = matriz_binaria(historico)
    curta = x[-min(janela_curta, len(x)):].mean(axis=0)
    longa = x[-min(janela_longa, len(x)):].mean(axis=0)
    tendencia = longa + shrinkage * (curta - longa)
    uniforme = np.full(N_NUMEROS, PROB_MARGINAL_UNIFORME)
    regularizado = 0.65 * tendencia + 0.35 * uniforme
    return normalizar(regularizado)


def modelo_intervalos_regularizado(historico: np.ndarray) -> np.ndarray:
    """Usa estabilidade dos intervalos, sem assumir que números atrasados estão 'devendo'."""
    x = matriz_binaria(historico)
    score = np.zeros(N_NUMEROS)

    for i in range(N_NUMEROS):
        ocorrencias = np.flatnonzero(x[:, i])
        if len(ocorrencias) < 8:
            score[i] = 1.0
            continue
        intervalos = np.diff(ocorrencias)
        cv = intervalos.std(ddof=1) / max(intervalos.mean(), 1e-9)
        estabilidade = 1.0 / (1.0 + cv)
        taxa = len(ocorrencias) / len(x)
        score[i] = 0.75 * taxa + 0.25 * PROB_MARGINAL_UNIFORME * estabilidade

    return normalizar(score)


def modelo_pares_regularizado(historico: np.ndarray, ultimos: int = 20) -> np.ndarray:
    """Rede de coocorrência regularizada, condicionada às dezenas recentes."""
    x = matriz_binaria(historico)
    n = len(x)
    cooc = x.T @ x
    np.fill_diagonal(cooc, 0)

    freq = x.sum(axis=0)
    esperado = np.outer(freq, freq) * (DEZENAS_POR_JOGO - 1) / max(n * DEZENAS_POR_JOGO, 1)
    excesso = (cooc - esperado) / np.sqrt(esperado + 5.0)
    excesso = np.clip(excesso, -3.0, 3.0)

    contexto = x[-min(ultimos, n):].mean(axis=0)
    score_rede = excesso @ contexto
    base = modelo_bayes_decay(historico)
    score_rede = (score_rede - score_rede.mean()) / (score_rede.std() + 1e-9)
    return normalizar(base * np.exp(0.08 * score_rede))


def modelo_quantum_inspirado(historico: np.ndarray) -> np.ndarray:
    """
    Modelo de amplitude inspirado em probabilidade quântica.

    Não representa um sistema físico quântico. Usa amplitudes complexas para combinar
    frequência regularizada e coocorrências, preservando positividade após |psi|².
    """
    base = modelo_bayes_decay(historico, alpha=30.0, meia_vida=350.0)
    x = matriz_binaria(historico)
    cooc = x.T @ x
    np.fill_diagonal(cooc, 0)
    cooc = cooc / (cooc.max() + 1e-9)

    fase = np.angle(np.exp(2j * np.pi * cooc.mean(axis=1)))
    amplitude = np.sqrt(base) * np.exp(1j * fase)

    kernel = cooc / (cooc.sum(axis=1, keepdims=True) + 1e-9)
    amplitude_interferida = 0.90 * amplitude + 0.10 * (kernel @ amplitude)
    prob = np.abs(amplitude_interferida) ** 2
    return normalizar(prob)


def modelo_popularidade_ev(historico: np.ndarray) -> np.ndarray:
    """Heurística de valor esperado condicional: reduz números tipicamente populares."""
    base = modelo_bayes(historico, alpha=50.0)
    numeros = np.arange(1, 61)

    popularidade = np.ones(60)
    popularidade[numeros <= 31] *= 1.65
    popularidade[numeros % 5 == 0] *= 1.12
    popularidade[numeros % 10 == 0] *= 1.10
    popularidade[[6, 12, 18, 24, 30, 36, 42, 48, 54, 60]] *= 1.05

    return normalizar(base / popularidade)


MODEL_FACTORIES: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "Uniforme": modelo_uniforme,
    "Bayes histórico": modelo_bayes,
    "Bayes com decaimento": modelo_bayes_decay,
    "Tendência regularizada": modelo_tendencia_regularizada,
    "Intervalos regularizados": modelo_intervalos_regularizado,
    "Rede de pares": modelo_pares_regularizado,
    "Quantum-inspirado": modelo_quantum_inspirado,
}


# =============================================================================
# AVALIAÇÃO E ENSEMBLE
# =============================================================================

@dataclass
class ResultadoBacktest:
    tabela: pd.DataFrame
    pesos: dict[str, float]
    previsoes_ensemble: list[np.ndarray]
    resultados_reais: list[np.ndarray]


def log_loss_multilabel(prob_dist: np.ndarray, realizado: np.ndarray) -> float:
    """Log loss binária média usando probabilidades marginais calibradas."""
    p = np.clip(prob_dist * DEZENAS_POR_JOGO, 1e-6, 1 - 1e-6)
    y = np.zeros(N_NUMEROS)
    y[realizado - 1] = 1
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_multilabel(prob_dist: np.ndarray, realizado: np.ndarray) -> float:
    p = np.clip(prob_dist * DEZENAS_POR_JOGO, 0, 1)
    y = np.zeros(N_NUMEROS)
    y[realizado - 1] = 1
    return float(np.mean((p - y) ** 2))


def acertos_top6(prob_dist: np.ndarray, realizado: np.ndarray) -> int:
    top = set(np.argsort(prob_dist)[-6:] + 1)
    return len(top.intersection(set(realizado)))


@st.cache_data(show_spinner=False)
def executar_backtest(
    historico_tuple: tuple[tuple[int, ...], ...],
    janela_treino: int,
    n_testes: int,
    passo: int,
) -> ResultadoBacktest:
    historico = np.asarray(historico_tuple, dtype=int)
    inicio = max(janela_treino, len(historico) - n_testes * passo)
    indices = list(range(inicio, len(historico), passo))

    acumulado = {
        nome: {"logloss": [], "brier": [], "top6": []}
        for nome in MODEL_FACTORIES
    }

    previsoes_por_data: list[dict[str, np.ndarray]] = []
    resultados: list[np.ndarray] = []

    for idx in indices:
        treino = historico[max(0, idx - janela_treino):idx]
        real = historico[idx]
        preds: dict[str, np.ndarray] = {}

        for nome, func in MODEL_FACTORIES.items():
            try:
                p = normalizar(func(treino))
            except Exception:
                p = modelo_uniforme(treino)
            preds[nome] = p
            acumulado[nome]["logloss"].append(log_loss_multilabel(p, real))
            acumulado[nome]["brier"].append(brier_multilabel(p, real))
            acumulado[nome]["top6"].append(acertos_top6(p, real))

        previsoes_por_data.append(preds)
        resultados.append(real)

    linhas = []
    for nome, metricas in acumulado.items():
        linhas.append({
            "Modelo": nome,
            "Log loss": np.mean(metricas["logloss"]),
            "Brier": np.mean(metricas["brier"]),
            "Acertos médios Top 6": np.mean(metricas["top6"]),
            "Testes": len(indices),
        })

    tabela = pd.DataFrame(linhas).sort_values("Log loss").reset_index(drop=True)
    baseline = float(tabela.loc[tabela["Modelo"] == "Uniforme", "Log loss"].iloc[0])

    # Peso baseado somente em desempenho fora da amostra. Modelos piores que o
    # uniforme recebem peso muito pequeno, evitando falsa confiança.
    vantagens = {}
    for _, row in tabela.iterrows():
        ganho = baseline - float(row["Log loss"])
        vantagens[row["Modelo"]] = ganho

    valores = np.array([vantagens[n] for n in MODEL_FACTORIES])
    escala = max(np.std(valores), 1e-4)
    logits = np.clip(valores / escala, -8, 8)
    exp = np.exp(logits - logits.max())
    pesos_arr = exp / exp.sum()
    pesos = dict(zip(MODEL_FACTORIES.keys(), pesos_arr))

    previsoes_ensemble = []
    for preds in previsoes_por_data:
        p = sum(pesos[n] * preds[n] for n in MODEL_FACTORIES)
        previsoes_ensemble.append(normalizar(p))

    tabela["Peso automático"] = tabela["Modelo"].map(pesos)
    tabela["Ganho vs uniforme"] = baseline - tabela["Log loss"]

    return ResultadoBacktest(tabela, pesos, previsoes_ensemble, resultados)


def ensemble_atual(historico: np.ndarray, pesos: dict[str, float]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    previsoes = {nome: normalizar(func(historico)) for nome, func in MODEL_FACTORIES.items()}
    final = sum(pesos.get(nome, 0.0) * p for nome, p in previsoes.items())
    return normalizar(final), previsoes


# =============================================================================
# GERAÇÃO DE JOGOS E TESTES ESTATÍSTICOS
# =============================================================================

def gerar_jogos_diversificados(
    prob: np.ndarray,
    quantidade: int,
    seed: int,
    diversidade: float = 0.30,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    jogos: list[list[int]] = []
    penalidade = np.ones(N_NUMEROS)

    tentativas = 0
    while len(jogos) < quantidade and tentativas < quantidade * 200:
        tentativas += 1
        p = normalizar(prob * penalidade)
        jogo = sorted(rng.choice(np.arange(1, 61), 6, replace=False, p=p).tolist())

        # Restrições apenas de diversificação; não aumentam a chance matemática.
        pares = sum(n % 2 == 0 for n in jogo)
        soma = sum(jogo)
        consecutivos = sum(b == a + 1 for a, b in zip(jogo, jogo[1:]))

        if not (2 <= pares <= 4 and 100 <= soma <= 260 and consecutivos <= 2):
            continue
        if any(len(set(jogo) & set(outro)) > 4 for outro in jogos):
            continue

        jogos.append(jogo)
        penalidade[np.array(jogo) - 1] *= max(0.45, 1.0 - diversidade)

    while len(jogos) < quantidade:
        jogo = sorted(rng.choice(np.arange(1, 61), 6, replace=False, p=normalizar(prob)).tolist())
        if jogo not in jogos:
            jogos.append(jogo)

    return jogos


def monte_carlo_inclusao(prob: np.ndarray, simulacoes: int, seed: int) -> np.ndarray:
    """Implementação correta: cada jogo é amostrado separadamente sem reposição."""
    rng = np.random.default_rng(seed)
    contagem = np.zeros(N_NUMEROS, dtype=np.int64)
    numeros = np.arange(1, 61)
    p = normalizar(prob)

    for _ in range(simulacoes):
        jogo = rng.choice(numeros, 6, replace=False, p=p)
        contagem[jogo - 1] += 1

    return contagem / contagem.sum()


def testes_uniformidade(historico: np.ndarray) -> dict[str, float]:
    freq = matriz_binaria(historico).sum(axis=0)
    esperado = np.full(N_NUMEROS, freq.sum() / N_NUMEROS)
    chi2, p_chi2 = stats.chisquare(freq, esperado)
    return {"chi2": float(chi2), "p_chi2": float(p_chi2)}


def matriz_informacao_mutua(historico: np.ndarray, limite: int = 30) -> np.ndarray:
    x = matriz_binaria(historico)[:, :limite]
    mi = np.zeros((limite, limite))
    for i in range(limite):
        for j in range(i + 1, limite):
            valor = mutual_info_score(x[:, i], x[:, j]) / math.log(2)
            mi[i, j] = mi[j, i] = valor
    return mi


# =============================================================================
# INTERFACE
# =============================================================================

st.set_page_config(
    page_title="Mega-Sena Statistical Lab",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.number-ball {display:inline-block;background:linear-gradient(135deg,#667eea,#764ba2);
color:white;font-weight:700;font-size:18px;width:48px;height:48px;line-height:48px;
text-align:center;border-radius:50%;margin:4px;box-shadow:0 4px 14px rgba(102,126,234,.4)}
.game-card {padding:14px;border:1px solid #394263;border-radius:12px;margin:8px 0;
background:linear-gradient(135deg,#171c2d,#222a46)}
.warning-box {padding:12px;border:1px solid #f59e0b;border-radius:8px;background:#2d1b00;color:#fcd34d}
.good-box {padding:12px;border:1px solid #22c55e;border-radius:8px;background:#082f1c;color:#bbf7d0}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📂 Fonte de dados")
    fonte = st.radio(
        "Origem",
        ["Auto-download CAIXA", "Upload manual", "Dados sintéticos"],
    )

    df: pd.DataFrame | None = None
    if fonte == "Auto-download CAIXA":
        with st.spinner("Carregando dados..."):
            df, status = baixar_dados_caixa()
        st.caption(status)
        if st.button("Forçar novo download", use_container_width=True):
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
            baixar_dados_caixa.clear()
            st.rerun()

    elif fonte == "Upload manual":
        arquivo = st.file_uploader("Excel ou CSV", type=["xlsx", "xls", "csv"])
        if arquivo:
            try:
                df = parse_upload(arquivo.read(), arquivo.name)
                st.success(f"{len(df)} concursos válidos")
            except Exception as exc:
                st.error(str(exc))

    else:
        n_sint = st.slider("Concursos sintéticos", 700, 5000, 2800, 100)
        seed_sint = st.number_input("Semente dos dados", 0, 999999, 42)
        df = gerar_historico_sintetico(n_sint, int(seed_sint))
        st.caption("Dados uniformes simulados")

    st.markdown("---")
    st.header("🧪 Backtesting")
    janela_treino = st.slider("Janela de treinamento", 300, 2000, 1000, 100)
    n_testes = st.slider("Concursos de teste", 30, 300, 120, 10)
    passo = st.slider("Passo entre testes", 1, 5, 1)

    st.markdown("---")
    st.header("🎟️ Geração")
    n_jogos = st.slider("Quantidade de jogos", 1, 30, 8)
    diversidade = st.slider("Diversidade entre jogos", 0.0, 0.7, 0.30, 0.05)
    seed = st.number_input("Semente de geração", 0, 999999, 2026)
    simulacoes_mc = st.select_slider(
        "Simulações Monte Carlo",
        options=[10_000, 25_000, 50_000, 100_000],
        value=25_000,
    )

    executar = st.button("🚀 Executar análise", type="primary", use_container_width=True)

st.title("🎲 Mega-Sena Statistical Lab")
st.caption("Modelagem probabilística, comparação fora da amostra e geração diversificada")
st.markdown(
    """
<div class="warning-box"><b>Limite científico:</b> a Mega-Sena é tratada como sorteio independente.
Este aplicativo não promete prever a combinação vencedora. Um modelo só recebe relevância quando
supera a distribuição uniforme em concursos que não foram usados no treinamento.</div>
""",
    unsafe_allow_html=True,
)

if df is None or len(df) < max(350, janela_treino + 10):
    st.info("Carregue dados suficientes para executar o backtesting.")
    st.stop()

historico = para_array(df)

if executar or "analise" not in st.session_state:
    with st.spinner("Executando modelos e backtesting..."):
        historico_tuple = tuple(tuple(int(v) for v in row) for row in historico)
        bt = executar_backtest(historico_tuple, janela_treino, n_testes, passo)
        p_final, previsoes = ensemble_atual(historico[-janela_treino:], bt.pesos)
        p_mc = monte_carlo_inclusao(p_final, simulacoes_mc, int(seed))
        jogos = gerar_jogos_diversificados(p_final, n_jogos, int(seed), diversidade)
        testes = testes_uniformidade(historico)

        st.session_state.analise = {
            "bt": bt,
            "p_final": p_final,
            "previsoes": previsoes,
            "p_mc": p_mc,
            "jogos": jogos,
            "testes": testes,
            "df": df,
            "historico": historico,
        }

R = st.session_state.analise
bt: ResultadoBacktest = R["bt"]
p_final = R["p_final"]
nums = np.arange(1, 61)
freq = matriz_binaria(R["historico"]).sum(axis=0)

abas = st.tabs([
    "🏠 Visão geral",
    "🧪 Backtesting",
    "📊 Modelos",
    "🔗 Dependências",
    "🎰 Monte Carlo",
    "🎯 Jogos",
])

with abas[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Concursos", f"{len(R['historico']):,}")
    c2.metric("Mais frequente", f"{freq.argmax()+1} ({freq.max()})")
    c3.metric("Menos frequente", f"{freq.argmin()+1} ({freq.min()})")
    c4.metric("p-valor uniformidade", f"{R['testes']['p_chi2']:.4f}")

    if R["testes"]["p_chi2"] < 0.05:
        st.warning(
            "O teste qui-quadrado encontrou desvio estatístico no histórico. Isso não prova "
            "capacidade de previsão e pode ocorrer por flutuação ou múltiplas comparações."
        )
    else:
        st.success("O histórico analisado não apresenta evidência forte contra frequência uniforme.")

    fig = go.Figure(go.Bar(
        x=nums,
        y=freq,
        marker=dict(color=p_final, colorscale="Viridis", showscale=True),
    ))
    fig.add_hline(y=freq.mean(), line_dash="dash", annotation_text=f"Média {freq.mean():.1f}")
    fig.update_layout(
        title="Frequência histórica — cor representa o ensemble atual",
        xaxis_title="Dezena",
        yaxis_title="Ocorrências",
        template="plotly_dark",
        height=410,
    )
    st.plotly_chart(fig, use_container_width=True)

    ultimos_cols = [c for c in ["Concurso", "Data", *COLS_DEZENAS] if c in R["df"].columns]
    with st.expander("Últimos concursos"):
        st.dataframe(R["df"][ultimos_cols].tail(15).sort_values("Concurso", ascending=False), use_container_width=True)

with abas[1]:
    st.header("Desempenho fora da amostra")
    st.caption(
        "Log loss e Brier menores são melhores. Ganho positivo indica resultado melhor que o "
        "modelo uniforme na janela avaliada."
    )

    tabela = bt.tabela.copy()
    tabela["Log loss"] = tabela["Log loss"].map(lambda x: f"{x:.6f}")
    tabela["Brier"] = tabela["Brier"].map(lambda x: f"{x:.6f}")
    tabela["Acertos médios Top 6"] = tabela["Acertos médios Top 6"].map(lambda x: f"{x:.3f}")
    tabela["Peso automático"] = tabela["Peso automático"].map(lambda x: f"{x:.2%}")
    tabela["Ganho vs uniforme"] = tabela["Ganho vs uniforme"].map(lambda x: f"{x:+.6f}")
    st.dataframe(tabela, use_container_width=True, hide_index=True)

    ganhos = bt.tabela.set_index("Modelo")["Ganho vs uniforme"]
    fig = go.Figure(go.Bar(x=ganhos.index, y=ganhos.values))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="Ganho de log loss em relação ao uniforme",
        xaxis_title="Modelo",
        yaxis_title="Ganho",
        template="plotly_dark",
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    melhor_ganho = float(bt.tabela["Ganho vs uniforme"].max())
    if melhor_ganho <= 0:
        st.warning(
            "Nenhum modelo superou a referência uniforme neste backtesting. O ensemble fica "
            "próximo do uniforme, que é a conclusão estatisticamente mais responsável."
        )
    else:
        st.markdown(
            "<div class='good-box'><b>Resultado:</b> ao menos um modelo apresentou ganho fora "
            "da amostra. Esse ganho deve ser revalidado em outras janelas antes de ser tratado "
            "como evidência estável.</div>",
            unsafe_allow_html=True,
        )

with abas[2]:
    st.header("Distribuições dos modelos")
    fig = go.Figure()
    for nome, p in R["previsoes"].items():
        fig.add_trace(go.Scatter(x=nums, y=p * 60, mode="lines", name=nome, opacity=0.65))
    fig.add_trace(go.Scatter(
        x=nums,
        y=p_final * 60,
        mode="lines",
        name="ENSEMBLE",
        line=dict(width=4),
    ))
    fig.update_layout(
        title="Score relativo por dezena — 1 representa distribuição uniforme",
        xaxis_title="Dezena",
        yaxis_title="Score × 60",
        template="plotly_dark",
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    ranking = pd.DataFrame({
        "Ranking": np.arange(1, 61),
        "Dezena": np.argsort(p_final)[::-1] + 1,
        "Score ensemble": np.sort(p_final)[::-1],
    })
    ranking["Probabilidade marginal calibrada"] = (
        ranking["Score ensemble"] * DEZENAS_POR_JOGO
    ).clip(upper=1)
    st.dataframe(ranking.head(20), use_container_width=True, hide_index=True)

with abas[3]:
    st.header("Informação mútua entre dezenas")
    st.caption(
        "A informação mútua é calculada com os quatro estados binários completos: "
        "ambas, somente uma, somente a outra e nenhuma."
    )
    mi = matriz_informacao_mutua(R["historico"], limite=30)
    fig = go.Figure(go.Heatmap(
        z=mi,
        x=np.arange(1, 31),
        y=np.arange(1, 31),
        colorscale="Magma",
        colorbar=dict(title="bits"),
    ))
    fig.update_layout(template="plotly_dark", height=570)
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Associações históricas pequenas são esperadas por acaso. Uma matriz visual não deve "
        "ser interpretada isoladamente como relação causal ou vantagem futura."
    )

with abas[4]:
    st.header("Validação Monte Carlo")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=nums, y=R["p_mc"] * 60, name="Monte Carlo"))
    fig.add_trace(go.Scatter(x=nums, y=p_final * 60, mode="lines", name="Alvo ensemble"))
    fig.update_layout(
        title=f"Frequência simulada em {simulacoes_mc:,} jogos",
        xaxis_title="Dezena",
        yaxis_title="Score × 60",
        template="plotly_dark",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    erro = float(np.abs(R["p_mc"] - p_final).mean())
    st.metric("Erro absoluto médio da simulação", f"{erro:.8f}")
    st.caption(
        "O Monte Carlo aqui valida a implementação da distribuição. Ele não cria informação "
        "nova nem aumenta a capacidade de previsão."
    )

with abas[5]:
    st.header("Combinações diversificadas")
    st.caption(
        "As restrições reduzem repetição entre os jogos e evitam concentrações extremas. "
        "Elas não alteram a probabilidade de uma combinação específica ser sorteada."
    )

    linhas_csv = []
    for i, jogo in enumerate(R["jogos"], start=1):
        bolas = "".join(f'<span class="number-ball">{n:02d}</span>' for n in jogo)
        score = sum(p_final[n - 1] for n in jogo)
        st.markdown(
            f'<div class="game-card"><b>Jogo {i}</b> &nbsp; '
            f'<small>score interno: {score:.6f}</small><br>{bolas}</div>',
            unsafe_allow_html=True,
        )
        linhas_csv.append({"Jogo": i, **{f"N{j+1}": n for j, n in enumerate(jogo)}})

    csv = pd.DataFrame(linhas_csv).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Exportar jogos em CSV",
        data=csv,
        file_name="jogos_megasena_statistical_lab.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown(
        """
<div class="warning-box"><b>Probabilidade oficial de uma aposta simples acertar as seis dezenas:</b>
1 em 50.063.860. Nenhuma pontuação mostrada neste aplicativo substitui essa probabilidade combinatória.</div>
""",
        unsafe_allow_html=True,
    )
