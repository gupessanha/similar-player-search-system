# Similaridade entre Jogadores

Estudo metodológico sobre a recuperação de similaridade entre jogadores de futebol a partir de diferentes combinações de **conjuntos de features** e **métricas de distância**.

Silva & Marinho (ECI) — Maio de 2026

---

## Motivação

Identificar jogadores similares é problema central no recrutamento esportivo:

- Encontrar substitutos viáveis no mercado de transferências.
- Comparar perfis táticos entre ligas e contextos.
- Apoiar decisões de scouting com critérios objetivos.

**Desafio metodológico:** a similaridade entre jogadores não possui um _ground truth_ objetivo. Métodos diferentes (distâncias, conjuntos de features) produzem recomendações diferentes, sem critério claro para arbitrar entre elas.

**Limitação adicional:** dados individualizados de jogadores reais (eventos espaço-temporais, métricas avançadas por jogador) são de difícil acesso público, restritos a temporadas e ligas específicas.

---

## Pergunta de Pesquisa

**Pergunta principal:**

> Em que medida diferentes combinações de conjunto de features e métrica de distância recuperam a similaridade real entre jogadores quando essa similaridade é conhecida por construção?

**Sub-perguntas:**

1. Qual combinação melhor agrupa jogadores por função tática (avaliação categórica)?
2. Qual combinação melhor recupera a similaridade total a partir de informação parcial (avaliação contínua)?
3. As três distâncias (Euclidiana, Cosseno, Pearson) convergem ou divergem nas recomendações?

---

## Estrutura do repositório

```
.
├── data/
│   ├── get_data.py         # baixa o dataset do Kaggle e copia para esta pasta
│   └── merged_players.csv  # gerado pelo script (não versionado)
├── notebooks/
│   └── 1_EDA.ipynb         # análise exploratória inicial
├── pyproject.toml
└── uv.lock
```

---

## Como executar

Requer Python 3.14 e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python data/get_data.py
```

O script baixa o _Football Manager 2023 Dataset_ via `kagglehub` e copia os arquivos para `data/`.
