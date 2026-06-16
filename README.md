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

**Limitação adicional:** dados individualizados de jogadores reais (eventos espaço-temporais, métricas avançadas por jogador) são de difícil acesso público, restritos a temporadas e ligas específicas. Por isso o estudo usa o _Football Manager 2023 Dataset_, em que cada jogador é descrito por atributos numéricos numa escala 1–20 — permitindo construir uma referência de similaridade **conhecida por construção**.

---

## Pergunta de Pesquisa

**Pergunta principal:**

> Em que medida diferentes combinações de conjunto de features e métrica de distância recuperam a similaridade real entre jogadores quando essa similaridade é conhecida por construção?

**Sub-perguntas:**

1. Qual combinação melhor agrupa jogadores por função tática (avaliação categórica)?
2. Qual combinação melhor recupera a similaridade total a partir de informação parcial (avaliação contínua)?
3. As três distâncias (Euclidiana, Cosseno, Pearson) convergem ou divergem nas recomendações?

---

## Dados

- **Fonte:** _Football Manager 2023 Dataset_ (Kaggle), baixado via `kagglehub`.
- **Bruto:** 91.672 linhas × 88 colunas. Cada jogador tem 60 atributos numéricos na **escala 1–20** (Técnico, Mental, Físico, Goleiro, Hidden/Personalidade), além de metadados (nome, clube, posição, etc.).
- **Chave única:** `UID`. As duplicatas (~4,9%) são cópias exatas de um merge e foram removidas por `UID`. `Name` **não** é único (ex.: 34 jogadores distintos chamados "Paulinho") e não serve como chave.
- **Após higiene:** 87.163 jogadores únicos. Nenhum atributo 1–20 tem valores ausentes.

---

## Metodologia

O pré-processamento está nos notebooks 1 (EDA) e 2 (features). Decisões centrais:

**Segregação de goleiros.** Os atributos de goleiro (`Aer`, `Cmd`, `Han`, `Kic`, `1v1`, `Ref`, …) formam um regime bimodal (média ≈ 2 em jogadores de linha vs ≈ 12 em goleiros) que dominaria qualquer distância no espaço misto. Goleiros (10,2% da base) são separados; o experimento principal trabalha sobre os **78.283 jogadores de linha**.

**Conjuntos de features.** 36 atributos de campo divididos em quatro conjuntos disjuntos:

| Conjunto | Nome       | Atributos                                                 |
| -------- | ---------- | --------------------------------------------------------- |
| `T(10)`  | Técnico    | `Cro Dri Fin Fir Hea Lon Mar Pas Tck Tec`                 |
| `M(14)`  | Mental     | `Agg Ant Bra Cmp Cnt Dec Det Fla Ldr OtB Pos Tea Vis Wor` |
| `F(8)`   | Físico     | `Acc Agi Bal Jum Nat.1 Pac Sta Str`                       |
| `SP(4)`  | Set Pieces | `Cor Fre Pen L Th`                                        |

`SP(4)` é tratado à parte porque bola parada reflete **papel atribuído pelo time** (quem é o batedor designado), não estilo de jogo intrínseco. Os 13 atributos _hidden/personalidade_ ficam fora dos conjuntos base — medem traços de personalidade, não desempenho observável em campo.

**Normalização: z-score por posição.** Cada atributo é padronizado para média 0 e desvio 1 **dentro de cada `primary_role`**:

$$z_{ij}^{(r)} = \frac{x_{ij} - \mu_j^{(r)}}{\sigma_j^{(r)}}$$

Isso remove o sinal posicional trivial (atacante tem `Fin` alto, zagueiro tem `Tck` alto) e faz a similaridade medir **estilo dentro da função tática** — Kroos vs Casemiro entre os meias, Van Dijk vs Rüdiger entre os zagueiros. É também o padrão de ferramentas de scouting (FBref, Comparisonator). Escolheu-se z-score sobre min-max e robust scaler porque preserva a forma da distribuição e dá peso uniforme às dimensões; os "outliers" do FM (jogadores de elite) são justamente o sinal de interesse. `μ` e `σ` são estimados apenas na população de jogadores de linha.

---

## Desenho experimental

O experimento principal cruza os conjuntos de features com as métricas de distância:

> **9 condições = 3 conjuntos `{T, M, F}` × 3 distâncias `{Euclidiana, Cosseno, Pearson}`**

As três distâncias capturam axiomas distintos de proximidade:

| Distância  | Similaridade significa…                       | Invariância         |
| ---------- | --------------------------------------------- | ------------------- |
| Euclidiana | valores absolutos próximos após padronização  | nenhuma             |
| Cosseno    | mesma direção do perfil, qualquer intensidade | escala              |
| Pearson    | mesma ordem relativa entre atributos          | escala + translação |

Cada combinação é avaliada por dois critérios complementares:

- **Avaliação A — categórica (defensiva).** Para cada jogador-consulta, mede a fração dos `K` vizinhos mais próximos com o mesmo `primary_role` (`recall@K`). Como a normalização por posição remove o sinal posicional por construção, os valores absolutos são propositalmente baixos — a comparação relevante é **entre as 9 condições**, não contra um baseline absoluto.
- **Avaliação B — contínua (construtiva).** O ranking calculado com **todos os 36 atributos via Euclidiana** é tratado como _ranking-verdade_ (perfil completo). Cada condição usa apenas um subconjunto parcial (`T`, `M` ou `F`) e é avaliada por quão bem reconstrói esse ranking (correlação de Spearman). Ground truth e condições vivem na **mesma normalização** (z-score por posição).

A combinação ideal é **alta em A e em B**: captura tanto a função tática quanto o estilo individual intra-grupo. A sub-pergunta 3 é respondida medindo a sobreposição entre as recomendações das três distâncias.

**Análise complementar.** `SP(4)` não entra nas 9 condições; é usado depois para um teste de robustez — o método vencedor identifica corretamente especialistas em bola parada?

---

## Estrutura do repositório

```
.
├── data/
│   ├── get_data.py          # baixa o dataset do Kaggle e copia para data/
│   ├── merged_players.csv   # bruto, gerado pelo script (não versionado)
│   └── processed/           # artefatos gerados pelo notebook 2 / pipeline (não versionados)
│       ├── outfield_z.csv     # 78.283 × 36 — z-score por posição (entrada da modelagem)
│       ├── outfield_raw.csv   # mesmos jogadores na escala 1–20
│       ├── gk_raw.csv         # goleiros preservados (estudo separado)
│       ├── feature_sets.json  # definição dos conjuntos T/M/F/SP
│       ├── zscore_stats.csv   # μ, σ por (role × atributo) — 216 linhas, formato long
│       └── neighbors.csv      # top-10 vizinhos de cada jogador por aspecto × métrica (gerado na Seção 6)
├── notebooks/
│   ├── 1_EDA.ipynb          # análise exploratória: higiene, famílias, perfis por posição
│   ├── 2_features.ipynb     # filtragem de GK, conjuntos de features, z-score por posição
│   └── 3_modelagem.ipynb    # 9 condições, Avaliações A/B, convergência, SP; Seção 6 (seleção nomeada)
├── src/spss/                # pacote reproduzível (instalado por `uv sync`)
│   ├── preprocessing.py     # higiene: drop/dedup por UID, primary_role, split GK
│   ├── features.py          # conjuntos T/M/F/SP e z-score por posição (ddof=0)
│   ├── distances.py         # primitivas Euclidiana/Cosseno/Pearson (fonte canônica)
│   ├── retrieval.py         # Retriever: similar(), show_grid(), export_neighbors()
│   └── pipeline.py          # CSV bruto -> os 5 artefatos (python -m spss.pipeline)
├── tests/                   # pytest: roles, features, distâncias, recuperação
├── pyproject.toml
└── uv.lock
```

Os artefatos em `data/processed/` são derivados reproduzíveis a partir do CSV bruto + notebooks, e não são versionados (cobertos pelo `.gitignore`).

---

## Estado atual

- [x] **Notebook 1 — EDA.** Higiene dos dados, categorização das 86 colunas, capacidade discriminante das posições.
- [x] **Notebook 2 — Features.** Segregação de goleiros, definição dos conjuntos T/M/F/SP, z-score por posição, persistência dos artefatos.
- [x] **Notebook 3 — Modelagem.** Implementação das 3 distâncias e das Avaliações A e B sobre as 9 condições; análise complementar de `SP`; **Seção 6** — seleção de jogadores similares **nomeada** por aspecto × métrica (`Retriever`) e export completo (`neighbors.csv`).
- [x] **Pacote `src/spss`.** Pipeline de pré-processamento reproduzível (`python -m spss.pipeline`) e API de recuperação (`spss.retrieval.Retriever`).
- [x] **`tests/`.** Testes de roles, features (z-score), distâncias e recuperação (`uv run pytest`).

---

## Como executar

Requer Python 3.14 e [uv](https://docs.astral.sh/uv/).

```bash
# 1. instalar dependências + o pacote `spss` (editável)
uv sync

# 2. baixar o dataset (gera data/merged_players (1).csv)
uv run python data/get_data.py

# 3a. regenerar os artefatos de data/processed/ (equivale aos notebooks 1+2)
uv run python -m spss.pipeline

# 3b. ou abrir os notebooks (ordem: 1 -> 2 -> 3)
uv run jupyter lab notebooks/

# 4. rodar os testes
uv run pytest
```

O script baixa o _Football Manager 2023 Dataset_ via `kagglehub` e copia os arquivos para `data/`. Como o `pyproject.toml` declara um `build-system`, `uv sync` também instala o pacote `spss` no ambiente (o mesmo venv usado pelo kernel Jupyter, então `import spss` funciona nos notebooks). Rodar `python -m spss.pipeline` (ou o notebook 2 de ponta a ponta) regenera os artefatos em `data/processed/`; a **Seção 6** do notebook 3 gera adicionalmente `neighbors.csv` — a seleção dos jogadores mais próximos de cada jogador em cada aspecto × métrica.
