# 🔢 🧠 SCTEC - Classificador Visual MNIST (Mini Projeto Módulo 2) — Pipeline de Classificação + App Desktop

[![Status](https://img.shields.io/badge/Status-Conclu%C3%ADdo-success)](#-entregáveis-da-avaliação)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Library-TensorFlow-FF6F00.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/Library-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-41CD52.svg)](https://doc.qt.io/qtforpython/)
[![MediaPipe](https://img.shields.io/badge/Vision-MediaPipe-4285F4.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

Mini-projeto avaliativo do **Módulo 2** da trilha de Inteligência Artificial.  
Pipeline completo de classificação de dígitos manuscritos (MNIST) com comparação entre **KNN**, **Random Forest** e **MLP**, análise de robustez OOD e aplicação desktop interativa com três modos de entrada: desenho com mouse, upload de imagem e rastreamento por webcam (MediaPipe / fallback OpenCV).

---

## 📚 Sumário

- [🎯 Problema de Negócio](#-problema-de-negócio)
- [🗂️ Estrutura do Repositório](#️-estrutura-do-repositório)
- [⚙️ Tecnologias Utilizadas](#️-tecnologias-utilizadas)
- [🚀 Como Executar](#-como-executar)
- [🔄 Arquitetura do Pipeline](#-arquitetura-do-pipeline)
- [📈 Fases do Projeto](#-fases-do-projeto)
- [🖥️ Aplicação Desktop](#️-aplicação-desktop)
- [🧠 Decisões Técnicas Relevantes](#-decisões-técnicas-relevantes)
- [🏆 Resultados](#-resultados)
- [💼 Veredito Técnico](#-veredito-técnico)
- [🌟 Boas Práticas](#-boas-práticas)
- [🌿 Fluxo Git (branches)](#-fluxo-git-branches)
- [🔮 Próximos Passos](#-próximos-passos)
- [🎯 Entregáveis da Avaliação](#-entregáveis-da-avaliação)
- [👤 Autor](#-autor)

---

## 🎯 Problema de Negócio

Classificar dígitos manuscritos de 0 a 9 a partir de imagens em tons de cinza 28×28 (vetorizadas em 784 features).

O projeto atende aos requisitos do mini-projeto:

- Pipeline reprodutível com caminhos relativos
- Comparação empírica entre três algoritmos (KNN, Random Forest e MLP)
- Avaliação com métricas ponderadas e matrizes de confusão 10×10
- Teste de robustez OOD (classes 4 e 9 ocultadas no treino)
- Inferência com imagens autorais
- Interface desktop completa para demonstração prática

O dataset oficial MNIST (`mnist_784` via OpenML) contém 70.000 imagens.

---

## 🗂️ Estrutura do Repositório

```
Mini Projeto/
├── app_desktop.py              # Aplicação desktop (PySide6 + MediaPipe/OpenCV)
├── notebooks/
│   └── mnist_pipeline.ipynb    # Pipeline completo (EDA → modelos → OOD → imagens)
├── src/
│   ├── __init__.py
│   └── preprocess.py           # Pré-processamento compartilhado (Otsu → 28×28)
├── models/
│   ├── mnist_mlp.keras         # MLP exportada pelo notebook
│   ├── mnist_model.keras       # Cópia canônica usada pelo app
│   └── mediapipe/              # Hand Landmarker (download sob demanda)
├── data/                       # Imagens autorais de dígitos (n0.png … n9a.png)
│   └── python.png              # Logo da interface (não é amostra de dígito)
├── docs/
│   └── Mini-Projeto-Avaliativo-Modulo2.pdf
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Tecnologias Utilizadas

| Camada | Bibliotecas |
|--------|-------------|
| Dados / ML clássico | NumPy, Pandas, scikit-learn, joblib |
| Deep Learning | TensorFlow / Keras |
| Visão | OpenCV, MediaPipe |
| Interface | PySide6 |
| Visualização | Matplotlib, Seaborn |
| Notebook | Jupyter, IPython |

---

## 🚀 Como Executar

### 1. Ambiente

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pipeline (notebook)

```bash
cd notebooks
jupyter notebook mnist_pipeline.ipynb
```

Execute as células em ordem. O notebook baixa o MNIST, treina os 3 modelos, avalia OOD e exporta os artefatos em `models/`.

### 3. Aplicação desktop

```bash
# A partir da raiz do projeto
python app_desktop.py
```

Modos de entrada: **mouse**, **upload de imagem** e **webcam** (MediaPipe com fallback OpenCV).  
Atalhos: `Espaço` = predizer · `Esc` = limpar.

---

## 🔄 Arquitetura do Pipeline

```
MNIST (OpenML)
    │
    ├─ Split estratificado 70 % treino / 10 % validação / 20 % teste
    ├─ Normalização /255.0  (após o split → sem leakage de escala)
    │
    ├─ KNN (n_neighbors=3, weights=distance)
    ├─ Random Forest (n_estimators=100, max_depth=15)
    └─ MLP 128→64→10 (adam, 10 epochs, validation_data explícito)
            │
            ├─ Matrizes de confusão + métricas weighted
            ├─ OOD: retreino sem classes 4 e 9 + baseline
            └─ Inferência em imagens autorais (src.preprocess)
                    │
                    └─ Export → models/mnist_mlp.keras + mnist_model.keras
                            │
                            └─ app_desktop.py (inferência em tempo real)
```

---

## 📈 Fases do Projeto

| Fase | Conteúdo |
|------|----------|
| **1 – EDA** | `fetch_openml('mnist_784')`, dimensões, distribuição de classes, grade 2×5 |
| **2 – Pré-processamento** | Split estratificado 70/10/20, normalização `/255.0`, justificativa técnica |
| **3 – Modelos** | KNN, Random Forest e MLP com ≥ 2 hiperparâmetros justificados cada |
| **4 – Avaliação** | Matrizes 10×10, accuracy / precision / recall / F1 weighted, tempos |
| **5 – OOD** | Classes 4 e 9 ocultadas, matriz filtrada, comparação com baseline |
| **5.3 – Imagens próprias** | Pipeline Otsu → bbox → scale → center 28×28 → predição + barras de probabilidade |

---

## 🖥️ Aplicação Desktop

- **Desenho com mouse** em canvas preto/branco  
- **Upload** de PNG/JPG com o mesmo pipeline de pré-processamento  
- **Webcam** com rastreamento de mão (MediaPipe Legacy ou Tasks; fallback OpenCV)  
- Telemetria da rede, preview 28×28 e barras de confiança  
- Carregamento prioritário de `models/mnist_model.keras` / `mnist_mlp.keras`

---

## 🧠 Decisões Técnicas Relevantes

1. **Split 70/10/20 estratificado** em duas etapas para atender a diretriz de treino / validação / teste sem leakage.  
2. **Normalização determinística** (`/255`) após o split.  
3. **MLP com `validation_data` explícito** (não apenas `validation_split` interno).  
4. **OOD com baseline**: além do modelo mascarado, avalia-se o modelo completo no mesmo subset.  
5. **Pré-processamento centralizado** em `src/preprocess.py` (notebook e app alinhados).  
6. **Escolha do melhor modelo** por F1 weighted no teste (critério empírico, não teórico).

---

## 🏆 Resultados

Os números exatos dependem da execução (semente fixa, mas ambiente pode variar levemente). Em execuções típicas neste projeto:

- KNN e MLP ficam na faixa de **~97–98 %** de F1 weighted.  
- Random Forest fica um pouco abaixo, com inferência mais rápida que o KNN.  
- A MLP é a escolha preferencial para o app por **velocidade de inferência**.  
- No teste OOD, o modelo sem classes 4/9 **sempre** atribui uma das 8 classes restantes (ausência de rejeição).

Reexecute o notebook para obter a tabela e as matrizes atualizadas da sua máquina.

---

## 💼 Veredito Técnico

1. Algoritmos clássicos (especialmente KNN) ainda são extremamente competitivos no MNIST.  
2. A MLP oferece o melhor compromisso entre desempenho e velocidade de inferência para uso em aplicação desktop.  
3. Classificadores fechados não possuem mecanismo nativo de rejeição — o teste OOD deixa isso evidente.  
4. O pré-processamento de imagens reais é tão importante quanto a escolha do algoritmo.

A aplicação desktop consolida o estudo, permitindo validação interativa dos mesmos passos realizados no notebook.

---

## 🌟 Boas Práticas

- **Caminhos relativos** e detecção automática da raiz do projeto  
- **Reprodutibilidade**: `random_state=42` e seeds do TensorFlow  
- **Código limpo**: funções bem nomeadas, tipagem com `from __future__ import annotations`  
- **Separação clara** entre estudo (notebook), utilitários (`src/`) e produto (app)  
- **Fallback robusto** para MediaPipe e carregamento de modelo  
- **Interface acessível** com atalhos de teclado e feedback visual imediato  
- **Exportação versionada** dos artefatos em `models/`  
- **`.gitignore` correto** (não versiona caches, venvs, secrets)

---

## 🌿 Fluxo Git (branches)

Organização utilizada no repositório:

| Branch | Objetivo |
|--------|----------|
| `main` | Código final estável |
| `develop` | Integração das features |
| `feat/eda-preprocessing` | Fase 1 e 2 (EDA + pré-processamento) |
| `feat/models-evaluation` | Fase 3 e 4 (modelos + métricas) |
| `feat/ood-custom-images` | Fase 5 e 5.3 (OOD + imagens autorais) |
| `feat/desktop-app` | Aplicação desktop |
| `docs/readme` | Documentação |

Commits no estilo imperativo e conciso (`implementa X`, `corrige Y`, `documenta Z`).

---

## 🔮 Próximos Passos

- Detecção de *out-of-distribution* com limiar de confiança ou OpenMax  
- Busca de hiperparâmetros (GridSearch / Keras Tuner)  
- Arquiteturas convolucionais leves (CNN)  
- Empacotamento com PyInstaller / Nuitka  
- Coleta de novas amostras para fine-tuning incremental  
- Histórico de predições e exportação de relatório

---

## 🎯 Entregáveis da Avaliação

| Entregável | Link / Localização |
|------------|--------------------|
| 📓 Notebook do pipeline | [`notebooks/mnist_pipeline.ipynb`](notebooks/mnist_pipeline.ipynb) |
| 🖥️ Aplicação Desktop | [`app_desktop.py`](app_desktop.py) |
| 🧩 Pré-processamento compartilhado | [`src/preprocess.py`](src/preprocess.py) |
| 📁 Repositório GitHub | — |
| 🎬 Vídeo de apresentação | — |
| 📂 Pasta completa do projeto | [Google Drive](https://drive.google.com/drive/folders/1PnBbA9M6TKhFfTYybqUJ4MizbM9Ekd_U?usp=sharing) |

---

## 👤 Autor

Projeto desenvolvido para o **Módulo 2** da Carreira Tech - Trilha Inteligência Artificial (SCTEC), consolidando conhecimentos em classificação supervisionada, avaliação de modelos, robustez OOD e construção de aplicações desktop com visão computacional.

**Leandro Wallauer dos Santos**
