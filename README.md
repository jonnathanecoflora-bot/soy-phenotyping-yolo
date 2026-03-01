# Classificador de Doenças Foliares da Soja (4 classes) — Python + Ultralytics

Este repositório treina um modelo de IA para classificar fotos de **folhas de soja** em 4 classes foliares comuns e relevantes:

1) **Ferrugem Asiática** (*Phakopsora pachyrhizi*) → `ferrugem_asiatica`  
2) **Mancha-alvo** (*Corynespora cassiicola*) → `mancha_alvo`  
3) **DFC – Septoria (mancha parda)** (*Septoria glycines*) → `dfc_septoria`  
4) **DFC – Cercospora (crestamento foliar)** (*Cercospora kikuchii*) → `dfc_cercospora`

O foco é um pipeline **simples e reproduzível**, para que um pesquisador (mesmo leigo em programação) consiga:
- organizar fotos por classe
- treinar o modelo
- testar e gerar uma planilha (CSV) com previsões

> Importante: este repositório **não** inclui fotos/dataset. Você deve fornecer suas imagens.

---

## Para que serve (aplicabilidade prática)

- **Triagem rápida**: classificar grandes volumes de fotos e separar por doença provável.  
- **Padronização**: reduzir subjetividade na avaliação visual (base comparável por safra/área).  
- **Base para pesquisa**: criar um modelo que pode ser melhorado com mais fotos reais do campo.  
- **Ponto de partida**: o pesquisador pode ampliar o dataset, ajustar classes e melhorar a generalização.

---

## Requisitos

- Windows 10/11 (funciona em Linux também)
- Python 3.10+ (recomendado 3.11)
- VS Code (opcional)
- CPU funciona (treino pode demorar). GPU NVIDIA melhora muito.

---

## Estrutura das fotos (passo manual mais importante)

Coloque as fotos `.jpg/.png` dentro da classe correta:

```
data/external_curated/
  ferrugem_asiatica/
  mancha_alvo/
  dfc_septoria/
  dfc_cercospora/
```

### Quantidade recomendada
- **mínimo técnico**: 50 fotos por classe (só para rodar)
- **recomendado**: 200–500 fotos por classe
- fotos com diversidade: luz, fundo, celular, estádios, distância.

---

## 1) Instalar dependências (uma vez)

Abra um terminal na pasta do projeto e rode:

```bash
python -m venv .venv

# Windows:
.\.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Gerar o dataset (train/val/test)

Este passo cria automaticamente:
- ~70% treino
- ~20% validação
- ~10% teste

```bash
python src/10_prepare_dataset_curated.py
```

Saída esperada:

```
data/processed/soy_4class/
  train/
  val/
  test/
```

---

## 3) Treinar o modelo (configuração amigável para CPU)

```bash
python src/11_train_4class_cpu.py
```

O modelo treinado (pesos) fica em:

```
models/soy_4class_yolov8s/weights/best.pt
```

### Se estiver lento ou travando
Abra `src/11_train_4class_cpu.py` e reduza:

- `batch=8` → `batch=4`
- `epochs=20` → `epochs=10`
- `workers=2` → `workers=1`

---

## 4) Testar o modelo e gerar planilha (CSV)

```bash
python src/12_infer_4class_test.py
```

Gera:

```
data/phenotypes/soy_4class_predictions_test.csv
```

Colunas principais do CSV:
- `true_label`: classe real (pasta onde a foto estava)
- `pred_label`: classe prevista pelo modelo
- `pred_conf`: confiança da previsão
- `is_correct`: 1 se acertou / 0 se errou

> O CSV é gerado com separador `;` e decimal `,` para abrir corretamente no Excel PT-BR.

---

## Generalização para fotos de campo (ponto crítico)

Um modelo treinado em um tipo de imagem pode errar em outro tipo (ex.: fundo diferente, luz diferente, celular diferente).  
Para melhorar a generalização:

- use **fotos reais do campo**
- aumente diversidade (ângulos, luz, fundos, estádios)
- mantenha rótulos consistentes
- evite fotos quase iguais (mesma folha, mesmo enquadramento)

---

## (Opcional) Testar em fotos novas (fora do dataset)

Você pode colocar fotos novas em:

```
data/raw/
```

E rodar:

```bash
python src/05_infer_raw.py
```

Gera:

```
data/phenotypes/soy_raw_predictions.csv
```

---

## Checklist para postar no GitHub (não subir dataset/pesos)

Este repositório **não** deve versionar:
- `.venv/`
- `data/external/`
- `data/external_curated/`
- `data/processed/`
- `models/`
- arquivos `.pt` (pesos)

Antes de subir, confira:

```bash
git status
```

Se aparecer qualquer pasta de dataset/pesos, corrija o `.gitignore` antes de fazer commit.

---

## Licença
MIT (veja o arquivo `LICENSE`).
