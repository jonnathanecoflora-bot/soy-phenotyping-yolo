# Fenotipagem em Soja — Classificador de Doenças Foliares (4 classes)

Este repositório contém um pipeline simples e reproduzível para treinar um **classificador de imagens de folhas de soja**
usando **Python + Ultralytics (YOLOv8 para classificação)**.

A ideia é facilitar para um pesquisador (mesmo leigo em programação):
- organizar fotos em pastas (uma pasta por doença),
- rodar alguns comandos,
- obter um modelo treinado e um CSV com as previsões.

## Classes usadas (4)
Nomes de pastas (mantenha exatamente estes):

- `ferrugem_asiatica` — Ferrugem Asiática (*Phakopsora pachyrhizi*)
- `mancha_alvo` — Mancha-alvo (*Corynespora cassiicola*)
- `dfc_septoria` — DFC / Mancha parda (*Septoria glycines*)
- `dfc_cercospora` — DFC / Crestamento foliar (*Cercospora kikuchii*)

> Observação: este repositório **não** inclui dataset/fotos. Você fornece as imagens.

---

## Para que serve (aplicabilidade prática)
- **Triagem rápida**: separar grande volume de fotos por doença provável.
- **Base para pesquisa**: criar um modelo inicial e ir melhorando com fotos reais do campo.
- **Workflow padronizado**: reprodutibilidade (útil para documentação e trabalho em equipe).

**Limitação importante:** se as fotos de treino forem muito diferentes das fotos reais de campo (fundo, luz, câmera, distância),
a acurácia cai. A correção é treinar com mais diversidade e fotos reais.

---

## Requisitos
- Windows 10/11 (Linux também funciona)
- Python 3.10+ (recomendado 3.11)
- CPU funciona (treino pode demorar). GPU NVIDIA ajuda muito.

---

## Estrutura das fotos (passo manual mais importante)
Crie estas pastas e coloque imagens `.jpg/.png` dentro da classe correta:

```text
data/external_curated/
  ferrugem_asiatica/
  mancha_alvo/
  dfc_septoria/
  dfc_cercospora/
Quantas imagens?

mínimo para rodar: ~50 por classe

recomendado: 200–500+ por classe

diversidade é melhor que duplicatas (varie luz, ângulo, fundo, estádio, celular).

1) Instalar dependências (uma vez)

Abra um terminal na pasta do projeto e rode:

python -m venv .venv

Ative o ambiente:

Windows (PowerShell)

.\.venv\Scripts\Activate.ps1

Windows (cmd)

.\.venv\Scripts\activate.bat

Instale dependências:

python -m pip install --upgrade pip
pip install -r requirements.txt
2) Criar splits (train/val/test)

Isso cria automaticamente:

~70% treino

~20% validação

~10% teste

python src/10_prepare_dataset_curated.py

Saída esperada:

data/processed/soy_4class/
  train/
  val/
  test/
3) Treinar o modelo (CPU-friendly)
python src/11_train_4class_cpu.py

Pesos do modelo treinado:

models/soy_4class_yolov8s/weights/best.pt
Se estiver lento ou travando

Edite src/11_train_4class_cpu.py e reduza:

batch=8 → batch=4 (ou 2)

epochs=20 → epochs=10

workers=2 → workers=1

4) Avaliar no teste e gerar CSV
python src/12_infer_4class_test.py

Arquivo gerado:

data/phenotypes/soy_4class_predictions_test.csv

Colunas:

true_label — pasta (verdadeiro)

pred_label — previsão do modelo

pred_conf — confiança

is_correct — 1 (acertou) / 0 (errou)

O CSV é gerado com separador ; e decimal , para abrir corretamente no Excel PT-BR.

(Opcional) Rodar previsão em fotos novas (fora do dataset)

Coloque fotos novas em:

data/raw/

Rode:

python src/05_infer_raw.py

Gera:

data/phenotypes/soy_raw_predictions.csv
Publicar no GitHub (manter o repositório limpo)

Este repositório não deve subir:

.venv/

data/external/ ou data/external_curated/

data/processed/

models/

*.pt (pesos)

Antes de dar commit/push:

git status
Licença

MIT — veja o arquivo LICENSE.