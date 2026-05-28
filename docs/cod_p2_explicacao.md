# Explicação do Código `cod_p2.py`

Classificador de imagens CIFAR-10 usando uma CNN treinada do zero com PyTorch.

---

## Sumário

1. [Imports e Configurações Globais](#1-imports-e-configurações-globais)
2. [set_seed](#2-set_seed)
3. [load_dataset](#3-load_dataset)
4. [show_dataset_info](#4-show_dataset_info)
5. [show_samples](#5-show_samples)
6. [get_transforms](#6-get_transforms)
7. [split_data](#7-split_data)
8. [show_split_table](#8-show_split_table)
9. [CIFAR10Dataset](#9-cifar10dataset)
10. [get_dataloaders](#10-get_dataloaders)
11. [CIFAR10CNN](#11-cifar10cnn)
12. [train_epoch](#12-train_epoch)
13. [eval_epoch](#13-eval_epoch)
14. [train_model](#14-train_model)
15. [plot_curves](#15-plot_curves)
16. [evaluate_model](#16-evaluate_model)
17. [discuss_errors](#17-discuss_errors)
18. [main](#18-main)

---
 
## 1. Imports e Configurações Globais

```python
RANDOM_STATE = 2025
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
```

- `RANDOM_STATE`: semente única usada em todo o código para garantir reprodutibilidade dos resultados.
- `CLASSES`: lista com os 10 rótulos do CIFAR-10, onde o índice da lista corresponde ao índice numérico da classe no dataset.
- `warnings.filterwarnings("ignore")`: suprime avisos do sklearn e do PyTorch que não afetam a execução.

---

## 2. `set_seed`

```python
def set_seed(seed):
```

Garante reprodutibilidade fixando a semente aleatória em todas as bibliotecas usadas:

| Biblioteca | O que controla |
|---|---|
| `random` | sorteios Python puros |
| `numpy` | operações vetoriais/matriciais |
| `torch` | operações da CPU |
| `torch.cuda` | operações da GPU (quando disponível) |
| `cudnn.deterministic = True` | força algoritmos determinísticos na GPU |
| `cudnn.benchmark = False` | desativa a busca automática por algoritmos mais rápidos (que pode variar entre execuções) |

Sem isso, os pesos iniciais da rede e o embaralhamento dos batches seriam diferentes a cada execução.

---

## 3. `load_dataset`

```python
def load_dataset():
    raw_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    raw_test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    images = np.concatenate([raw_train.data, raw_test.data], axis=0)
    labels = np.array(raw_train.targets + raw_test.targets)
    return images, labels
```

Baixa o CIFAR-10 (se ainda não estiver em `./data`) e **une** os 50.000 exemplos de treino com os 10.000 de teste em um único array de 60.000 imagens.

O motivo de unir é fazer uma divisão customizada (70/15/15) com split estratificado, em vez de usar a divisão original 83/17.

- `images`: array NumPy de shape `(60000, 32, 32, 3)`, dtype `uint8`
- `labels`: array NumPy de shape `(60000,)` com inteiros de 0 a 9

---

## 4. `show_dataset_info`

```python
def show_dataset_info(images, labels, classes):
```

Imprime um resumo estatístico do dataset completo:

- Total de imagens, shape, dtype e range de pixel
- Tabela com contagem e proporção de cada classe

O CIFAR-10 é perfeitamente balanceado: 6.000 imagens por classe (5.000 originais de treino + 1.000 de teste).

---

## 5. `show_samples`

```python
def show_samples(images, labels, classes, seed):
```

Sorteia 10 imagens aleatórias usando `np.random.RandomState(seed)` (reprodutível) e exibe uma grade 2×5 com o nome da classe como título de cada imagem.

Salva o resultado em `random_samples.png`.

---

## 6. `get_transforms`

```python
def get_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
```

Define dois pipelines de transformação distintos:

### `train_tf` — usado no treino (com augmentation)

| Transformação | Efeito |
|---|---|
| `ToPILImage` | converte array NumPy para imagem PIL |
| `RandomCrop(32, padding=4)` | adiciona 4px de borda e recorta aleatoriamente de volta a 32×32 — simula pequenas translações |
| `RandomHorizontalFlip` | espelha horizontalmente com 50% de chance |
| `ColorJitter(b=0.2, c=0.2, s=0.2, h=0.1)` | perturbação aleatória de brilho, contraste, saturação e matiz |
| `ToTensor` | converte para tensor `[0, 1]` e reorganiza para `(C, H, W)` |
| `Normalize(mean, std)` | normaliza cada canal para média≈0 e desvio≈1 |

### `val_tf` — usado em validação e teste (sem augmentation)

Apenas `ToPILImage → ToTensor → Normalize`. Sem perturbações, para avaliação determinística.

Os valores de `mean` e `std` são os estatísticos calculados sobre o conjunto de treino original do CIFAR-10 e são convenção amplamente usada na literatura.

---

## 7. `split_data`

```python
def split_data(images, labels, seed):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, ...)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, ...)
```

Divide os 60.000 exemplos em três partições com dois passos de `StratifiedShuffleSplit`:

1. **Passo 1**: separa 15% (9.000) como teste, mantendo 85% (51.000) como treino+validação.
2. **Passo 2**: do pool de 51.000, separa `0.15/0.85 ≈ 17.6%` como validação → isso equivale a 15% do total original (9.000 imagens).

O **estratificado** garante que cada partição mantenha a mesma proporção de classes (10% cada), evitando viés de distribuição.

Resultado:

| Partição | Tamanho | % do total |
|---|---|---|
| Treino | 42.000 | 70% |
| Validação | 9.000 | 15% |
| Teste | 9.000 | 15% |

---

## 8. `show_split_table`

```python
def show_split_table(labels, idx_train, idx_val, idx_test, classes):
```

Imprime a contagem de imagens por classe em cada partição, confirmando que o split estratificado funcionou corretamente (4.200 / 900 / 900 por classe).

---

## 9. `CIFAR10Dataset`

```python
class CIFAR10Dataset(Dataset):
```

Dataset customizado do PyTorch que encapsula arrays NumPy. Implementa a interface mínima exigida pelo `DataLoader`:

- `__len__`: retorna o número de exemplos
- `__getitem__`: retorna `(imagem_transformada, label_int)` para um índice dado

A transformação é aplicada **sob demanda** (lazy), apenas quando o item é acessado, não antecipadamente.

---

## 10. `get_dataloaders`

```python
def get_dataloaders(...):
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, ...)
```

Cria três `DataLoader`s:

| Loader | Shuffle | Propósito |
|---|---|---|
| `train_loader` | Sim | embaralha os dados a cada época para variedade nos batches |
| `val_loader` | Não | avaliação determinística durante o treino |
| `test_loader` | Não | avaliação final após o treino |

- `batch_size=128`: processa 128 imagens por vez
- `num_workers=0`: sem paralelismo de carregamento (compatível com Windows sem problemas de multiprocessing)
- `generator=g`: fixa a semente do embaralhamento do treino para reprodutibilidade

---

## 11. `CIFAR10CNN`

```python
class CIFAR10CNN(nn.Module):
```

Arquitetura da rede neural convolucional. Segue o padrão VGG-like com dois blocos convolucionais seguidos de uma cabeça classificadora.

### Diagrama de fluxo

```
Input: (B, 3, 32, 32)
    ↓
[block1]
  Conv2d(3→32, 3×3, pad=1) → BN → ReLU
  Conv2d(32→32, 3×3, pad=1) → BN → ReLU
  MaxPool2d(2×2)
    ↓ (B, 32, 16, 16)
[block2]
  Conv2d(32→64, 3×3, pad=1) → BN → ReLU
  Conv2d(64→64, 3×3, pad=1) → BN → ReLU
  MaxPool2d(2×2)
    ↓ (B, 64, 8, 8)
[head]
  Dropout(0.3) → Flatten → Linear(4096→256) → ReLU → Dropout(0.5) → Linear(256→10)
    ↓
Output: (B, 10) — logits por classe
```

### Componentes explicados

| Componente | Função |
|---|---|
| `Conv2d` com `padding=1` | mantém as dimensões espaciais após a convolução |
| `BatchNorm2d` | normaliza as ativações por batch, estabiliza e acelera o treino |
| `ReLU(inplace=True)` | ativação não-linear; `inplace` economiza memória |
| `MaxPool2d(2×2)` | reduz a resolução espacial pela metade (downsampling) |
| `Dropout(0.3)` antes do flatten | regularização leve antes da cabeça |
| `Dropout(0.5)` na cabeça | regularização mais forte entre as lineares |
| `Linear(4096→256→10)` | classificador fully-connected |

**Total de parâmetros: 1.117.354**

---

## 12. `train_epoch`

```python
def train_epoch(model, loader, criterion, optimizer, device):
```

Executa uma época completa de treino:

1. `model.train()` — ativa BatchNorm e Dropout no modo de treino
2. Para cada batch:
   - Move dados para o device (CPU/GPU)
   - `optimizer.zero_grad()` — limpa gradientes do passo anterior
   - Forward pass → calcula a loss (Cross-Entropy)
   - `loss.backward()` — backpropagation, calcula gradientes
   - `optimizer.step()` — atualiza os pesos
3. Acumula loss total e acertos para calcular métricas da época

Retorna `(loss_media, acuracia)` sobre todos os batches.

---

## 13. `eval_epoch`

```python
def eval_epoch(model, loader, criterion, device):
```

Executa uma época de avaliação (validação ou teste):

- `model.eval()` — desativa Dropout e congela estatísticas do BatchNorm
- `torch.no_grad()` — desativa o cálculo de gradientes (mais rápido, menos memória)
- Não há `optimizer.step()` — os pesos não são atualizados

Retorna `(loss_media, acuracia)` sobre todos os batches.

---

## 14. `train_model`

```python
def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, epochs, patience, save_path, device):
```

Loop principal de treinamento com três mecanismos de controle:

### Early Stopping

```python
if val_acc > best_val_acc:
    patience_counter = 0
    torch.save(checkpoint, save_path)
else:
    patience_counter += 1

if patience_counter >= patience:
    # para o treino
```

Se a acurácia de validação não melhorar por `patience=10` épocas consecutivas, o treino é interrompido. Isso evita overfitting e desperdício de computação.

### Checkpoint

Sempre que a validação melhora, salva um checkpoint com:
- `epoch`: número da época
- `model_state`: pesos da rede
- `optimizer_state`: estado do otimizador
- `val_acc`: melhor acurácia de validação

### Learning Rate Scheduler

```python
scheduler = MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)
```

Reduz o LR por fator 10 nos epochs 60 e 120:
- Epochs 1–59: LR = 0.1
- Epochs 60–119: LR = 0.01
- Epochs 120+: LR = 0.001

### Log de progresso

Imprime linha por linha para épocas 1–5 e depois a cada 10 épocas.

---

## 15. `plot_curves`

```python
def plot_curves(history, best_val_acc):
```

Gera e salva `training_curves.png` com dois subplots:

- **Esquerda**: Loss de treino vs. validação por época
- **Direita**: Acurácia de treino vs. validação por época, com linha tracejada indicando a melhor val_acc

Permite diagnosticar visualmente overfitting (gap crescente entre treino e validação) ou underfitting (ambas as curvas com performance baixa).

---

## 16. `evaluate_model`

```python
def evaluate_model(model, test_loader, criterion, device, classes, checkpoint_path):
```

Avalia o melhor modelo salvo no conjunto de teste:

1. Carrega o checkpoint com os melhores pesos (`torch.load`)
2. Calcula acurácia e loss no teste via `eval_epoch`
3. Coleta todas as predições e rótulos verdadeiros
4. Imprime o **Classification Report** com precision, recall e F1 por classe
5. Gera e salva `confusion_matrix.png` como heatmap

### Métricas do Classification Report

| Métrica | Definição |
|---|---|
| Precision | dos que o modelo disse ser classe X, quantos realmente eram X |
| Recall | dos que realmente são classe X, quantos o modelo acertou |
| F1-score | média harmônica entre precision e recall |
| Support | número de exemplos reais de cada classe no teste |

---

## 17. `discuss_errors`

```python
def discuss_errors(cm, classes):
```

Analisa a matriz de confusão para identificar os pontos fracos do modelo:

1. **Classes com menor recall**: normaliza a matriz de confusão por linha (acurácia por classe) e lista as 4 piores.
2. **Pares mais confundidos**: zera a diagonal da matriz (acertos) e encontra iterativamente as 4 células com maior valor — os pares de classes que o modelo mais confunde entre si.

No resultado obtido:
- `dog → cat` (342 amostras) é a confusão mais comum — visualmente muito similares
- `frog → cat` (267) e `bird → cat` (129) indicam que `cat` atua como "classe coringa" para animais de quatro patas ou formas arredondadas

---

## 18. `main`

```python
def main():
```

Orquestra a execução completa em sequência:

```
set_seed
    ↓
load_dataset + show_dataset_info + show_samples
    ↓
get_transforms
    ↓
split_data + show_split_table + get_dataloaders
    ↓
CIFAR10CNN + print(model) + total_params
    ↓
CrossEntropyLoss + SGD(Nesterov) + MultiStepLR
    ↓
train_model + plot_curves
    ↓
evaluate_model + discuss_errors
```

### Hiperparâmetros do otimizador

| Parâmetro | Valor | Efeito |
|---|---|---|
| Otimizador | SGD com Nesterov | momentum com look-ahead, converge mais rápido que SGD padrão |
| LR inicial | 0.1 | alto para explorar o espaço de parâmetros nas primeiras épocas |
| Momentum | 0.9 | acumula gradientes nas direções consistentes |
| Weight decay | 5e-4 | regularização L2, penaliza pesos grandes |
| Loss | CrossEntropyLoss | padrão para classificação multiclasse; combina LogSoftmax + NLLLoss |

---

## Resultado obtido na execução

| Métrica | Valor |
|---|---|
| Melhor val_acc | 69.88% (epoch 18) |
| Test accuracy | 70.13% |
| Test loss | 0.8399 |
| Epochs treinados | 28 (early stopping) |

A rede parou cedo (epoch 28) antes das reduções de LR nos epochs 60 e 120. Com `patience` maior, o treinamento continuaria e provavelmente alcançaria 75–80% com os decaimentos de LR.
