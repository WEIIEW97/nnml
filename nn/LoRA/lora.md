# Low-Rank Adapation(LoRA)

Low-Rank Adapation(LoRA) freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the transformer. This makes it possible to efficiently fine tune large language models by reducing trainable parameters by a large factor.

## LoRA Linear Layer

LoRA linear layer adds a low-rank decomposition to the pre-trained weight matrix($W_0 \in \mathbb{R}^{d\times k}$) of the linear layer.
$$
W_0+\Delta W=W_0+B A
$$
, where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, and the rank $r \ll$ $\min (d, k)$.

All parameters are frozen except $A$ and $B$.

$\Delta W$ is initailized to be zero at the beginning of the training. They multiple $x \Delta W^T$ by $\frac{\alpha}{r}$ where $\alpha$ is a hyper parameter. Once $\alpha$ is tuned it can be kept the same when varing $r$.

- set $\alpha=r$ if not provided.
- get the pre-trained weight $W_0$ and freeze its grads.
- also freeze the bias parameter $b_0$.
- set scaling factor $\frac{\alpha}{r}$.
- initialize matrix $A \in \mathbb{R}^{r \times k}$ with normal linear layer.
- initialize matrix $B \in \mathbb{R}^{d \times r}$ to $0$ so that $\Delta W=B A$ is $0$ at initialization.
- compute $x W_0^T+b_0$.
- Add $\frac{\alpha}{r} x \Delta W^T=\frac{\alpha}{r} x(B A)^T=\frac{\alpha}{r} x A^T B^T$

## LoRA Embedding Layer

Similar to LoRA linear layer this adds a low-rank decomposition to the pre-trained embedding weights matrix($W_0 \in \mathbb{R}^{d\times k}$).
$$
W_0+\Delta W=W_0+B A
$$

- set $\alpha=r$ if not provided.
- get the pre-trained weight $W_0$ and freeze its grads.
- set scaling factor $\frac{\alpha}{r}$.
- initialize matrix $A \in \mathbb{R}^{r \times k}$ with normal linear layer.
- initialize matrix $B \in \mathbb{R}^{d \times r}$ to $0$ so that $\Delta W=B A$ is $0$ at initialization.
- compute the embeddings $\operatorname{onehot}(x) W_0$.
- Add $\frac{\alpha}{r} \operatorname{onehot}(x) \Delta W^T=\frac{\alpha}{r} \operatorname{onehot}(x) A^T B^T$.
