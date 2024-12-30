**Proof**:

$$
D_{K L}(P \| Q)=\int_{-\infty}^{\infty} p(x) \log \left(\frac{p(x)}{q(x)}\right) d x
$$

for distribution $P(x)$
$$
p(x)=\frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp \left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}\right)
$$

for distribution $Q(x)$

$$
q(x)=\frac{1}{\sqrt{2 \pi \sigma_q^2}} \exp \left(-\frac{\left(x-\mu_q\right)^2}{2 \sigma_q^2}\right)
$$

Therefore we make KL to be:
$$
D_{K L}(P \| Q)=\int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp \left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}\right) \log \left(\frac{\frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp \left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}\right)}{\frac{1}{\sqrt{2 \pi \sigma_q^2}} \exp \left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_q^2}\right)}\right) d x
$$

By simplifying the argument of the logarithm:

$$
\log \left(\frac{\frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp \left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}\right)}{\frac{1}{\sqrt{2 \pi \sigma_q^2}} \exp \left(-\frac{\left(x-\mu_q\right)^2}{2 \sigma_q^2}\right)}\right)
$$

we get
$$
\begin{gathered}
\log \left(\frac{1}{\sqrt{2 \pi \sigma_p^2}} \cdot \frac{\sqrt{2 \pi \sigma_q^2}}{1}\right)+\log \left(\frac{\exp \left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}\right)}{\exp \left(-\frac{\left(x-\mu_q\right)^2}{2 \sigma_q^2}\right)}\right) \\
\quad=\log \left(\frac{\sigma_q}{\sigma_p}\right)+\left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}+\frac{\left(x-\mu_q\right)^2}{2 \sigma_q^2}\right)
\end{gathered}
$$

Then KL divergence becomes
$$
D_{K L}(P \| Q)=\int_{-\infty}^{\infty} \frac{1}{\sqrt{2 \pi \sigma_p^2}} \exp \left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}\right)\left[\log \left(\frac{\sigma_q}{\sigma_p}\right)+\left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}+\frac{\left(x-\mu_q\right)^2}{2 \sigma_q^2}\right)\right] d x
$$

$$
D_{K L}(P \| Q)=\log \left(\frac{\sigma_q}{\sigma_p}\right) \int_{-\infty}^{\infty} p(x) d x+\int_{-\infty}^{\infty} p(x)\left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}+\frac{\left(x-\mu_q\right)^2}{2 \sigma_q^2}\right) d x
$$

And remind of an integral of PDF should be $1$. Thus we leave the second integral to be:
$$
\int_{-\infty}^{\infty} p(x)\left(-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}+\frac{\left(x-\mu_q\right)^2}{2 \sigma_q^2}\right) d x
$$

Now we handle this second part integration:

$-\frac{\left(x-\mu_p\right)^2}{2 \sigma_p^2}+\frac{\left(x-\mu_q\right)^2}{2 \sigma_q^2}$ = $-\frac{x^2}{2 \sigma_p^2}+\frac{2 \mu_p x}{2 \sigma_p^2}-\frac{\mu_p^2}{2 \sigma_p^2}+\frac{x^2}{2 \sigma_q^2}-\frac{2 \mu_q x}{2 \sigma_q^2}+\frac{\mu_q^2}{2 \sigma_q^2}$ = $\left(\frac{x^2}{2 \sigma_q^2}-\frac{x^2}{2 \sigma_p^2}\right)+\left(-\frac{2 \mu_q x}{2 \sigma_q^2}+\frac{2 \mu_p x}{2 \sigma_p^2}\right)+\left(\frac{\mu_q^2}{2 \sigma_q^2}-\frac{\mu_p^2}{2 \sigma_p^2}\right)$

Therefore,
$$
I=\int_{-\infty}^{\infty} p(x)\left(\frac{x^2}{2 \sigma_q^2}-\frac{x^2}{2 \sigma_p^2}\right) d x+\int_{-\infty}^{\infty} p(x)\left(-\frac{2 \mu_q x}{2 \sigma_q^2}+\frac{2 \mu_p x}{2 \sigma_p^2}\right) d x+\int_{-\infty}^{\infty} p(x)\left(\frac{\mu_q^2}{2 \sigma_q^2}-\frac{\mu_p^2}{2 \sigma_p^2}\right) d x
$$

The first integral involves $x^2$ terms:

$$
\int_{-\infty}^{\infty} p(x)\left(\frac{x^2}{2 \sigma_q^2}-\frac{x^2}{2 \sigma_p^2}\right) d x
$$

Using the properties of the normal distribution $p(x)$, where the expectation $\mathbb{E}\left[x^2\right]=\mu^2+\sigma^2$, we can split this into:

$$
\frac{1}{2 \sigma_q^2} \int_{-\infty}^{\infty} p(x) x^2 d x-\frac{1}{2 \sigma_p^2} \int_{-\infty}^{\infty} p(x) x^2 d x
$$

For a normal distribution $N\left(\mu, \sigma^2\right)$, the second moment $\mathbb{E}\left[x^2\right]=\mu^2+\sigma^2$, so the integrals yield:

$$
\frac{1}{2 \sigma_q^2}\left(\mu_q^2+\sigma_q^2\right)-\frac{1}{2 \sigma_p^2}\left(\mu_p^2+\sigma_p^2\right)
$$

Next, we have the linear terms in $x$ :

$$
\int_{-\infty}^{\infty} p(x)\left(-\frac{2 \mu_q x}{2 \sigma_q^2}+\frac{2 \mu_p x}{2 \sigma_p^2}\right) d x
$$

These are expectation terms for a normal distribution, and since $\mathbb{E}[x]=\mu$ for a normal distribution:

$$
-\frac{\mu_q}{\sigma_q^2} \int_{-\infty}^{\infty} p(x) x d x+\frac{\mu_p}{\sigma_p^2} \int_{-\infty}^{\infty} p(x) x d x
$$

Both integrals yield the means $\mu_q$ and $\mu_p$, so this simplifies to:

$$
-\frac{\mu_q^2}{\sigma_q^2}+\frac{\mu_p^2}{\sigma_p^2}
$$

The third integral involves constant terms:

$$
\int_{-\infty}^{\infty} p(x)\left(\frac{\mu_q^2}{2 \sigma_q^2}-\frac{\mu_p^2}{2 \sigma_p^2}\right) d x
$$

Since these are constants, we can factor them out and compute the remaining integral:

$$
\left(\frac{\mu_q^2}{2 \sigma_q^2}-\frac{\mu_p^2}{2 \sigma_p^2}\right) \int_{-\infty}^{\infty} p(x) d x
$$

The integral of $p(x)$ over its entire support is 1 , so this simplifies to:

$$
\frac{\mu_q^2}{2 \sigma_q^2}-\frac{\mu_p^2}{2 \sigma_p^2}
$$

To conclude,
$$
D_{K L}(P \| Q)=\log \left(\frac{\sigma_q}{\sigma_p}\right)+\frac{\sigma_p^2+\left(\mu_p-\mu_q\right)^2}{2 \sigma_q^2}-\frac{1}{2}
$$
