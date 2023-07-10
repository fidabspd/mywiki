# Mathematical expression

## Variational Inference

$$
\begin{align*}
& g(x) = \log(x) \qquad Assume\ it's\ intractable \\
& Approximation\ with\ a\ tangent\ linear\ function \\
& \Rightarrow\ f(x) = \lambda x + b \\
& \\
Let \quad & f^*(\lambda) = \min_x \{ \lambda x - f(x) \}\ , \quad for\ given\ \lambda \\
Then \quad & \lambda x - f^*(\lambda) \geq g(x)\ , \quad for\ all\ \lambda, x \\
& \\
Let \quad & J(x, \lambda) = \lambda x - f^*(\lambda) \quad and \\
& \lambda_0 = \argmin_\lambda \{ J(x_0, \lambda) \},\ for\ given\ x_0 \\
Then \quad & \log(x) \approx J(x, \lambda_0) \qquad for\ x\ adjacent\ to\ x_0
\end{align*}
$$

## VAE ELBO

$$
\begin{align*}
& -\log \big( p(\mathbf{x}) \big) \\
= & -\log \big( p(\mathbf{x})  \big)\int_{-\infin}^{\infin} q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z}
\qquad \because \int_{-\infin}^{\infin} q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z}=1 \\
= & -\int_{-\infin}^{\infin} \log \big( p(\mathbf{x}) \big) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z} \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{z}|\mathbf{x})} \bigg) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z}
\qquad \because p(\mathbf{z}|\mathbf{x}) = \dfrac{p(\mathbf{x}, \mathbf{z})}{p(\mathbf{x})} \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}, \mathbf{z})\ q(\mathbf{z}|\mathbf{x})}{q(\mathbf{z}|\mathbf{x})\ p(\mathbf{z}|\mathbf{x})} \bigg) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z} \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \bigg) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z} - 
\int_{-\infin}^{\infin} \log \bigg( \dfrac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})} \bigg) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z} \\
\leq & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \bigg) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z}
\qquad \because D_{KL}(q(\mathbf{z}|\mathbf{x}) || p(\mathbf{z}|\mathbf{x})) \geq 0 \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}|\mathbf{z})p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \bigg) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z} \\
= & -\int_{-\infin}^{\infin} \log \big( p(\mathbf{x}|\mathbf{z}) \big) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z} -
\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \bigg) q(\mathbf{z}|\mathbf{x}) \mathbf{d}\mathbf{z} \\
= & -\mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x})} \bigg[ \log \big( p(\mathbf{x}|\mathbf{z}) \big) \bigg] -
\mathbb{E}_{\mathbf{z} \sim q(\mathbf{z}|\mathbf{x})} \bigg[ \log \bigg( \dfrac{p(\mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \bigg) \bigg]
\qquad \because definition\ of\ expectation
\end{align*}
$$

## Difference of VAE ELBO and Diffusion ELBO

$$
\begin{align*}
& -\log \big( p(\mathbf{x}_0) \big) \\
= & -\log \big( p(\mathbf{x}_0)  \big)\int_{-\infin}^{\infin} q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t
\qquad \because \int_{-\infin}^{\infin} q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t=1 \\
= & -\int_{-\infin}^{\infin} \log \big( p(\mathbf{x}_0) \big) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}_0, \mathbf{x}_t)}{p(\mathbf{x}_t|\mathbf{x}_0)} \bigg) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t
\qquad \because p(\mathbf{x}_t|\mathbf{x}_0) = \dfrac{p(\mathbf{x}_0, \mathbf{x}_t)}{p(\mathbf{x}_0)} \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}_0, \mathbf{x}_t)\ q(\mathbf{x}_t|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)\ p(\mathbf{x}_t|\mathbf{x}_0)} \bigg) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}_0, \mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_0)} \bigg) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t - 
\int_{-\infin}^{\infin} \log \bigg( \dfrac{q(\mathbf{x}_t|\mathbf{x}_0)}{p(\mathbf{x}_t|\mathbf{x}_0)} \bigg) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t \\
\leq & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}_0, \mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_0)} \bigg) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t
\qquad \because D_{KL}(q(\mathbf{x}_t|\mathbf{x}_0) || p(\mathbf{x}_t|\mathbf{x}_0)) \geq 0 \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}_0|\mathbf{x}_t)p(\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_0)} \bigg) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t \\
= & -\int_{-\infin}^{\infin} \log \bigg( \dfrac{p(\mathbf{x}_0|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_0)} \bigg) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t -
\int_{-\infin}^{\infin} \log \big( p(\mathbf{x}_t) \big) q(\mathbf{x}_t|\mathbf{x}_0) \mathbf{d}\mathbf{x}_t \\
= & -\mathbb{E}_{\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)} \bigg[ \log \bigg( \dfrac{p(\mathbf{x}_0|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_0)} \bigg) \bigg] -
\mathbb{E}_{\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)} \bigg[ \log \big( p(\mathbf{x}_t) \big) \bigg]
\qquad \because definition\ of\ expectation
\end{align*}
$$

## Loss (Diffusion ELBO)

$$
\begin{align*}
& -\log \big( p_{\theta}(\mathbf{x}_0) \big) \\
= & -\int_{-\infin}^{\infin} \int_{-\infin}^{\infin} ... \int_{-\infin}^{\infin} \log \big( p_{\theta}(\mathbf{x}_0) \big) q(\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T|\mathbf{x}_0) \mathbf{d}\mathbf{x}_1 \mathbf{d}\mathbf{x}_2 ... \mathbf{d}\mathbf{x}_T \\
& \qquad \because \int_{-\infin}^{\infin} \int_{-\infin}^{\infin} ... \int_{-\infin}^{\infin} q(\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T|\mathbf{x}_0) \mathbf{d}\mathbf{x}_1 \mathbf{d}\mathbf{x}_2 ... \mathbf{d}\mathbf{x}_T=1 \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \big( p_{\theta}(\mathbf{x}_0) \big) \bigg]
\qquad \because definition\ of\ expectation \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, ...,  \mathbf{x}_T)}{p_{\theta}(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, ..., \mathbf{x}_T|\mathbf{x}_0)} \bigg) \bigg]
\qquad \because p_{\theta}(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, ..., \mathbf{x}_T|\mathbf{x}_0) = \dfrac{p_{\theta}(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_t)}{p_{\theta}(\mathbf{x}_0)} \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, ...,  \mathbf{x}_T)\ q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)\ p_{\theta}(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, ..., \mathbf{x}_T|\mathbf{x}_0)} \bigg) \bigg] \\
\leq & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, ...,  \mathbf{x}_T)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg) \bigg]
\qquad \because KL\ divergence \geq 0 \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg) \bigg]
\qquad \because notation \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_T) \prod_{t=1}^{T} p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{\prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1})} \bigg) \bigg]
\qquad \because *_1\ and\ *_2 \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \big( p_{\theta}(\mathbf{x}_T) \big) + \sum_{t=1}^{T} \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \bigg) \bigg] \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \big( p_{\theta}(\mathbf{x}_T) \big) + \sum_{t=2}^{T} \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \bigg) + \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \bigg) \bigg] \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \big( p_{\theta}(\mathbf{x}_T) \big) + \sum_{t=2}^{T} \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} \cdot \dfrac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} \bigg) + \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \bigg) \bigg]
\qquad \because *_3 \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \big( p_{\theta}(\mathbf{x}_T) \big) + \sum_{t=2}^{T} \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} \bigg) + \log \bigg( \prod_{t=2}^{T} \bigg( \dfrac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)} \bigg) \cdot \dfrac{p_{\theta}(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_0)} \bigg) \bigg] \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \big( p_{\theta}(\mathbf{x}_T) \big) + \sum_{t=2}^{T} \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} \bigg) + \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_T|\mathbf{x}_0)} \bigg) \bigg] \\
= & - \mathbb{E}_{\mathbf{x}_{1:T} \sim q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \bigg[ \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_0)} \bigg) + \sum_{t=2}^{T} \log \bigg( \dfrac{p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)} \bigg) + \log \big( p_{\theta}(\mathbf{x}_0|\mathbf{x}_1) \big) \bigg] \\
= &\ \mathbb{E}_{q} \bigg[ D_{KL}(q(\mathbf{x}_T|\mathbf{x}_0) || p(\mathbf{x}_T)) + \sum_{t>1} D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) || p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log \big( p_{\theta}(\mathbf{x}_0|\mathbf{x}_1) \big) \bigg] \\
\end{align*}
$$

## Appendix

$$
\begin{align*}
*_1 & \\
& p_{\theta}(\mathbf{x}_{0:T}) \\
= & p_{\theta}(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T) \\
= & \dfrac{p_{\theta}(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T)}{p_{\theta}(\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T)}\cdot \dfrac{p_{\theta}(\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T)}{p_{\theta}(\mathbf{x}_2, \mathbf{x}_3, ..., \mathbf{x}_T)} \cdot ...\ \cdot \dfrac{p_{\theta}(\mathbf{x}_{T-1}, \mathbf{x}_T)}{p_{\theta}(\mathbf{x}_T)} \cdot p_{\theta}(\mathbf{x}_T) \\
= & p_{\theta}(\mathbf{x}_0|\mathbf{x}_1) \cdot p_{\theta}(\mathbf{x}_1|\mathbf{x}_2) \cdot ... \cdot p_{\theta}(\mathbf{x}_{T-1}|\mathbf{x}_T) \cdot p_{\theta}(\mathbf{x}_T) \\
= & p_{\theta}(\mathbf{x}_T) \prod_{t=1}^{T} p_{\theta}(\mathbf{x}_{t-1}|\mathbf{x}_t)
\end{align*}
$$

$$
\begin{align*}
*_2 & \\
& q(\mathbf{x}_{1:T}|\mathbf{x}_0) \\
= & \dfrac{q(\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, ..., \mathbf{x}_T)}{q(\mathbf{x}_0)} \\
= & \dfrac{q(\mathbf{x}_1, \mathbf{x}_0)}{q(\mathbf{x}_0)} \cdot \dfrac{q(\mathbf{x}_2, \mathbf{x}_1, \mathbf{x}_0)}{q(\mathbf{x}_1, \mathbf{x}_0)} \cdot ... \cdot \dfrac{q(\mathbf{x}_T, ..., \mathbf{x}_0)}{q(\mathbf{x}_{T-1}, ..., \mathbf{x}_0)} \\
= & \dfrac{q(\mathbf{x}_1, \mathbf{x}_0)}{q(\mathbf{x}_0)} \cdot \dfrac{q(\mathbf{x}_2, \mathbf{x}_1)}{q(\mathbf{x}_1)} \cdot ... \cdot \dfrac{q(\mathbf{x}_T, \mathbf{x}_{T-1})}{q(\mathbf{x}_{T-1})}
\qquad \because Markov\ chain\ property \\
= & q(\mathbf{x}_1|\mathbf{x}_0) \cdot q(\mathbf{x}_2|\mathbf{x}_1) \cdot ...\ \cdot q(\mathbf{x}_T|\mathbf{x}_{T-1}) \\
= & \prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1})
\end{align*}
$$

$$
\begin{align*}
*_3 & \\
& q(\mathbf{x}_t|\mathbf{x}_{t-1}) \\
= & q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)
\qquad \because Markov\ chain\ property \\
= & \dfrac{q(\mathbf{x}_t, \mathbf{x}_{t-1}, \mathbf{x}_0)}{q(\mathbf{x}_{t-1}, \mathbf{x}_0)} \\
= & \dfrac{q(\mathbf{x}_t, \mathbf{x}_{t-1}, \mathbf{x}_0)\ q(\mathbf{x}_t, \mathbf{x}_0)}{q(\mathbf{x}_t, \mathbf{x}_0)\ q(\mathbf{x}_{t-1}, \mathbf{x}_0)} \\
= & q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \cdot \dfrac{\frac{q(\mathbf{x}_t, \mathbf{x}_0)}{q(\mathbf{x}_0)}}{\frac{q(\mathbf{x}_{t-1}, \mathbf{x}_0)}{q(\mathbf{x}_0)}} \\
\end{align*}
$$

$$
\begin{align*}
*_4 \\
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) & = q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0)\ \dfrac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}
\qquad \because Bayes'\ rule\\
& = q(\mathbf{x}_t | \mathbf{x}_{t-1})\ \dfrac{q(\mathbf{x}_{t-1}|\mathbf{x}_0)}{q(\mathbf{x}_t|\mathbf{x}_0)}
\qquad \because Markov\ chain\ property \\
& \propto \exp \bigg( -\dfrac{1}{2} \bigg(
    \dfrac{(\mathbf{x}_t - \sqrt{\alpha_t}\mathbf{x}_{t-1})^2}{\beta_t}
    + \dfrac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0)^2}{1 - \bar{\alpha}_{t-1}}
    - \dfrac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1 - \bar{\alpha}_{t}}
\bigg) \bigg) \\
& \qquad\qquad\qquad \because Gaussian\ PDF = \dfrac{1}{\sigma \sqrt{2\pi}} \exp \bigg( - \dfrac{1}{2} \dfrac{(x - \mu)^2}{\sigma^2} \bigg) \\
& = \exp \bigg( -\dfrac{1}{2} \bigg(
    \dfrac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t}\mathbf{x}_t\textcolor{blue}{\mathbf{x}_{t-1}} + \alpha_t \textcolor{red}{\mathbf{x}_{t-1}^2}}{\beta_t}
    + \dfrac{\textcolor{red}{\mathbf{x}_{t-1}^2} - 2\sqrt{\bar{\alpha}_{t-1}}\textcolor{blue}{\mathbf{x}_{t-1}}\mathbf{x}_0 + \bar{\alpha}_{t-1}\mathbf{x}_0^2}{1 - \bar{\alpha}_{t-1}}
    - \dfrac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0)^2}{1 - \bar{\alpha}_{t}}
\bigg) \bigg) \\
& = \exp \bigg( -\dfrac{1}{2} \bigg(
    \textcolor{red}{
        \big( \dfrac{\alpha_t}{\beta_t} + \dfrac{1}{1 - \bar{\alpha}_{t-1}} \big) \mathbf{x}_{t-1}^2
    }
    - \textcolor{blue}{
        \big( \dfrac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \dfrac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 \big) \mathbf{x}_{t-1}
    }
    + C(\mathbf{x}_t, \mathbf{x}_0)
\bigg) \bigg) \\
& \qquad\qquad Let \quad \dfrac{\alpha_t}{\beta_t} + \dfrac{1}{1 - \bar{\alpha}_{t-1}} = A, \quad \dfrac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \dfrac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0 = B, \quad C(\mathbf{x}_t, \mathbf{x}_0) = C \\
& = \exp \bigg( -\dfrac{1}{2} \bigg(
    A\mathbf{x}_{t-1}^2 - B\mathbf{x}_{t-1} + C
\bigg) \bigg) \\
& = \exp \bigg( -\dfrac{1}{2} A \bigg(
    \mathbf{x}_{t-1}^2 - \dfrac{B}{A}\mathbf{x}_{t-1} + \big(\dfrac{B}{2A}\big)^2 - \big(\dfrac{B}{2A}\big)^2 + \dfrac{C}{A}
\bigg) \bigg) \\
& \propto \exp \bigg( -\dfrac{1}{2} \bigg(
    \dfrac{ \big( \mathbf{x}_{t-1} - \dfrac{B}{2A} \big) ^2 }{\dfrac{1}{A}} \bigg) 
\bigg)  \\
\therefore\ \tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) &:= \dfrac{B}{2A} \quad and \quad \tilde{\beta}_t := \dfrac{1}{A}
\end{align*}
$$

$$
\begin{align*}
\tilde{\beta}_t &= 1 / \big( \dfrac{\alpha_t}{\beta_t} + \dfrac{1}{1 - \bar{\alpha}_{t-1}} \big) \\
& = 1 / \big( \dfrac{\alpha_t - \bar{\alpha}_t + \beta_t}{1 - \bar{\alpha}_{t-1}} \cdot \dfrac{1}{\beta_t} \big) \\
& = \textcolor{green}{\dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \cdot \beta_t}
&\\
&\\
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) &=
    \big(
        \dfrac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \dfrac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0
    \big)
    / \big(
        \dfrac{\alpha_t}{\beta_t} + \dfrac{1}{1 - \bar{\alpha}_{t-1}}
    \big) \\
& = \big(
    \dfrac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \dfrac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0
\big)
\cdot \textcolor{green}{\dfrac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_{t}} \cdot \beta_t} \\
& = \dfrac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \dfrac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0
\end{align*}
$$

$$
\begin{align*}
*_5 \qquad\qquad\\
\tilde{\mu}_t(\mathbf{x}_t, \mathbf{x}_0) &= \dfrac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \dfrac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 \\
& = \dfrac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t
+ \dfrac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t}
    \cdot \dfrac{1}{\sqrt{\bar{\alpha}_t}} ( \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon ) \\
    & \qquad\qquad \because \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 +  \sqrt{1 - \bar{\alpha}_t} \epsilon \\
    & \qquad\qquad \quad \ \mathbf{x}_0 = \dfrac{1}{\sqrt{\bar{\alpha}_t}} ( \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon ) \\
& =
    \bigg(
        \dfrac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}
        + \dfrac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{(1 - \bar{\alpha}_t)\sqrt{\bar{\alpha}_{t}}}
    \bigg) \mathbf{x}_t
    - \dfrac{\sqrt{\bar{\alpha}_{t-1}} \beta_t \sqrt{1 - \bar{\alpha}_t}}{(1 - \bar{\alpha}_t)\sqrt{\bar{\alpha}_{t}}} \epsilon \\
& =
    \bigg(
        \dfrac{\alpha_t(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}
        + \dfrac{\beta_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}}
    \bigg) \mathbf{x}_t
    - \dfrac{\beta_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_{t}}} \epsilon \\
& =
    \bigg(
        \dfrac{1}{\sqrt{\alpha_t}} \cdot
        \dfrac{\alpha_t - \bar{\alpha}_{t} + \beta_t}{1 - \bar{\alpha}_t}
    \bigg) \mathbf{x}_t
    - \dfrac{\beta_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_{t}}} \epsilon \\
& =
    \dfrac{1}{\sqrt{\alpha_t}} \bigg(
        \mathbf{x}_t
        - \dfrac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon
    \bigg)
\end{align*}
$$
