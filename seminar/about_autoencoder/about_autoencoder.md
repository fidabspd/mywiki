# About AutoEncoder

본 문서에서는 일반적인 AE(AutoEncoder)는 물론 VAE(Variational AutoEncoder)와 AAE(Adversarial AutoEncoder) 등 AutoEncoder 구조를 가진 다양한 모델들을 다룬다. 이해를 돕기 위해 MNIST Dataset을 이용하여 간단한 Implement를 진행하였고 모든 코드는 [fidabspd/about-autoencoder](https://github.com/fidabspd/about-autoencoder)에서 확인 가능하다.

많은 내용을 이활석님의 강의 [오토인코더의 모든 것](https://www.youtube.com/watch?v=o_peo6U7IRM)에서 따왔다. 정말 내용이 깊고 유익한 강의이니 꼭 한번 들어보시길 추천!

## AutoEncoder

![ae_architecture](./ae_architecture.png)

흔하게 알고있는 AutoEncoder의 구조이다. 보통 원래의 데이터 $x$의 차원을 축소하였다가 다시 차원을 늘려가며 복원하는 구조로 되어있다.

모델 구조만 보면 *'오토인코더는 원래 데이터를 다시 만들어내는데 사용되는구나'* 라고 생각하기 쉽다. 하지만 오토인코더는 원래의 데이터 $x$를 잘 나타내는 lower dimension features of $x$인 latent variable $z$를 찾고자 하는 생각에 만들어진 모델이다. 그리고 그를 위해 Encoder와 Decoder형태의 모델을 사용했을 뿐, 원래의 목적은 Encoder에 있다.

### Implementation of AE

```python
class AEEncoder(torch.nn.Module):
    
    def __init__(self, in_ch, hidden_ch, kernel_size, latent_dim, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv0 = torch.nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv1 = torch.nn.Conv2d(hidden_ch, hidden_ch*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.linear_out = torch.nn.Linear(hidden_ch*2*img_size*img_size, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv0(x))
        x = torch.relu(self.conv1(x)).flatten(1)
        z = self.linear_out(x)
        return z
    
    
class AEDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, hidden_ch, kernel_size, out_ch, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_ch = hidden_ch
        self.img_size = img_size
        self.linear_in = torch.nn.Linear(latent_dim, hidden_ch//2*img_size*img_size)
        self.convt0 = torch.nn.ConvTranspose2d(hidden_ch//2, hidden_ch, kernel_size=kernel_size, padding=kernel_size//2)
        self.convt_out = torch.nn.ConvTranspose2d(hidden_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, z):
        z = torch.relu(self.linear_in(z))
        z = z.reshape((-1, self.hidden_ch//2, self.img_size, self.img_size))
        z = torch.relu(self.convt0(z))
        x_hat = torch.sigmoid(self.convt_out(z))
        return x_hat


class AutoEncoder(torch.nn.Module):
    
    def __init__(self, in_ch, latent_dim, hidden_ch, kernel_size, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = AEEncoder(in_ch, hidden_ch, kernel_size, latent_dim, img_size)
        self.decoder = AEDecoder(latent_dim, hidden_ch, kernel_size, in_ch, img_size)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
```

원래의 데이터 `x`와 모델을 통해 생성된 데이터인 `x_hat`을 비교하여 학습하면 된다. 이미지는 각 픽셀값이 0~255 값을 가지므로 0~1로 스케일링하고 CrossEntropy를 사용하도록 한다.

latent variable $z$를 뽑고싶다면 $x$를 Encoder에 통과시켜 해당 값을 사용하면 된다.

## Variational AutoEncoder

![architecture](./vae_architecture.png)

$\mu_i, \sigma_i, \epsilon_i$를 제외하고 구조만 보면 지극히 평범한 Auto Encoder와 다르지 않다. 하지만 latent variable을 찾는 것(혹은 dimention reduction)에 초점이 있는 AE와는 다르게 VAE는 gneration에 초점이 있다. 다른 말로 하면, AE는 Encoder(Dimension Reduction)에, VAE는 Decoder(Data Generating)에 초점이 있다. 즉, 그 구조가 비슷할지언정 AE와 VAE는 모델이 만들어진 생각 자체가 다르다.

개인적인 생각에 VAE는 코드로 구현된 것을 이해하는 것이 VAE의 개념을 이해하는 것 보다 훨씬 쉽다. 하지만 VAE의 진짜 중요한 부분은 '어떻게 구현하는지'가 아니라 '어떤 생각으로 만들어졌는지'라고 생각한다. 따라서 보다 쉽게 접근 하기 위해 어떻게 구현하는지를 먼저 살펴보고, 그 뒤에 완벽하게 이해하기 위해 왜 이렇게 만들어졌는지를 알아보도록 하자.

### Implementation of VAE

#### Model

```python
class VAEEncoder(torch.nn.Module):
    
    def __init__(self, in_ch, hidden_ch, kernel_size, latent_dim, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv0 = torch.nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv1 = torch.nn.Conv2d(hidden_ch, hidden_ch*2, kernel_size=kernel_size, padding=kernel_size//2)
        self.linear_out = torch.nn.Linear(hidden_ch*2*img_size*img_size, latent_dim*2)

    def forward(self, x):
        x = torch.relu(self.conv0(x))
        x = torch.relu(self.conv1(x)).flatten(1)
        x = self.linear_out(x)
        mu, sigma = x[:, :self.latent_dim], torch.exp(x[:, self.latent_dim:])
        return mu, sigma
    
    
class VAEDecoder(torch.nn.Module):
    
    def __init__(self, latent_dim, hidden_ch, kernel_size, out_ch, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_ch = hidden_ch
        self.img_size = img_size
        self.linear_in = torch.nn.Linear(latent_dim, hidden_ch//2*img_size*img_size)
        self.convt0 = torch.nn.ConvTranspose2d(hidden_ch//2, hidden_ch, kernel_size=kernel_size, padding=kernel_size//2)
        self.convt_out = torch.nn.ConvTranspose2d(hidden_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, z):
        z = torch.relu(self.linear_in(z))
        z = z.reshape((-1, self.hidden_ch//2, self.img_size, self.img_size))
        z = torch.relu(self.convt0(z))
        x_hat = torch.sigmoid(self.convt_out(z))
        return x_hat


class VariationalAutoEncoder(torch.nn.Module):
    
    def __init__(self, in_ch, latent_dim, hidden_ch, kernel_size, img_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VAEEncoder(in_ch, hidden_ch, kernel_size, latent_dim, img_size)
        self.decoder = VAEDecoder(latent_dim, hidden_ch, kernel_size, in_ch, img_size)

    def reparameterize(self, mu, sigma):
        epsilon = torch.randn(self.latent_dim).to(mu.device)
        z = mu + sigma * epsilon
        return z
        
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma
```

#### Loss

```python
class LogLikelihood(torch.nn.Module):
    
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale
        
    def forward(self, x, x_hat):
        log_likelihood = torch.mean(x * torch.log(x_hat) + (1 - x) * torch.log(1 - x_hat))
        return log_likelihood * self.scale


class KLDivergence(torch.nn.Module):

    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    def forward(self, mu, sigma):
        kl_divergence = 0.5 * torch.mean(torch.square(mu) + torch.square(sigma) - torch.log(1e-8 + torch.square(sigma)) - 1)
        return kl_divergence * self.scale


class ELBO(torch.nn.Module):
    
    def __init__(self, scale=1.):
        super().__init__()
        self.log_likelihood = LogLikelihood(scale)
        self.kl_divergence = KLDivergence(scale)
        
    def forward(self, x, x_hat, mu, sigma):
        _log_likelihood = self.log_likelihood(x, x_hat)
        _kl_divergence = self.kl_divergence(mu, sigma)
        elbo = _log_likelihood - _kl_divergence
        return elbo, _log_likelihood, _kl_divergence
```

`ELBO`의 음수값을 minimize하도록 모델을 학습하면 된다.

데이터를 생성하고 싶다면 $\mathcal{N}(0, I)$에서 추출한 sample $z$를 Decoder에 넣어주면 된다.

AE와 VAE의 차이점을 위주로 무슨 의미인지 하나씩 살펴보도록 하자.

### mu & sigma

AE와 다르게 VAE는 Encoder(`VAEEncoder`)의 output이 `z`가 아니라 `mu, sigma`이다. 이는 그리고 `VariationalAutoEncoder`의 `reparameterize`에서 random sample인 epsilon을 통해 `z = mu + sigma * epsilon` 이와 같이 z를 새로 만든다.

### ELBO

ELBO는 (Log-likelihood - KL-Divergence) 형태를 가지고 있다.

Log-likelihood는 Log-likelihood라는 단어를 사용했지만 Cross-Entropy의 부호를 양수로 바꾼 것과 같다. ELBO에 -1을 곱하고 해당 값을 minimize하면 Cross-Entropy를 사용하는 것과 완전히 같다. 여기까지는 Loss의 구성이 AE와 똑같다.

하지만 KL-Divergence를 이에 추가로 뺀다. 본래 KL-Divergence는 두 확률분포의 차이를 계산하는데 사용하는 함수인데 여기서는 어떤 두 확률분포의 차이를 계산해서 저렇게 식이 나온 것인지 알기 힘들다. 일단은 위의 `KLDivergence`를 수식으로 나타내면 다음과 같다.

$$
KL\ Divergence = \dfrac{1}{2} \sum_{j}^{K} (\mu_{ij}^2 + \sigma_{ij}^2 - \log \sigma_{ij}^2 - 1)
$$

where

- $i$: index of data
- $j$: index of feature(= dimension of latent variable($z$))
- $K$: number of feature

ELBO를 어떻게 만드는지는 알았다.

여기까지 잘 왔다면 VAE를 구현하고 사용하는 것 자체에는 큰 무리가 없다. 하지만 무슨생각으로 이렇게 만들었고, 왜 이게 잘 작동하는 것인지는 알 수 없다.

### Likelihood

모든 AutoEncoder의 Encoder($q_{\phi}$)는 $x$를 condition으로 $z$를 생성해내고, Decoder($g_{\theta}$)는 $z$를 condition으로 $x$를 생성해낸다. 이 때 Encoder를 통해 생성된 $z$와 Decoder를 통해 생성된 $x$는 다음과 같은 분포를 따른다고 할 수 있다.

$$
z \sim q_{\phi}(z|x) \qquad\qquad x \sim g_{\theta}(x|z)
$$

[Generative Model](https://ta.wiki.42maru.com/doc/generative-model-I9wNzGFeJX)에서 설명했듯이 기본적으로 Generative Model은 $x$의 PDF인 $p(x)$에 대해 Likelihood를 Maximize하고자 하는 생각에서 출발했다. 그리고 이를 위해 식을 재정의한다. (잘 모르겠다면 [Flow-Based Generative Model의 Likelihood 재정의](https://ta.wiki.42maru.com/doc/waveglow-flow-based-generative-model-detail-SPgWVHtf0U#h-likelihood) 참고)

VAE는 Likelihood $p(x)$를 다음과 같이 재정의한다.

$$
\begin{align*}
\log p(x)& = \int \log(p(x))q_{\phi}(z|x)\mathrm{d}z \\
& = \int \log \bigg( \dfrac{p(x, z)}{p(z|x)} \bigg) q_{\phi}(z|x)\mathrm{d}z \\
& = \int \log \bigg( \dfrac{p(x, z)}{q_{\phi}(z|x)} \cdot \dfrac{q_{\phi}(z|x)}{p(z|x)} \bigg) q_{\phi}(z|x)\mathrm{d}z \\
& = \int \log \bigg( \dfrac{p(x, z)}{q_{\phi}(z|x)} \bigg) q_{\phi}(z|x)\mathrm{d}z + \int \log \bigg( \dfrac{q_{\phi}(z|x)}{p(z|x)} \bigg) q_{\phi}(z|x)\mathrm{d}z
\end{align*}
$$

이 때 두번째 항은 $q_{\phi}(z|x)$와 $p(z|x)$. 두 분포 사이의 거리를 나타내는 KL-Divergence 수식과 같다. 따라서 무조건 0보다 크다. 이에 따라 다음 수식이 성립한다.

$$
\log p(x) \geq \int \log \bigg( \dfrac{p(x, z)}{q_{\phi}(z|x)} \bigg) q_{\phi}(z|x)\mathrm{d}z \\
\because \int \log \bigg( \dfrac{q_{\phi}(z|x)}{p(z|x)} \bigg) q_{\phi}(z|x)\mathrm{d}z = KL(q_{\phi}(z|x) || p(z|x))
$$

즉, 첫번째 항을 최대화하면 $p(x)$의 Log-Likelihood와 같아진다. 그리고 이 첫번째 항이 ELBO(Evidence of Lower BOund)이다.

### ELBO

ELBO는 또다시 수식 정리가 가능하다.

$$
\begin{align*}
ELBO(\phi)& = \int \log \bigg( \dfrac{p(x, z)}{q_{\phi}(z|x)} \bigg) q_{\phi}(z|x)\mathrm{d}z \\
& = \int \log \bigg( \dfrac{p(x|z)p(z)}{q_{\phi}(z|x)} \bigg) q_{\phi}(z|x)\mathrm{d}z \\
& = \int \log \Big( p(x|z) \Big) q_{\phi}(z|x)\mathrm{d}z - \int \log \bigg( \dfrac{q_{\phi}(z|x)}{p(z)} \bigg) q_{\phi}(z|x)\mathrm{d}z \\
& = \int \log \Big( p(x|z) \Big) q_{\phi}(z|x)\mathrm{d}z - KL(q_{\phi}(z|x) || p(z))
\end{align*}
$$

첫째항을 Reconstruction Error Term, 둘째항을 Regularization Term이라고 할 수 있다.

#### Reconstruction Error

첫째항은 다음과 같이 재정의 할 수 있다.

$$
\begin{align*}
&\ \int \log \Big( p(x|z) \Big) q_{\phi}(z|x)\mathrm{d}z \\
= &\ \mathbb{E}_{q_{\phi}(z|x_i)} \Big[ \log (p_{\theta}(x_i|z)) \Big] \\
\approx &\ \dfrac{1}{L}\sum_{z_{ij}} \log (p_{\theta}(x_i|z_{ij}))  \qquad (\because MonteCarlo\ technique)\\
\approx &\ \log (p_{\theta}(x_i|z_i)) \qquad (if\ L=1)
\end{align*}
$$

($L$은 $x$를 given하여 뽑는 sample $z$의 개수인데 보통 한개만 뽑는다.)

현재 MNIST Dataset을 이용하고 있으므로 이미지의 각 픽셀 값을 0~1사이로 scaling한다면 각 픽셀값들 $x_i$는 베르누이 분포를 따른다고 가정 할 수 있다. 그렇다면 위 수식은 한번 더 정리가 가능하다.

$$
\begin{align*}
&\ \log (p_{\theta}(x_i|z_i)) \\
= &\ \log \Big( p_i^{x_i}(1-p_i)^{1-x_i} \Big) \\
= &\ x_i \log p_i + (1-x_i) \log (1-p_i) \\
= &\ x_i \log \hat{x_i} + (1-x_i) \log (1-\hat{x_i})
\end{align*}
$$

즉 CrossEntropy를 최소화함으로써 Likelihood를 최대화 할 수 있다.

(정규분포 가정일 때는 MSE를 최소화함으로써 Likelihood를 최대화 할 수 있다. 수식 유도는 생략한다.)

#### Regularization

$$
KL(q_{\phi}(z|x) || p(z))
$$

$x$ condition이 붙은 $z$의 분포와 아무 조건 없을 때의 $z$의 분포 사이의 차이를 구하는 식이다. 현재 두 분포 모두 정규분포를 가정하고 있다. 두개의 다변량 정규분포 $\mathcal{N}_0(\mu_0, \Sigma_0^2), \mathcal{N}_1(\mu_1, \Sigma_1^2)$ 둘의 KLD는 다음과 같다.

$$
D_{KL}(\mathcal{N}_0||\mathcal{N}_1) = \dfrac{1}{2} \bigg\{ tr(\Sigma_1^{-1}\Sigma_0) + (\mu_1 - \mu_0)^T \Sigma_1^{-1} (\mu_1 - \mu_0) - k + \ln \dfrac{|\Sigma_1|}{|\Sigma_0|} \bigg\}  \\
where\ k\ is\ the\ dimension\ of\ vector\ space
$$

위 공식을 $q_{\phi}(z|x), p(z)$에 대해 적용하면 다음과 같다.

$$
KL(q_{\phi}(z|x) || p(z)) = \dfrac{1}{2} \sum_{j}^{K} (\mu_{ij}^2 + \sigma_{ij}^2 - \log \sigma_{ij}^2 - 1)
$$

이를 Regularization이라고 칭하는 이유는 단순히 Reconstruction만을 잘 수행하도록 하는 것 외에 $x$가 주어졌을 때의 $z$의 분포와 아무 조건 없는 $z$의 비슷하게 만들어주는 제약을 걸기 때문이다.

Variational AutoEncoder는 $z$의 sampling개념과 Regularization term을 제외하면 코드 구현에서는 AutoEncoder와 사실상 다르지 않다. 하지만 만들어진 생각의 차이가 크기 때문에 AE는 Dimension Reduction에 목적이 있고, VAE는 Generating에 목적이 있다.

여기까지 잘 이해했다면 VAE가 어떤 생각으로 만들어졌고, 왜 잘 작동하고, 어떻게 구현할 수 있는지 알게 된 것이다.
