import React, { useState } from 'react';
import Layout from '@/components/Layout/Layout';

// ── Types ──────────────────────────────────────────────────────────────────

type Category = 'all' | 'foundations' | 'deep-learning' | 'training' | 'algorithms';
type Tab = 'math' | 'code' | 'qa';
type Difficulty = 'Foundations' | 'Intermediate' | 'Advanced';

interface Artifact {
  id: string;
  icon: string;
  title: string;
  description: string;
  tags: string[];
  category: Exclude<Category, 'all'>;
  difficulty: Difficulty;
  math: React.ReactNode;
  code: React.ReactNode;
  qa: React.ReactNode;
}

// ── Sub-components ─────────────────────────────────────────────────────────

function MathBlock({ children }: { children: string }) {
  return (
    <pre className="bg-surface-alt border border-border rounded-xl px-4 py-3 font-mono text-xs leading-7 text-ink overflow-x-auto whitespace-pre my-0">
      {children}
    </pre>
  );
}

function CodeBlock({ children }: { children: string }) {
  return (
    <div className="bg-[#0d1117] rounded-xl p-4 overflow-x-auto my-2">
      <pre className="font-mono text-[11.5px] leading-relaxed text-[#e6edf3] whitespace-pre">
        {children}
      </pre>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-semibold uppercase tracking-widest text-ink-muted mt-5 mb-2 first:mt-0">
      {children}
    </p>
  );
}

function Prose({ children }: { children: React.ReactNode }) {
  return <p className="text-sm text-ink-muted leading-relaxed mb-3">{children}</p>;
}

function KeyPoint({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-4 py-3 bg-primary-soft border-l-2 border-primary rounded-r-xl my-3">
      <p className="text-xs text-ink-muted leading-relaxed">{children}</p>
    </div>
  );
}

function WarnBlock({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-4 py-3 bg-amber-50 border-l-2 border-amber-400 rounded-r-xl my-3">
      <p className="text-xs text-ink-muted leading-relaxed">{children}</p>
    </div>
  );
}

function QA({ q, a }: { q: string; a: React.ReactNode }) {
  return (
    <div className="mb-4">
      <p className="text-sm font-semibold text-ink mb-1">{q}</p>
      <p className="text-sm text-ink-muted leading-relaxed">{a}</p>
    </div>
  );
}

function ComplexityTable({ rows }: { rows: [string, string, string, string][] }) {
  return (
    <table className="w-full text-xs border-collapse mb-3">
      <thead>
        <tr className="border-b border-border">
          {['Method', 'Time', 'Space', 'Best for'].map(h => (
            <th key={h} className="text-left py-2 px-2 text-ink-muted font-medium text-[10px] uppercase tracking-wider">{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i} className="border-b border-border last:border-0 hover:bg-surface-alt">
            {row.map((cell, j) => (
              <td key={j} className={`py-2 px-2 text-ink ${j > 0 && j < 3 ? 'font-mono' : ''}`}>{cell}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── Artifact data ──────────────────────────────────────────────────────────

const artifacts: Artifact[] = [
  {
    id: 'linreg',
    icon: '📐',
    title: 'Linear Regression',
    description: 'OLS derivation via MLE, gradient descent, normal equation, Ridge/Lasso regularisation.',
    tags: ['MLE', 'Gradient Descent', 'Ridge', 'Lasso'],
    category: 'foundations',
    difficulty: 'Foundations',
    math: (
      <>
        <SectionLabel>Model</SectionLabel>
        <MathBlock>{'ŷ = Xw + b      (matrix form: ŷ = Xθ, with bias absorbed)'}</MathBlock>
        <Prose>We learn weights <strong>θ</strong> by minimising Mean Squared Error. Under a Gaussian noise assumption on residuals, this is equivalent to Maximum Likelihood Estimation.</Prose>
        <SectionLabel>Loss — MLE derivation</SectionLabel>
        <MathBlock>{`y ~ N(Xθ, σ²I)

log L(θ) = -n/2 · log(2πσ²) - 1/(2σ²) · ‖y - Xθ‖²

Maximising log L  ≡  Minimising  J(θ) = ‖y - Xθ‖²`}</MathBlock>
        <SectionLabel>Normal Equation</SectionLabel>
        <MathBlock>{`∂J/∂θ = -2Xᵀ(y - Xθ) = 0\n\n⟹   θ* = (XᵀX)⁻¹ Xᵀy`}</MathBlock>
        <KeyPoint><strong>Complexity:</strong> O(nd²) to form XᵀX, O(d³) to invert. Fails if XᵀX is singular (multicollinearity). Ridge (λI) always makes it invertible.</KeyPoint>
        <SectionLabel>Gradient</SectionLabel>
        <MathBlock>{'θ ← θ - α · (1/n) Xᵀ(Xθ - y)'}</MathBlock>
        <SectionLabel>Regularisation</SectionLabel>
        <MathBlock>{`Ridge (L2):  J = ‖y-Xθ‖² + λ‖θ‖²   → θ* = (XᵀX + λI)⁻¹ Xᵀy
Lasso (L1):  J = ‖y-Xθ‖² + λ‖θ‖₁  → sparse via coordinate descent`}</MathBlock>
        <WarnBlock>Lasso sets some weights to exactly zero (feature selection). Ridge shrinks all weights. Elastic Net combines both: λ₁‖θ‖₁ + λ₂‖θ‖².</WarnBlock>
      </>
    ),
    code: (
      <>
        <CodeBlock>{`import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000, reg=None, lam=0.1):
        self.lr, self.epochs = lr, epochs
        self.reg, self.lam = reg, lam   # 'l1'|'l2'|None

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            pred = X @ self.w + self.b
            err  = pred - y               # (n,)
            dw   = (X.T @ err) / n
            db   = err.mean()
            if self.reg == 'l2':
                dw += self.lam * self.w
            elif self.reg == 'l1':
                dw += self.lam * np.sign(self.w)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return X @ self.w + self.b

    def normal_equation(self, X, y):
        lam = self.lam if self.reg == 'l2' else 0
        A = X.T @ X + lam * np.eye(X.shape[1])
        theta = np.linalg.solve(A, X.T @ y)
        self.w, self.b = theta[1:], theta[0]`}</CodeBlock>
        <SectionLabel>Complexity</SectionLabel>
        <ComplexityTable rows={[
          ['Normal Eq.', 'O(nd² + d³)', 'O(d²)', 'd small, exact solution'],
          ['Batch GD', 'O(nd·epochs)', 'O(nd)', 'large n, approximate'],
          ['SGD', 'O(d·epochs)', 'O(d)', 'huge n, online learning'],
        ]} />
      </>
    ),
    qa: (
      <>
        <QA q="Why MSE over MAE?" a="MSE is differentiable everywhere and penalises large errors more. MAE is robust to outliers but requires sub-gradients." />
        <QA q="When does the normal equation fail?" a="When XᵀX is singular — perfect multicollinearity. Ridge (adding λI) always makes it invertible." />
        <QA q="Derive the bias-variance tradeoff." a={<>E[(y-ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²(noise). Higher λ → more bias, less variance. Lower λ → less bias, more variance.</>} />
        <QA q="Why normalise features?" a="GD converges faster when features are on the same scale — the loss surface becomes more spherical rather than an elongated ellipse." />
      </>
    ),
  },
  {
    id: 'logreg',
    icon: '⚖️',
    title: 'Logistic Regression',
    description: 'Sigmoid derivation, binary cross-entropy from MLE, multi-class softmax.',
    tags: ['Sigmoid', 'Cross-Entropy', 'Softmax', 'MLE'],
    category: 'foundations',
    difficulty: 'Foundations',
    math: (
      <>
        <SectionLabel>Sigmoid</SectionLabel>
        <MathBlock>{'σ(z) = 1 / (1 + e⁻ᶻ)     z = wᵀx + b\n\nP(y=1|x) = σ(z)   →   predict 1 if z ≥ 0'}</MathBlock>
        <SectionLabel>BCE from MLE</SectionLabel>
        <MathBlock>{`y ~ Bernoulli(σ(z))

log L = Σᵢ [ yᵢ log σ(zᵢ) + (1-yᵢ) log(1-σ(zᵢ)) ]

BCE = -1/n Σᵢ [ yᵢ log ŷᵢ + (1-yᵢ) log(1-ŷᵢ) ]`}</MathBlock>
        <SectionLabel>Gradient — elegant cancellation</SectionLabel>
        <MathBlock>{'∂BCE/∂w = 1/n · Xᵀ(ŷ - y)   ← same form as linear regression!'}</MathBlock>
        <KeyPoint><strong>Why?</strong> d/dz σ(z) = σ(z)(1-σ(z)). Through the chain rule, the σ(1-σ) terms cancel with the log derivative, leaving just (ŷ-y).</KeyPoint>
        <SectionLabel>Softmax (multi-class)</SectionLabel>
        <MathBlock>{`P(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)\n\n∂L/∂Wₖ = 1/n · Xᵀ(ŷₖ - yₖ)   (same elegant form!)`}</MathBlock>
      </>
    ),
    code: (
      <>
        <CodeBlock>{`import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500):
        self.lr, self.epochs = lr, epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            z    = X @ self.w + self.b
            yhat = self._sigmoid(z)
            err  = yhat - y               # (ŷ - y)
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * err.mean()

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, thresh=0.5):
        return (self.predict_proba(X) >= thresh).astype(int)`}</CodeBlock>
      </>
    ),
    qa: (
      <>
        <QA q="Why not MSE for classification?" a="MSE with sigmoid creates a non-convex loss — many local minima. BCE gives a convex loss, so GD finds the global minimum." />
        <QA q="Log-loss blows up when ŷ→0. Fix?" a={<>Clamp predictions: <code className="font-mono text-xs bg-surface-alt px-1 rounded">np.log(np.clip(yhat, 1e-9, 1-1e-9))</code></>} />
        <QA q="How does L2 change the decision boundary?" a="It shrinks weights toward zero → boundary moves toward origin → more conservative, wider margin from training points." />
      </>
    ),
  },
  {
    id: 'transformer',
    icon: '🔀',
    title: 'Transformer Architecture',
    description: 'Scaled dot-product attention, multi-head, positional encoding, encoder-decoder from scratch.',
    tags: ['Attention', 'Multi-head', 'Positional Enc.', 'Layer Norm'],
    category: 'deep-learning',
    difficulty: 'Advanced',
    math: (
      <>
        <SectionLabel>Scaled dot-product attention</SectionLabel>
        <MathBlock>{`Q = X Wᵠ   K = X Wᴷ   V = X Wᵛ    ∈ ℝⁿˣᵈᵏ

Attention(Q,K,V) = softmax( QKᵀ / √dₖ ) · V`}</MathBlock>
        <Prose>Dividing by <strong>√dₖ</strong> prevents dot products from saturating softmax in high dimensions — where random vectors concentrate near 90°.</Prose>
        <SectionLabel>Multi-head attention</SectionLabel>
        <MathBlock>{`headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢᵛ)
MultiHead = Concat(head₁,…,headₕ) Wᴼ
dₖ = dᵥ = d_model / h`}</MathBlock>
        <SectionLabel>Positional encoding (sinusoidal)</SectionLabel>
        <MathBlock>{`PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Property: PE(pos+k) is a linear function of PE(pos)`}</MathBlock>
        <SectionLabel>Encoder layer</SectionLabel>
        <MathBlock>{`x₁ = LayerNorm(x  + MultiHead(x, x, x))   ← self-attn + residual
x₂ = LayerNorm(x₁ + FFN(x₁))              ← FFN + residual`}</MathBlock>
        <KeyPoint><strong>Residuals:</strong> Skip connections let gradients flow directly to earlier layers. Layer Norm stabilises activation distributions.</KeyPoint>
        <SectionLabel>Complexity</SectionLabel>
        <MathBlock>{`Self-attention: O(n²·d)  ← quadratic in sequence length
Memory:         O(n²)    ← the QKᵀ matrix`}</MathBlock>
      </>
    ),
    code: (
      <>
        <CodeBlock>{`import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q,K,V: (batch, heads, seq, d_k)"""
    dk     = Q.shape[-1]
    scores = Q @ K.swapaxes(-2,-1) / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V, weights

class MultiHeadAttention:
    def __init__(self, d_model, h):
        assert d_model % h == 0
        self.h, self.dk = h, d_model // h
        self.Wq = np.random.randn(d_model, d_model) * 0.02
        self.Wk = np.random.randn(d_model, d_model) * 0.02
        self.Wv = np.random.randn(d_model, d_model) * 0.02
        self.Wo = np.random.randn(d_model, d_model) * 0.02

    def _split(self, x):
        B, n, d = x.shape
        return x.reshape(B, n, self.h, self.dk).transpose(0,2,1,3)

    def __call__(self, x, mask=None):
        Q = self._split(x @ self.Wq)
        K = self._split(x @ self.Wk)
        V = self._split(x @ self.Wv)
        ctx, _ = scaled_dot_product_attention(Q, K, V, mask)
        B, h, n, dk = ctx.shape
        ctx = ctx.transpose(0,2,1,3).reshape(B, n, h*dk)
        return ctx @ self.Wo

def positional_encoding(n, d_model):
    pe  = np.zeros((n, d_model))
    pos = np.arange(n)[:, None]
    i   = np.arange(0, d_model, 2)
    div = 10000 ** (i / d_model)
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)
    return pe`}</CodeBlock>
      </>
    ),
    qa: (
      <>
        <QA q="Why scale by √dₖ?" a="Variance of dot products scales with dₖ. Dividing normalises it to 1 — keeps softmax in the informative region rather than saturating to near-uniform or near-one-hot." />
        <QA q="Multi-head vs single-head?" a="Each head attends to different aspects simultaneously (syntax, semantics, coreference). Heads are cheaper than one big head since projections are split." />
        <QA q="How does the causal mask work?" a="Lower-triangular boolean matrix. Positions where mask=False get score=−∞ → softmax ≈ 0 → future tokens contribute nothing to context." />
        <QA q="Main weakness?" a="O(n²) attention — quadratic in sequence length. Addressed by Flash Attention (IO-aware), Longformer (sparse), and linear-attention variants." />
      </>
    ),
  },
  {
    id: 'backprop',
    icon: '⬅️',
    title: 'Backpropagation',
    description: 'Chain rule, computational graphs, vanishing gradients — derive gradients for a 2-layer MLP by hand.',
    tags: ['Chain Rule', 'MLP', 'Vanishing Grad', 'ReLU'],
    category: 'training',
    difficulty: 'Intermediate',
    math: (
      <>
        <SectionLabel>2-layer MLP forward pass</SectionLabel>
        <MathBlock>{`z₁ = X W₁ + b₁         (n × h)
a₁ = ReLU(z₁)          (n × h)
z₂ = a₁ W₂ + b₂        (n × k)
ŷ  = softmax(z₂)        (n × k)
L  = -mean(y ⊙ log ŷ)   scalar`}</MathBlock>
        <SectionLabel>Backward pass</SectionLabel>
        <MathBlock>{`δ₂  = ŷ - y                      ← ∂L/∂z₂  (n × k)
dW₂ = a₁ᵀ δ₂ / n               ← ∂L/∂W₂
db₂ = δ₂.mean(axis=0)

δ₁  = (δ₂ W₂ᵀ) ⊙ (z₁ > 0)     ← ReLU mask!
dW₁ = Xᵀ δ₁ / n
db₁ = δ₁.mean(axis=0)`}</MathBlock>
        <KeyPoint><strong>ReLU gradient:</strong> d/dz ReLU(z) = 1 if z &gt; 0, else 0. The mask zeroes gradients for inactive neurons.</KeyPoint>
        <WarnBlock><strong>Vanishing gradients:</strong> sigmoid/tanh multiply gradients by ≤0.25 per layer → exponential decay with depth. ReLU avoids this for positive activations. Residual connections bypass it entirely.</WarnBlock>
      </>
    ),
    code: (
      <>
        <CodeBlock>{`class MLP:
    def __init__(self, d_in, d_hidden, d_out, lr=0.01):
        scale = 0.01
        self.W1 = np.random.randn(d_in,     d_hidden) * scale
        self.b1 = np.zeros(d_hidden)
        self.W2 = np.random.randn(d_hidden, d_out)    * scale
        self.b2 = np.zeros(d_out)
        self.lr = lr

    def forward(self, X):
        self.X  = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)       # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        e = np.exp(self.z2 - self.z2.max(1, keepdims=True))
        self.yhat = e / e.sum(1, keepdims=True)  # softmax
        return self.yhat

    def backward(self, y):
        n  = y.shape[0]
        d2 = self.yhat - y                       # δ₂
        dW2 = self.a1.T @ d2 / n
        db2 = d2.mean(0)
        d1  = (d2 @ self.W2.T) * (self.z1 > 0)  # δ₁ + ReLU mask
        dW1 = self.X.T @ d1 / n
        db1 = d1.mean(0)
        self.W2 -= self.lr * dW2;  self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1;  self.b1 -= self.lr * db1`}</CodeBlock>
      </>
    ),
    qa: (
      <>
        <QA q="Dying ReLU problem?" a="If a neuron's pre-activation is always negative, its gradient is always 0 — it never updates. Fix: Leaky ReLU max(0.01z, z) or careful initialisation." />
        <QA q="Why Xavier/He initialisation?" a={<>Xavier: σ = √(2/(fan_in + fan_out)). He: σ = √(2/fan_in) for ReLU. Both keep activation variance constant across layers.</>} />
        <QA q="What does gradient clipping do?" a="Caps the L2 norm of the gradient vector: g ← g · clip_norm / max(‖g‖, clip_norm). Prevents exploding gradients in RNNs/transformers." />
      </>
    ),
  },
  {
    id: 'optimizers',
    icon: '🏃',
    title: 'Optimizers',
    description: 'SGD, Momentum, RMSProp, Adam — update rules derived, intuition explained.',
    tags: ['Adam', 'SGD', 'Momentum', 'RMSProp', 'AdamW'],
    category: 'training',
    difficulty: 'Intermediate',
    math: (
      <>
        <SectionLabel>SGD with Momentum</SectionLabel>
        <MathBlock>{`v ← β·v + (1-β)·∇L      (β ≈ 0.9)
θ ← θ - α·v

Intuition: EMA of gradients.
Dampens oscillations in high-curvature directions.`}</MathBlock>
        <SectionLabel>RMSProp</SectionLabel>
        <MathBlock>{`v ← β·v + (1-β)·∇L²     (element-wise square)
θ ← θ - α · ∇L / (√v + ε)

Adapts lr per parameter — high-variance → smaller lr.`}</MathBlock>
        <SectionLabel>Adam = Momentum + RMSProp</SectionLabel>
        <MathBlock>{`m ← β₁·m + (1-β₁)·∇L         1st moment (mean)
v ← β₂·v + (1-β₂)·∇L²        2nd moment (variance)

Bias correction:
m̂ = m / (1-β₁ᵗ)
v̂ = v / (1-β₂ᵗ)

θ ← θ - α · m̂ / (√v̂ + ε)

Defaults: β₁=0.9, β₂=0.999, ε=1e-8, α=3e-4`}</MathBlock>
        <KeyPoint><strong>Bias correction:</strong> At t=1, m ≈ 0.1·g — massively underestimates. Dividing by (1-β₁ᵗ) corrects the cold-start bias. Irrelevant after t≈50.</KeyPoint>
      </>
    ),
    code: (
      <>
        <CodeBlock>{`class Adam:
    def __init__(self, params, lr=3e-4,
                 b1=0.9, b2=0.999, eps=1e-8):
        self.params = params   # list of np arrays
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (g, p) in enumerate(zip(grads, self.params)):
            self.m[i] = self.b1 * self.m[i] + (1-self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1-self.b2) * g**2
            mhat = self.m[i] / (1 - self.b1**self.t)
            vhat = self.v[i] / (1 - self.b2**self.t)
            self.params[i] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)`}</CodeBlock>
        <SectionLabel>Comparison</SectionLabel>
        <ComplexityTable rows={[
          ['SGD', 'O(d)', '—', 'CV with tuned lr schedule'],
          ['Momentum', 'O(2d)', '—', 'Smooth loss surfaces'],
          ['Adam', 'O(3d)', '—', 'NLP, transformers, default'],
          ['AdamW', 'O(3d)', '—', 'Proper weight decay'],
        ]} />
      </>
    ),
    qa: (
      <>
        <QA q="Adam vs SGD?" a="Adam converges faster and requires less tuning. SGD+schedule often generalises better in CV. For transformers and NLP, AdamW is the standard." />
        <QA q="Adam vs AdamW?" a="Adam applies L2 inside the gradient — adaptive scaling distorts it. AdamW applies weight decay directly to parameters, separate from the gradient update." />
        <QA q="When does Adam fail?" a="Sparse gradients with many zero dimensions — the v accumulator can shrink the effective lr too aggressively. Also known to generalise worse than SGD on some image benchmarks." />
      </>
    ),
  },
  {
    id: 'knn',
    icon: '📍',
    title: 'k-Nearest Neighbours',
    description: 'Distance metrics, curse of dimensionality, KD-tree vs brute-force trade-offs.',
    tags: ['Euclidean', 'Cosine', 'KD-tree', 'Curse of Dim.'],
    category: 'algorithms',
    difficulty: 'Foundations',
    math: (
      <>
        <SectionLabel>Distance metrics</SectionLabel>
        <MathBlock>{`Euclidean (L2):  d(x,y) = √Σᵢ (xᵢ-yᵢ)²
Manhattan (L1):  d(x,y) = Σᵢ |xᵢ-yᵢ|
Cosine:          d(x,y) = 1 - (x·y)/(‖x‖‖y‖)
Minkowski:       d(x,y) = (Σᵢ |xᵢ-yᵢ|ᵖ)^(1/p)`}</MathBlock>
        <SectionLabel>Prediction</SectionLabel>
        <MathBlock>{`Classification:  ŷ = majority_vote({yᵢ : i ∈ Nₖ(x)})
Regression:      ŷ = mean({yᵢ : i ∈ Nₖ(x)})
Weighted kNN:    ŷ = Σ (1/dᵢ)·yᵢ / Σ (1/dᵢ)`}</MathBlock>
        <SectionLabel>Curse of dimensionality</SectionLabel>
        <MathBlock>{`In d dimensions → all points become equidistant
Fraction of data within distance ε vanishes as d → ∞

→ k-NN becomes meaningless without dim. reduction
→ PCA / feature selection critical before using kNN`}</MathBlock>
      </>
    ),
    code: (
      <>
        <CodeBlock>{`class KNN:
    def __init__(self, k=5, task='classification'):
        self.k, self.task = k, task

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []
        for x in X:
            # vectorised L2 to all training points
            diffs = self.X_train - x          # (n, d)
            dists = (diffs**2).sum(axis=1)    # no sqrt needed
            idx   = np.argsort(dists)[:self.k]
            nn_y  = self.y_train[idx]
            if self.task == 'classification':
                counts = np.bincount(nn_y.astype(int))
                preds.append(np.argmax(counts))
            else:
                preds.append(nn_y.mean())
        return np.array(preds)`}</CodeBlock>
        <SectionLabel>Complexity</SectionLabel>
        <ComplexityTable rows={[
          ['Brute force', 'O(1)', 'O(n·d)', 'd high or n small'],
          ['KD-Tree', 'O(n log n)', 'O(log n) avg', 'd ≤ ~20'],
          ['Ball Tree', 'O(n log n)', 'O(log n)', 'non-Euclidean metrics'],
        ]} />
      </>
    ),
    qa: (
      <>
        <QA q="KD-tree in high dimensions?" a="Splitting hyperplanes become useless — nearly all leaves must be checked. Rule of thumb: beats brute force only for d ≤ ~20." />
        <QA q="Cosine vs Euclidean for text?" a="Use cosine! TF-IDF vectors differ greatly in norm (long vs short docs). Cosine focuses on direction — much more meaningful for semantic similarity." />
        <QA q="How to choose k?" a="Cross-validate. Small k → high variance (noisy). Large k → high bias (oversmoothed boundaries). Typically k = √n is a reasonable starting point." />
      </>
    ),
  },
];

// ── Difficulty badge ───────────────────────────────────────────────────────

function DiffBadge({ d }: { d: Difficulty }) {
  const styles: Record<Difficulty, string> = {
    'Foundations': 'bg-primary-soft text-primary',
    'Intermediate': 'bg-amber-50 text-amber-700',
    'Advanced': 'bg-red-50 text-red-700',
  };
  return (
    <span className={`text-[10px] font-semibold px-2.5 py-1 rounded-full ${styles[d]}`}>
      {d}
    </span>
  );
}

// ── Detail Panel ───────────────────────────────────────────────────────────

function Panel({ artifact, onClose }: { artifact: Artifact; onClose: () => void }) {
  const [tab, setTab] = useState<Tab>('math');

  const tabs: { id: Tab; label: string }[] = [
    { id: 'math', label: 'Math' },
    { id: 'code', label: 'Code' },
    { id: 'qa', label: 'Interview Q&A' },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-stretch justify-end" onClick={onClose}>
      <div
        className="w-full max-w-xl bg-white border-l border-border shadow-2xl overflow-y-auto flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Head */}
        <div className="sticky top-0 bg-white border-b border-border px-6 py-4 flex items-start justify-between gap-3 z-10">
          <div>
            <p className="text-[10px] font-semibold uppercase tracking-widest text-ink-muted mb-1">
              {artifact.category.replace('-', ' ')}
            </p>
            <h2 className="font-serif text-xl text-ink leading-snug">{artifact.title}</h2>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg border border-border text-ink-muted hover:text-ink hover:bg-surface-alt transition text-sm flex-shrink-0 mt-0.5"
          >
            ✕
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border px-6">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`text-xs font-medium py-3 px-4 border-b-2 transition -mb-px ${
                tab === t.id
                  ? 'border-ink text-ink'
                  : 'border-transparent text-ink-muted hover:text-ink'
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="px-6 py-6 flex-1">
          {tab === 'math' && artifact.math}
          {tab === 'code' && artifact.code}
          {tab === 'qa' && (
            <div className="space-y-1">
              {artifact.qa}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────

const FILTERS: { id: Category; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'foundations', label: 'Foundations' },
  { id: 'deep-learning', label: 'Deep Learning' },
  { id: 'training', label: 'Training' },
  { id: 'algorithms', label: 'Algorithms' },
];

export default function InterviewPrepPage() {
  const [filter, setFilter] = useState<Category>('all');
  const [selected, setSelected] = useState<Artifact | null>(null);

  const visible = filter === 'all'
    ? artifacts
    : artifacts.filter(a => a.category === filter);

  return (
    <Layout
      title="ML Interview Prep — Maria Zorkaltseva"
      description="Math derivations, intuitions, and Python implementations for ML interview preparation. Linear regression, transformers, backprop, optimizers, and more."
    >
      {/* Header */}
      <div className="border-b border-border bg-surface-alt">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-14">
          <p className="text-xs font-semibold uppercase tracking-widest text-primary mb-2">
            Interview Preparation
          </p>
          <h1 className="font-serif text-4xl md:text-5xl text-ink mb-4">
            ML Algorithms<br />from Scratch
          </h1>
          <p className="text-ink-muted max-w-xl leading-relaxed">
            Math derivations, intuitions, and clean NumPy/Python implementations.
            Everything you need to walk through an ML interview with confidence.
          </p>
          <div className="flex flex-wrap gap-2 mt-5">
            <span className="text-xs font-medium px-3 py-1.5 rounded-full bg-primary-soft text-primary border border-primary/20">
              {artifacts.length} artefacts
            </span>
            <span className="text-xs font-medium px-3 py-1.5 rounded-full bg-secondary-soft text-secondary border border-secondary/20">
              Math + Code
            </span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="border-b border-border bg-white sticky top-14 z-30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="flex gap-1 py-3 overflow-x-auto">
            {FILTERS.map(f => (
              <button
                key={f.id}
                onClick={() => setFilter(f.id)}
                className={`flex-shrink-0 text-xs font-medium px-4 py-2 rounded-full border transition-all ${
                  filter === f.id
                    ? 'bg-ink text-white border-ink'
                    : 'border-border text-ink-muted hover:text-ink hover:bg-surface-alt'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Grid */}
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10">
        <p className="text-xs text-ink-muted mb-6">
          {visible.length} artefact{visible.length !== 1 ? 's' : ''}
          {filter !== 'all' && ` in ${filter.replace('-', ' ')}`}
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {visible.map(artifact => (
            <button
              key={artifact.id}
              onClick={() => setSelected(artifact)}
              className="text-left bg-white rounded-2xl border border-border shadow-card hover:shadow-card-hover hover:border-primary/20 transition-all p-5 group post-card"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="w-10 h-10 rounded-xl bg-surface-alt border border-border flex items-center justify-center text-base">
                  {artifact.icon}
                </div>
                <svg
                  className="w-4 h-4 text-ink-muted group-hover:text-primary group-hover:translate-x-0.5 transition-all mt-1"
                  fill="none" stroke="currentColor" viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </div>

              <h3 className="font-serif text-lg text-ink leading-snug mb-2 group-hover:text-primary transition-colors">
                {artifact.title}
              </h3>
              <p className="text-xs text-ink-muted leading-relaxed mb-4">
                {artifact.description}
              </p>

              <div className="flex flex-wrap gap-1.5 mb-4">
                {artifact.tags.slice(0, 3).map(tag => (
                  <span key={tag} className="text-[10px] px-2 py-0.5 rounded bg-surface-alt text-hint border border-border">
                    {tag}
                  </span>
                ))}
              </div>

              <div className="flex items-center justify-between pt-3 border-t border-border">
                <DiffBadge d={artifact.difficulty} />
                <span className="text-[10px] text-ink-muted">Math + Code</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Panel */}
      {selected && (
        <Panel artifact={selected} onClose={() => setSelected(null)} />
      )}
    </Layout>
  );
}
