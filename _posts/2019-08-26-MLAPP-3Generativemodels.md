---
layout:     post
title:      "[MLAPP] Chapter 3: Generative Models for Discrete Data"
subtitle:   "Learning Notes on the book Machine Learning: A Probabilistic Perspective"
author:     "Z"
header-style: text
catalog: true
tags:
    - Machine Learning
    - Learning Notes
---

<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            displayMath: [ ['$$', '$$']],
            inlineMath: [['$','$']],
            processEscapes: true
            }
        });
    </script>
</head>


## 3.1 Introduction

* Generative Models aim to model $P(X,y)$
* While discriminative Models aim to directly model $P(y\|X)$

Applying Bayes rule to a generative classifier of the form:
$$p(y = c|x, \theta) \propto p(x|y = c, \theta) \cdot p(y = c|\theta)$$


## 3.2 Bayesian concept learning

In reality, it is considered that children learn from positive examples and obtain negative examples during an active learning process, such as 

> Parents: Look at the cute dog!   
> Children(Point out a cat): Cute Doggy!  
> Parents: That’s a cat, dear, not a dog

**But** psychological research has shown that people can learn concepts from positive examples alone (Xu and Tenenbaum 2007).

* Concept learning: We can think of learning the meaning of a word as equivalent to concept learning, which in turn is equivalent to binary classification.

* A example on number game to introduce: *posterior predictive distribution, induction, generalization gradient, hypothesis space, version space.*

> 泛化梯度（generalization gradient）是指相似性程度不同的刺激引起的不同强度的反应的一种直观表征。它表明 了泛化的水平，是泛化反应强度变化的指标。


#### 3.2.1 Likelihood

这个例子很有趣！当你看到$D=\{16\}$的时候，你会觉得它属于哪个数据集（even or power of 2 ?），但当你看到$D=\{16, 2, 8, 64\}$的时候呢？实验证明人们会倾向于选择认为他们是power of 2这个数据集。

* The key intuition is that we want to avoid **suspicious coincidences**. If the true concept was even numbers, how come we only saw numbers that happened to be powers of two?
* The **extension** of a concept is just the set of numbers that belong to it. 
* **Strong sampling assumption**: we assume that our data points are drawn uniformly and independently. Given this assumption, the probability of independently sampling N items (with replacement) from h is given by

$$
p(D|h)=\Big[\frac{1}{size(h)}\Big]^N=\Big[\frac{1}{|h|}\Big]^N
$$

* **Size principle** the model favors the simplest (smallest) hypothesis consistent with the data. This is more commonly known as **Occam’s razor**

> 奥卡姆剃刀定律（Occam's Razor, Ockham's Razor）又称“奥康的剃刀”，它是由14世纪英格兰的逻辑学家、圣方济各会修士奥卡姆的威廉（William of Occam，约1285年至1349年）提出。这个原理称为“如无必要，勿增实体”，即“简单有效原理”。正如他在《箴言书注》2卷15题说“切勿浪费较多东西去做，用较少的东西，同样可以做好的事情。

对于上面这个例子，比如，100以下的2的倍数的数字只有6个，但是是偶数的有50个，那么

$$
p(D|h_{two})=1/6,p(D|h_{even})=1/50
$$

由于出现了4个例子（{16, 2, 8, 64}），那么$h_{two}$的likelihood是$(1/6)^4$，$h_{even}$的likelihood是$(1/50)^4$， **likelihood ratio**几乎是5000:1，所以说我们倾向于认为这组数据出自power of 2的数据集。


#### 3.2.2 Prior
* Bayesian reasoning is **subjective**. 
* Different People will have different priors, also different hypothesis spaces.


#### 3.2.3 Posterior
* The posterior is simply the likelihood times the prior, normalized.
	$$
	p(h|D)=\frac{p(D|h)p(h)}{\sum_{h'\in \mathcal{H}}p(D,h')}
	$$
* In the case of most of the concepts, the prior is uniform, so the posterior is proportional to the likelihood.
* “Unnatural” concepts of “powers of 2, plus 37” and “powers of 2, except 32” have low posterior support, despite having high likelihood, due to the low prior.
* MAP estimate: 

	$$
	\widehat{h}^{MAP}=argmax_h \ p(D|h)p(h)=argmax_h \ \log p(D|h)+ \log p(h)
	$$
	
* As we get more and more data, the MAP estimate converges towards the maximum likelihood estimate or MLE: 

	$$
	\widehat{h}^{mle}=argmax_h \ p(D|h)=argmax_h \ \log p(D|h)
	$$

	In other words, if we have enough data, we see that **the data overwhelms the prior**.
	
#### 3.2.4 Posterior predictive distribution
* The posterior is our internal belief state about the world.
* We should justify them by predicting
	$$
	p(\tilde{x} \in C|D)=\sum_h p(y=1|\tilde{x},h)p(h|D)
	$$
* This is just a weighted average of the predictions of each individual hypothesis and is called **Bayes model averaging**
* ?need to be read?

#### 3.2.5 A more complex prior

## 3.3 The beta-binomial model

#### 3.3.1 Likelihood
Suppose $X_i \sim Ber(\theta) $, and $X_i = 1$ represents heads while $X_i = 0$ represents tails. $\theta \in [0, 1]$ is the rate parameter of probability of heads. If the data are iid, the likelihood has the form 

$$
p(D|theta)=\theta^{N_1}(1-\theta)^{N_0}
$$

Now suppose the data consists of the count of the number of heads $N_1$ observed in a fixed number $N$, then $N_1 \sim Bin(N,\theta)$

$$
Bin(k|n,\theta) = \binom{n}{k}\theta^{n}(1-\theta)^{n-k}
$$

#### 3.3.2 Prior
To make the math easier, it would convenient if the prior had the same form as the likelihood, 

$$
p(\theta)\propto \theta^{\gamma_1}(1-\theta)^{\gamma_2}
$$

for some prior parameters $\gamma_1$ and $\gamma_2$.

Then the posterior is 

$$p(\theta) \propto p(D|\theta)p(\theta)= \theta^{N_1+\gamma_1}(1-\theta)^{N_0+\gamma_2}$$

When the prior and the posterior have the same form, we say that the prior is a **conjugate prior** for the corresponding likelihood. 

In the case of the Bernoulli, the conjugate prior is the beta distribution, 

$$
Beta(\theta|a,b) \propto \theta^{a-1}(1-\theta)^{b-1}
$$

The parameters of the prior are called **hyper-parameters**. (we set them!)

#### 3.3.3 Posterior
If we multiply the likelihood by the beta prior we get the following posterior

$$
p(\theta|D) \propto p(D|\theta)p(\theta) = Bin(N_1|\theta, N_0+N_1)Beta(\theta|a,b)Beta(\theta|N_1+a,N_0+b)
$$

batch mode v.s. sequential mode??



