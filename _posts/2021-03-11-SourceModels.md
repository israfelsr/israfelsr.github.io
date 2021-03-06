---
layout: "post"
title: "Lecture 2. Source Models"
date: 2021-03-11
tags: information_theory
usemathjax: true
---

We will consider the compression of messages generated by a discrete value source. A message consists of a succession of symbols $$x$$ belonging to some alphabet, $$x_i \in X$$, generated by a sequence of random variables $$x_1, x_2,..,x_n$$.

There are two families of source coding methods:

- Those requiring a model of the source (Hufmann coding, arithmetic based codes). 
- Those without requiring a model (Dictionary based techniques)

**<u>The model depends on:</u>**

- The fact that the source is <u>stationary</u> or not. Usually the stationary assumption is not correct so we use an approximate.

  > Stationary Process: It's a random process that its statistical properties don't depend on time. 

- The link between elements of the sequence: 
  - Memoryless Sources: meaning there is no link between the elements.
  - Markov Sources: A certain number of elements are related to each other. It can be first, second,..,n-order. 

#### Memoryless Sources

$$\forall x_n, n\geq1$$ with $$x_i \in X$$. If the source is <u>stationary</u>, a memoryless source can be described by:

$$
p_i = \Pr(x_n=a_i), \hspace{1cm} a_i \in X,\hspace{0.2cm} i=\{1,..,j\}\\
\Pr(x_n=a_{i_n}|x_{n-1}=a_{i_{n-1}},...,x_1=a_{i_{1}}) = \Pr(x_n=a_{i_{n}})
$$

We can see that there is no correlation between the actual symbol and the previous symbols. Using this assumption we have:

$$
\Pr(x_n=a_{i_n},x_{n-1}=a_{i_{n-1}},...,x_1=a_{i_{1}}) = \Pi_{l=1}^{n} \Pr(x_l=a_{n_{l}})
$$

> It's important to remember that conditional probabilities follow:
>
> $$
> P(A,B) = P(A|B)P(B) = P(B|A)P(A)
> $$

---

**<u>Example.</u>** 

Consider a memory less source with $$X=\{0,1\}$$ generating four samplings $$\{0,1,1,0\}$$. In this configuration we will have 

$$
\Pr(x_n = 0) = p\\
\Pr(x_n = 1) = 1-p
$$

Using this we have that:

$$
P(x_1=0,x_2=1,x_3=1,x_4=0) = p^2(1-p)^2
$$

---

We can conclude that to describe a stationary memory less source, a single vector of probability is required $$p=(p_1,..,p_n)$$ which each value corresponds to each symbol $$x_i \in X$$.

---

**<u>Assignament</u>:** 

Consider a source $$x$$ described by $$X=\{1,2,3,4\}$$ and described by $$p=\{0.1, 0.3, 0.4, 0.2\}$$.

The following code in matlab generates the numbes using the cumulative sum of the probaility vector. 

```matlab
function s = memoryless_generator(x, p, n)
%MEMORYLESS_GENERATOR
%   s = memoryless_generator(a,p,n) produces a sequence s of n random values
%   chosen from a and according to the probability distrdistribution p
csum = cumsum(p);
s = zeros(1,n);
for i=1:n
sample = rand();
s(i) = x(find(sample<csum, 1, 'first'));
end
end
```

We can also plot the histogram of the generated sampling to check if the samples are in the right proportion. 

<img src="/assets/img/informationtheory/img_02/samples.png" alt="samples" style="zoom:60%;" />

---

### Source Information

To each source symbol we may associate information. Here $$I(p_j)$$ is the information associated to the event "*The source has generated a symbol $$a_i$$*"

$$
I(p_j) = \log_2\Big(\frac{1}{p_j}\Big)\\
I(p_j) = -\log_2(p_j)\hspace{0.3cm} [bits/symbol]
$$

From this equation we can see that if $$p_j$$ is small, then $$I(p_j)$$ will be large. On the other hand, if $$p_j$$ is large then $$I(p_j)$$ is small. 
This has a lot of sense, let's for instance imagine an event with high probability. Since the event is very probable there is not much information in letting one know that the event will happen.

We can prove the following property:

$$
\begin{align}
I(x_{n-1}=a_i, x_n = a_j)& = I(p_i\cdot p_j) = -\log_2(pi\cdot p_j)\\
&= -\log_2(p_i)-\log_2(p_j)\\ 
&=I(p_i)+I(p_j)
\end{align}
$$

### Entropy

Entropy is the <u>Average Information Associated to a Source</u> and can be defined by:

$$
\begin{align}
H(x) &= -\sum p_j\log_2p_j\\
&=E(I(x))\hspace{0.3cm}[bits/sample]
\end{align}
$$

The unit of the entropy is bits if we are using $$\log_2$$ are $$bits$$ and $$nats$$ if we use $$\ln$$.

For a memoryless source we would have the following properties. 

- **Property 1:** $$H(x)\leq \log_2(J)$$

Where $$J$$ is the alphabet of the source.

Equality is achieved when all symbols are equiprobable. In this case let's notice that if $$n$$ is the amount of symbols then the probability of each one is $$p = 1/n$$. In this case $$H(x) = -\sum p_j\log_2(p_j)=-p\sum\log_2(p) = np\log_2(J) = \log_2(J)$$

- **Property 2:** Consider $$n$$ i.i.d random variables $$\{x_1,..,x_n\}$$. Then $$H(x_1,..,x_n) = nH(x_1)$$

> To the demonstration of this property, we will need the marginalization property. Marginalisation, is a method that requires summing over the possible values of one variable to determine the marginal contribution of another. If the variables are independents then the marginalization is zero.
>
> $$
> \Pr(X) = \sum_y\Pr(X,Y=y)=\sum_y\Pr(X|Y=y)\cdot\Pr(Y=y)
> $$
>
> ---
>
> **Example**
>
> Take $$x_1$$ and $$x_2$$ are iid random variables with $$X = \{0,1\}$$ and $$p(x=0)=p$$ and $$p(x=1)=(1-p)$$.
> 
> $$
> \begin{align}
> \Pr(x_1=0,x_2=1) + \Pr(x_1=0, x_2=1) &= p^2 + p(1-p)\\
> &=p(p+1-p)\\
> &=p\\&=\Pr(x=0)
> \end{align}
> $$
>
> ---

Let's prove it for the case in which $$n=2$$:

$$
\begin{align}
H(x_1,x_2) &= -\Big[\sum_{a_{1}\in X}\sum_{a_{2}\in X}\Pr(x_1=a_1,x_2=a_2)\log_2(P(x_1=a_1,x_2=a_2))\Big]\\
(iid)&= -\Big[\sum_{a_{1}\in X}\sum_{a_{2}\in X}\Pr(x_1=a_1)\Pr(x_2=a_2)(\log_2\Pr(x_1=a_1)+\log_2\Pr(x_2=a_2))\Big]\\
(marginalization)&=-\sum_{a_{1}\in X}\Pr(x_1=a_1)\log_2\Pr(x_1=a_1)-\sum_{a_{2}\in X}\Pr(x_2=a_2)\log_2\Pr(x_2=a_2)\\
&=H(x_1)+H(x_2) \\
(iid)&= 2H(x_1) 
\end{align}
$$

In the case a source is emitting with a $$R_s$$ symbol rate, then the information rate will be $$R_s\cdot H \hspace{0.3cm}[bit/sec]$$. 

### Maximum Entropy

Let's study the binary case. $$H(x) = -p\log_2(p) - (1-p)\log_2(1-p)$$. From this equation we can see that $$H(p=0)=0$$ and $$H(p=1)=0$$

<img src="/assets/img/informationtheory/img_02/h_binary.png" alt="h_binary" style="zoom:30%;" />

The source entropy reach its maximum when the symbols are equiprobable. Another way to understand Entropy is seeing it as a <u>measure of uncertainty of a random variable</u> or the <u>average amount of information required to describe the random variable</u>. 

- $$H(X)=0$$ if and only if the source $$X$$ is certain. 
- The bigger the value of the entropy, the less a priori information one has of the value of the random variable.
- It $$X$$ and $$Y$$ are $$iid$$ , then $$H(X,Y) = H(X) + H(Y)$$.

### Markov Sources

Most real world sources exhibit memory, resulting in correlated source signals. The memory can be modeled as a Markov process. 

* First order MP: the current symbol depends only on the previous symbol $$\Pr(x_n=a_{i_n}\vert x_{n-1}=a_{i_{n-1}})$$. 
* $$N$$-order MP: the current symbol depends on $$N$$ previous symbols $$\Pr(x_n=a_{i_n}\vert x_{n-1}=a_{i_{n-1}},..,x_1=a_{i_1})$$.

If the process is stationary then the probability to obtain a given sequence of symbols does not depend on time.

$$
P(x_n=a_j|x_{n-1}=a_i)=P(x_{n+k}=a_j|x_{n-1+k}=a_i)
$$

In this case, a first order MP source is characterized by its transition probability matrix $$P$$ with $$p_{ij}=\Pr(x_n=a_j\vert x_{n-1}=a_i)$$.

---

**Example**

Consider the first order markov source $$X$$ with $$P =
\begin{pmatrix}
0.8 & 0.2\\
0.1 & 0.9
\end{pmatrix}$$. (notice that rows sums one). 

In this case: 

- $$P(x_1=1\vert x_2=0)=0.2$$.
- $$P(x_1=0\vert x_2=1)=0.1$$.

**Assignment.** 

Generate 1000 samples according to the transition probability given by $$P$$.

```python
samples = np.zeros((1000))
for i in range(1,len(samples)):
    if samples[i-1] == 0:
        p = [0.8, 0.2]
        samples[i] = np.random.choice(2, size=1, p=p)
    else:
        p = [0.1, 0.9]
        samples[i] = np.random.choice(2, size=1, p=p)
```

---

### Entropy on a Markov Source

$$
\begin{align}
H(x_1,...x_n) &= -\sum_{a_{1}\in X}..\sum_{a_{n}\in X}\Pr(x_1=a_1,..,x_n=a_n)\log_2\Pr(x_1=a_1,..,x_n=a_n)\\
\end{align}
$$

> Using the Markov property that dictates that the knowledge of the previous *state* is al that we need to determine the probability distribution of the current *state*.
>
> $$
> P(X_n=i_n|X_{n-1}=i_{n-1},..,X_0=i_0) = P(X_n=i_n|X_{n-1}=i_{n-1})
> $$
>
> We obtain the chain rule for markov processes:
>
> $$
> P(x_1=a_1,..,x_n=a_n) = P(x_{n}=a_{n}|x_{n-1}=a_{n-1})..P(x_2=a_2|x_1=a_1)P(x_1=a_1)
> $$

Using this property:

$$
\begin{align}
H(x_1,...x_n) &= -\sum_{a_{1}\in X}..\sum_{a_{n}\in X}\Pr(x_1=a_1,..,x_n=a_n)\big[\log_2\Pr(x_{n}=a_{n}|x_{n-1}=a_{n-1}) +...+ \log_2\Pr(x_{1}=a_{1})\big] \\
\end{align}
$$

Now let's consider one term of this sum:

$$
-\sum_{a_{1}}..\sum_{a_{n}}\Pr(x_1=a_1,..,x_n=a_n)\log_2\Pr(x_{n}=a_{n}|x_{n-1}=a_{n-1})\\
=-\sum_{a_{n-1}}\sum_{a_{n}}\Pr(x_{n-1}=a_{n-1},x_n=a_n)\log_2\Pr(x_{n}=a_{n}|x_{n-1}=a_{n-1})\\
=H(x_n|x_{n-1}) \hspace{0.5cm}// \hspace{0.2cm} conditional \hspace{0.2cm} entropy
$$

Finally we obtain the chain rule for the entropy. 

$$
H(x_1,...x_n) = H(x_n|x_{n-1})+...+H(x_2|x_{1})+H(x_1)
$$

Now, if the source is stationary:

$$
H(x_1,...x_n) = (n-1)H(x_2|x_{1})+H(x_1)
$$

We can compute the entropy rate of the markov source

$$
\begin{align}
\lim_{n\rightarrow\infty}\frac{1}{n}H(x_1,...x_n) &= \lim_{n\rightarrow\infty}\frac{n-1}{n}H(x_2|x_{1})+\frac{1}{n}H(x_1)\\
&= H(x_2|x_{1})\\
&= -\sum_{a_{1}}\sum_{a_{2}}\Pr(x_{1}=a_{1},x_2=a_2)\log_2\Pr(x_{2}=a_{2}|x_{1}=a_{1})\\
\end{align}
$$

- **Property:** $$H(x_n\vert x_{n-1})\leq H(x_n)$$ Conditioning reduces entropy.

---

**Assignment:** Prove the property

The equality is the easiest part, it's accomplished when the variables are independent from each other.

We can use the definition of mutual information:

$$
0\le I(x_1;x_2) = H(x_1)-H(x_1|x_2)
$$

Conditioning reduces entropy can also be paraphrased into *"Knowing another random variable $$x_2$$ reduces (on average) the uncertainty of variable $$x_1$$".*

---

### Memoryless from Markov Source

When considering a first order Markov Source, one may also build a memoryless model of that source with parameters $$p = (\Pr(x=a_1),.., \Pr(x=a_j))$$. What is the link between $$P$$ and $$p$$? 

In this case we will have that:

$$
p = p\cdot P
$$

$$p$$ is a row vector.

**Proof:**

The entropy $$p_{ij}$$ of $$P$$ is $$p_{ij}=\Pr(x_n=a_j\vert x_{n-1}=a_{i})$$. If the assume that the source is stationary, then $$p=(P(x_1=a_1),..,P(x_1=a_j))=(P(x_n=a_1),..,P(x_n=a_j))$$ so our equation becomes 

$$
\begin{align}
p\cdot P &= (P(x_{n-1}=a_1),..,P(x_{n-1=a_j}))
\begin{pmatrix}
P(x_n=a_1|x_{n-1}=a_{1}),...,P(x_n=a_j|x_{n-1}=a_{1})\\
P(x_n=a_1|x_{n-1}=a_{2}),...,P(x_n=a_j|x_{n-1}=a_{2})\\
\vdots\\
P(x_n=a_1|x_{n-1}=a_{j}),...,P(x_n=a_j|x_{n-1}=a_{j})\\
\end{pmatrix}\\
&= \Big(\sum_{i=1}^{J}P(x_{n-1}=a_i)P(x_n=a_1|x_{n-1}=a_{i}),.., \sum_{i=1}^{J}P(x_{n-1}=a_i)P(x_n=a_j|x_{n-1}=a_{i})\Big)\\
&=\Big(\sum_{i=1}^{J}P(x_n=a_1,x_{n-1}=a_i),.., \sum_{i=1}^{J}P(x_n=a_j,x_{n-1}=a_i)\Big)\\
&=(P(x_n=a_1),..,P(x_n=a_j))\\
&=p
\end{align}
$$


The order of the Markov source is mainly determined by the memory of the source. The higher the order, the more complex the model. When a source alphabet is of size $J$, for stationary sources:

- Memoryless source : $$p$$ of size $$J$$, with $$J-1$$ parameters.
- First-order markov: $$\underline{\underline{p}}$$ of size $$J^2$$, $$J(J-1)$$ parameters.
- Second-order markov: $$\underline{\underline{p}}$$ is a Tensor of size $$J^3$$, $$J^2(J-1)$$ parameters.

Markov models are used in auto-completions tools for sms typing: models for the whole word from $$1$$ to $$3$$ typed characters, models for words from previous words $$\approx50000$$ words. First-order  markov model at word level requires a matrix of size $$50000^2$$. Usually $$1000-2000$$ words are used.

---

**Assignment** 

What is the stationary probability $$p$$ of

$$
P =
\begin{pmatrix}
0.8 & 0.2 & 0 \\
0.1 & 0.8 & 0.1 \\
0 & 0 & 1
\end{pmatrix}
$$

Using Matlab we can compute the stationary probability as follows

```matlab
P = [0.8 0.2 0; 0.1 0.8 0.8; 0 0 1];
[v,d] = eig(P');
p = v(:,1)/sum(v(:,1)) % [0 0 1]
p' * P % Checking Stationary Probability [0 0 1]
```

---

