---
layout: "post"
title: "Lecture 3. Source Codes"
date: 2021-03-12
tags: information_theory
usemathjax: true
---

Consider a source $$x$$ with values in $$X=\{A,B,C,D\}$$. Consider two codes for the symbols in $$X$$.

- $$A\longleftrightarrow 00$$; $$B\longleftrightarrow 01$$; $$C\longleftrightarrow 10$$; $$D\longleftrightarrow 11$$  (Fixed-length binary code)
- $$A\longleftrightarrow 0$$; $$B\longleftrightarrow 110$$; $$C\longleftrightarrow 10$$; $$D\longleftrightarrow 111$$ (Variable-length binary code)

We will want to answer these questions:

- Why do we choose one or another?

- Which properties should be satisfied by a good code?

- Is there an optimal code?

  

- **Definition:** A source code $$C$$ for a source $$x$$ and an alphabet $$X$$ is a mapping between $$X$$ and the set $$D^*$$ of all finite and semi-finite sequences of elements of the code alphabet $$D$$. When $$\#(D)=D$$, a $$D$$-ary code is obtained.

  For example if $$D=\mathbb{B}=\{0,1\}$$ then $$D^*=\{0,01,10,11,001,010,...\}$$

- **Definition:** The average length of a code $$C$$ for a memoryless source $$x$$ described by $$p$$ is 

$$
\bar{l}(C)=\sum_{j=1}^JP(x=a_j)l(a_j)
$$

​	where $$l(a_j)$$ is the number of elements of $$D$$ forming $$C(a_j)$$.

### Different Types of Code

- **Definition:** A source code $$C$$ is  <span style='color:red'><u>non-singular</u></span> if

$$
\forall x_1\in X,\space \forall x_2 \in X,\space x_1\ne x_2\Rightarrow C(x_1)\ne C(x_2)
$$

> When two sources symbols are different, they have a different code.

This propierty is not sufficient to have a good code 

- **Definition:** The extension $$C^*$$ of a code $$C$$ is a function that maps all sequences of elements of $$X$$ to a sequence of elements $$D$$ as follows:

$$
C^*(x_1,..,x_n) = C(x_1)C(x_2)..C(x_n)
$$

​	where $$C(x_1)C(x_2)..C(x_n)$$ is the concatenation of the codes associated to $$x_1$$ and $$x_2$$.

- **Definition:** A code is <span style='color:red'><u>uniquely decodable</u></span> if its extension is non-singular. Uniquely decodable doesn't mean easily decodable.
- **Definition:** A code $$C$$ is <span style='color:red'><u>prefix-free</u></span> or <span style='color:red'><u>instantaneously decodable</u></span> if no codeword is the prefix of another codeword.

---

**Example:**

| x    | Singular | Non-Singular | Uni-Decodable | Prefix-free |
| ---- | -------- | ------------ | ------------- | ----------- |
| A    | 0        | 0            | 10            | 0           |
| B    | 0        | 010          | 00            | 10          |
| C    | 0        | 01           | 11            | 110         |
| D    | 0        | 10           | 110           | 111         |

With the non-singular code, `010` may be decoded as B, AD or CA. With an uniquely decodable code `11000` is uniquely decoded as DB although we need to test different possibilities. With a prefix-free code `11000` may be decoded as CAA.

Prefix-Free codes are easy to decode.

---

### Kraft Inequality

Kraft inequality provides constraints to the codeword length in prefix-free code.

**Theorem:** Consider a prefix-free code $$C$$ defined over a $$D$$-ary alphabet $$D$$, with codeword length $$l_1, l_2,..l_J$$. Then:

$$
\sum_{j=1}^JD^{-l_j}\le 1
$$

For binary prefix-free codes we will have 

$$
\sum_{j=1}^J2^{-l_j}\le 1
$$

**Proof:** We organize the codewords in a tree of codewords. 

<img src="/assets/img/informationtheory/img_03/tree.png" style="zoom:30%;" />

Let's consider $$l_{max} = \max l_j$$ and also the code of $$c_j$$ of length $$l_j$$. If one wants to extend this codeword to get $$l_{max}$$, then one may get $$2^{l_{max}-l_j}$$ different codewords to be obtained by an extension up to $$l_{max}$$ for a binary codeword and $$D^{l_{max}-l_j}$$ for a $$D$$-ary codewords. With codewords of length $$l_{max}$$, at most $$2^{l_{max}}$$ or $$D^{l_{max}}$$ different codewords can be built.

$$
D^{l_{max}} \ge \sum_{j=1}^JD^{l_{max}-l_j}
$$

Dividing this result by $$D_{max}$$, one gets

$$
\sum_{j=1}^JD^{-l_j}\le1
$$


---

**Example**

Is it possible to have a binary prefix-free codeword of length $$l=\{1,2,3,3,3\}$$?

$$
2^{-1}+2^{-2}+2^{-3}+2^{-3}+2^{-3} =
0.5+0.25+3\cdot0.125\\
\Rightarrow 1.125>1
$$

The answer is no since Kraft's inequality is not satisfied.

---

### Kraft McMillan Inequality

What is the condition satisfied by a uniquely decodable code?

**Theorem:** A $$D$$-ary uniquely decodable code $$C$$ with $$J$$ codewords of length $$l_1,l_2,..,l_j$$ is such that 

$$
K(C)=\sum_{j=1}^{J}D^{-l_j}\le1
$$

In the binary case 

$$
K(C)=\sum_{j=1}^{J}2^{-l_j}\le1
$$

**Proof:**

One considers the evolution of $$(K(C))^n$$, where $$n\in\mathbb{N}$$. If $$K(C)>1$$ then $$(K(C))^n$$ should increase exponentially, if this is not the case then necessarily $$K(C)\le1$$. We will assume w.l.o.g that $$D=2$$.

For $$n\ge0$$, one has 

$$
\begin{align}
\Big(\sum_{j=1}^J2^{-l_j}\Big)^n&=\Big(\sum_{j_1=1}^J2^{-l_{j_1}}\Big)...\Big(\sum_{j_n=1}^J2^{-l_{j_n}}\Big)\\
&=\sum_{j_1=1}^J...\sum_{j_n=1}^J2^{-(l_{j_1}+..+l_{j_n})}
\end{align}
$$

 The exponent $$-(l_{j_1}+..+l_{j_n})$$ is the length of a bitstream formed by a $$n$$ codewords of length $$l_{j_1},..,l_{j_n})$$. Introduce $$l_{min}=min(l_j)$$ and $$l_{max}=max(l_j)$$. Consider a sequence of $$n$$ codewords, its cumulated length is between $$n\cdot l_{min}$$ and $$n\cdot l_{max}$$.

Let $$A_k$$ be the number of different sequences of $$n$$ codewords which cumulated codeword length is $$k$$ bits. Since the code is uniquely decodable $$A_k\le2^k$$.

$$
\begin{align}
\Big(\sum_{j=1}J2^{-l_j}\Big)^n&=\sum_{k=nl_{min}}^{nl_{max}}A_k\cdot 2^{-k}\\
&\le\sum_{k=nl_{min}}^{nl_{max}}2^k\cdot 2^{-k}\\
&\le n(l_{max}-l_{min})+1
\end{align}
$$

$$(K(C))^n$$ is upper-bounded by a sequence which grows linearly with $$n$$. This implies that $$K(C)\le1$$

---

**Example**

Consider $$C=\{00, 101, 111\}$$. How many sequences of $$n=2$$ codewords of cumulated length $$k=5$$ can we form?

We can form $$\{00\space101,00\space111, 101\space00,111\space00\}$$. This means $$A_5=4\le2^5$$.

**Assignment**

How many sequences of $$n=3$$ and cumulated length of $$k=7$$ can we form?

We can form $$\{0000101,0010100,1010000,0000111,0011100,1110000\}$$

$$A_7 = 6 \le2^7$$.

---

- **Definition:** a prefix-free code is <span style='color:red'><u>complete</u></span> if all semi-infinite sequences of symbols of $$D$$ can be interpreted as a sequence of codewords. For complete codes, Kraft's inequality becomes an equality. 