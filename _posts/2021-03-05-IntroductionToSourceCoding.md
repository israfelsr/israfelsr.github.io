---
layout: "post"
title: "Lecture 1: Introduction to Source Coding"
date: 2021-03-05
tags: information_theory
usemathjax: true
---
> This is an introductory post about source coding. Taking the example of a video encoder, this post defines some important concepts of Information Theory and Source Coding. 

### Motivation

Let's consider HD video at 25 frames per second transmitted over a wireless channel. Each pixel, when considering standard dynamic range of color is encoded in 8 bits for red, green and blue. Without compression, what is the required transmission rate?

Since we know that HD corresponds to 1920 x 1080 pixels we can compute the following.

$$
Rs = 1920\cdot1080\cdot3 [\frac{pixels}{frames}]\cdot8[\frac{bits}{pixels}]\cdot25[\frac{frames}{s}] = 1,244 x 10^9 [bits/s] = 1,244 [Gbits/s]
$$

If we consider the 5G networks targets a maximum transmission rate of 1Gb/s for a whole cell (*cumulated rate for cell users connected to 5G access point*) we see that it's not possible to send this HD video without compression even in the most advanced wireless communication system.

### Sources of Redundancy

Redundancy presented in the data can help to compress it and with that reducing the size to be stored or transmitted. We would have different forms of redundancy:

**Temporal** : successive frames of a video/audio sequence are quite similar.\\
**Spatial**: neighbourneighbours pixels of an image tend to be similar (luminescence). This also happens when using sensors, temperature, vibrations, etc tends to have similar values when the sensors are close to each other. \\
**Spectral**: the spectral domain of a signal to be compressed is not entirely occupied. Satellite multispectral images contain different spectrums (IR, Red, Green, Blue, etc) and the content on these is similar.\\
**Statistical**: imagine a text, some letters are more frequent than others. 

All these sources of redundancy needs to be identified and may be exploited by a source coder.

### Introduction to Video Coding

A video coder exploits all these redundancies. We consider frames (RBG components) which compose a picture. Each frame will contain 3 matrices, each corresponding to red, green and blue color components of the picture. 

![RBG](/assets/img/informationtheory/img_01/rbg_converter.png)

We apply a transformation from the RBG to YUV where Y corresponds to the luminescence of the image, a grey scale content, and UV, two matrices which correspond to chrominance, color content of the image. The transformation $$T(r,b,g)\rightarrow(y,u,v) \in \mathbb{R}^{3x3}$$.

---

##### Assignament 1: 
Find the transformation $$T$$.

Searching, it's easy to find that the function $$T$$ corresponds to the following transformation:

$$
Y =  0.257 \cdot R + 0.504 \cdot G + 0.098 \cdot B +  16\\
U = -0.148 \cdot R - 0.291 \cdot G + 0.439 \cdot B + 128\\
V =  0.439 \cdot R - 0.368 \cdot G - 0.071 \cdot B + 128\\
$$

Let's play a little with this transformation and check the output after using an image. For that we will use the image below.

<img src="/assets/img/informationtheory/img_01/wave.jpg" alt="img_rgb" style="zoom:5%;" />

Using the formula above we apply the transformation on the image and obtain the following figure for the $$y$$, $$u$$ and $$v$$ matrices respectively.

<img src="/assets/img/informationtheory/img_01/yuv.png" alt="yuv" style="zoom:50%;" />

---

Since most of the information is on the $$y$$ component, usually the $$y,v$$ components are subsampled. For example using a (4:2:0) subsampling. 

---

##### Assignment: 
What means the 4, 2 and 0 in the $$(4:2:0)$$ subsampling?

This subsampling is called chroma subsampling. On it, we encode images using less resolution for the luminesce or chroma information. In this representation of the form $$(J:a:b)$$,  $$J$$ corresponds to the horizontal sampling reference (width of the conceptual region). $$a$$ to the number of chrominance samples (Cr, Cb) in the first row of $$J$$ pixels. And $$b$$, the number of changes of chrominance samples (Cr, Cb) between the first and second row of $$J$$ pixels.

---

![compression](/assets/img/informationtheory/img_01/compression.jpg)

If we use the (4:2:0) subsampling as shown in the figure above, what will be the compression ratio $$C_r$$?

$$
C_r = \frac{Size_{before\hspace{0.1cm}compression}}{Size_{after\hspace{0.1cm}compression}} = \frac{3\cdot rc}{rc + \frac{1}{4}rc + \frac{1}{4}rc}=2
$$

From now on we will focus only on the $$y$$ component of the image. The procedure can be extended to the $$u$$ and $$v$$ components.

![block](/assets/img/informationtheory/img_01/block.png)

**<u>From the Transmitter:</u>** 

- **Transform**: This stage exploits the spatial redundancy to represent the pixels in a basis better suited for further compression.

$$
\begin{bmatrix}
9 & 10\\
11 & 9
\end{bmatrix}
= 9 
\begin{bmatrix}
1 & 0\\
0 & 0
\end{bmatrix}
+ 10 
\begin{bmatrix}
0 & 1\\
0 & 0
\end{bmatrix}
+ 11
\begin{bmatrix}
0 & 0\\
1 & 0
\end{bmatrix}
+ 9
\begin{bmatrix}
0 & 0\\
0 & 1
\end{bmatrix}\\
=10 
\begin{bmatrix}
1 & 1\\
1 & 1
\end{bmatrix}
\Bigg(
- 1
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
+ 1
\begin{bmatrix}
0 & 0\\
1 & 0
\end{bmatrix}
+ 0
\begin{bmatrix}
0 & 1\\
0 & 1
\end{bmatrix}\Bigg)
$$

Above we have two different bases on $$\mathbb{R}^{2x2}$$. One using the canonical matrices and the other using an <u>approximate</u> way with <u>fewer components</u>. These two transformations are <u>reversibles</u>.

- **Quantization**: It takes the transform coefficients and represents them using a (small) countable set of quantization indices. This mapping is not reversible and introduces distortion and losses but it improves significantly the compression efficiency.

- **Entropy Coding**: It exploits the statistical redundancy present in the succession of quantized transform coefficients. For example using a short binary representation for frequent indices, and long representation for less frequent indices.

The content of the output of the entropy coding block has much less redundancy. This makes it much more sensitive to transmission errors and for this reason we need to protect it, generally using a <u>channel code</u>.

- **Channel code** introduces structural redundancy to help error detection and correction.

![strucured](/assets/img/informationtheory/img_01/structural.png)

**<u>From the receiver part</u>**:

- **Entropy decoding**: To map bitstream to quantization indices.
- **Desindexacion**: To map indizes to estimates of the transform coefficients.
- **Inverse transform**: to get again pixel form of transform coefficients.

Let's consider the case in which we map $$Y_2$$ using the scheme $$E_2 = Y_2 - Y_1$$. In this case we then compress $$E_2$$ to get at the receiver part $$\tilde{E_2}$$. Finally, we sum the value of $$\tilde{Y}_1 \Rightarrow \tilde{Y}_2=\tilde{Y}_1+\tilde{E}_2$$ 

$$
\Delta_1 = Y_1 - \tilde{Y}_1\\ 
\Delta_2 = Y_2 - \tilde{Y}_2 = Y_2 - \tilde{Y}_1 - \tilde{E}_2 = E_2 + Y_1 - \tilde{Y}_1 - \tilde{E}_2\\
\Rightarrow \Delta_2 = E_2 - \tilde{E}_2 + \Delta_1
$$

We can see here that for every frame $$i$$, the error will be composed with a component due to quantization $$E_2 - \tilde{E}_2$$, but also with a component that belongs to the error of the previous frame. We can see that using the scheme the distortion accumulates.

---

##### Assignment: 
Using this same scheme evaluates $$\Delta_3$$.

Following the same logic we can obtain that:

$$
\Delta_3 = Y_3 - \tilde{Y}_3\\
\Delta_3 = Y_2 + E_3 - \tilde{Y}_2 - \tilde{E}_3\\
\Delta_3 = E_3 - \tilde{E}_3 +  Y_2 - \tilde{Y}_2 = E_3 - \tilde{E}_3 + E_2 - \tilde{E}_2 + \Delta_1
$$

---

When a local decoder is used, the encoder is able to get back $$\tilde{Y}_i$$. In this set up we can compute a new $$E'_i = Y_i - \tilde{Y}_{i-1}$$. Again, at the decoder we will obtain $$\tilde{E'}_i$$ which is added to $$\tilde{Y}_{i-1}$$. Now, evaluating the error in this case we obtain:

$$
\Delta_2 = Y_2 - \tilde{Y}_2 = E'_2 + \tilde{Y}_1 - \tilde{E'}_2 -\tilde{Y}_1\\
\Rightarrow \Delta_2 = E'_2- \tilde{E'}_2
$$

We see that in this case there is no accumulation of the distortion, the distortion of the $$i$$-frame only depends on the quantization. This structure of video coder is still valid for H265 video coder (vvc) and it has been proposed in the late 80's.