---
layout: post
title: MLOps&#58; 
subtitle: Ways to optimize your deep learning model
author: Maria Zorkaltseva
categories: [Machine Learning Theory]
tags: [mlops, deep learning]
feature-img: "assets/img/sample_feature_img.png"
excerpt_begin_separator: <!--excerpt-->
excerpt_separator: <!--more-->
comments: true
---

<!--excerpt-->

With increasing accuracy and ability to generalize, often achieved by highly
over-parameterized models (and thus big in size), neural networks have been applied in a wide range of areas, such as real-time healthcare apptications, IoT sensors, autonomous driving, speech recognition and etc. At the same time, it became necessary to deploy the model on edge devices. This creates a problem for utilization such model architechtures with low energy consumption while saving the
high accuracy, in resource-constrained environments. In this article we will consider the ways to optimize deep learning models in size and to reduce its redundancy. Such techniques as quantization, pruning, knowledge distillation and Neural Architecture Search (NAS) will be considered.

![algorithms](/assets/img/2021-06-24-arrays-algorithmic-approaches/Competitive-Programming.jpg)
<!--more-->

<!-- TOC -->

- [Quantization](#quantization)
    - [Dynamic Quantization in practice (PyTorch)](#dynamic-quantization-in-practice-pytorch)
- [References](#references)

<!-- /TOC -->

### Quantization

Lets assume that we have NN with $$L$$ layers and learnable parameters $${W_1, W_2, ... W_L}$$ and $$\theta$$ denotes the combination of such parameters. Lets say we want to minimize following loss function:

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N l(x_i; y_i; \theta)$$

where $$(x, y)$$ is the input data and $$N$$ total number of data points. Lets denote the input hidden activations of $$i^{th}$$ as $$h_i$$ and corresponding output hidden layer activation as $$a_i$$. We assume that we storing $$\theta$$ parameters in floating point. The idea of quantization will be to reduce the precision of $$\theta$$, $$h_i$$ and $$a_i$$ with minimal impact on generalization power and accuracy.

The are two types of function quantization: uniform and non-uniform, lets consider uniform quantization operator for simplicity.

![figure 1](/assets/img/2022-04-22-model-optimization-techniques/figure1.png)
*<center>Uniform quantization (left) and non-uniform quantization (right). Real values in the continuous domain r are mapped into discrete, lower precision values in the quantized domain Q, which are marked with the orange bullets.</center>*

Uniform quantization operator $$Q$$ looks as follows:

$$Q(r) = Int(r/S) - Z$$

where $$r$$ - real values input (activation maps or weights), $$S$$ - real valued scaling factor and $$Z$$ is an integer zero point. $$Int$$ is a rounding operation (round to nearest and truncate).  It is possible to recover real values $$r$$ from the quantized values $$Q(r)$$ through an operation that is
often referred to as dequantization:

$$\tilde{r} = S(Q(r) + Z)$$

Scaling factor $$S$$ could be determined via symmetric or asymmetric quantization scheme.  This scaling factor essentially divides a given range of real values r into a number of partitions

$$S = \frac{\beta - \alpha}{2^b - 1}$$

where $$[\alpha, \beta]$$ is the clipping range and $$b$$ is quantization bit width. Usually parameters chose to be min/max of signal values $$\alpha = r_{min}$$ and $$\beta = r_{max}$$. This is so called asymmetric quantization scheme. The asymmetric scheme is adaptive and can be used in cases with imbalanced activation weights, for example, with RELU activation, since values always non-negative. For symmetric quantization scheme parameters are equal to $$-\alpha = \beta = max(\vert r_{max} \vert, \vert r_{min} \vert)$$. 

![figure 2](/assets/img/2022-04-22-model-optimization-techniques/figure2.png)
*<center>Symmetric and asymmetric quantization schemes.</center>*

Also quantizations algorithms divided according to when clipping range is determined: **dynamic** and **static** quantization. In dynamic quantization, clipping range is dynamically calculated for each activation map during runtime. It consequently leds to higher accuracy as the signal range computed for every input. Another quantization approach is static quantization, in which the clipping range is pre-calculated and static during inference.

Quantization process can be applied within the training process ([Quantization-Aware Training](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html){:target="_blank"}) and during inference stage ([Post-Training Quantization](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3){:target="_blank"}). However, application of quantization process during training introduce inaccuracy to weights and may make model to diverge. Thus, to accelerate training usually mixed precision technique is used.

#### Dynamic Quantization in practice (PyTorch)

{% highlight python %}
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
{% endhighlight %}

### References

1. [A Survey of Quantization Methods for Efficient Neural Network Inference [Arxiv]](https://arxiv.org/pdf/2103.13630.pdf){:target="_blank"}
2. [How We Scaled Bert To Serve 1+ Billion Daily Requests on CPUs [Medium]](https://medium.com/@quocnle/how-we-scaled-bert-to-serve-1-billion-daily-requests-on-cpus-d99be090db26){:target="_blank"}
3. [Awesome quantization [GitHub]](https://github.com/htqin/awesome-model-quantization){:target="_blank"}
4. [Quantization Aware Training with TensorFlow Model Optimization Toolkit - Performance with Accuracy](https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html){:target="_blank"}
5. [Introducing the Model Optimization Toolkit for TensorFlow](https://medium.com/tensorflow/introducing-the-model-optimization-toolkit-for-tensorflow-254aca1ba0a3){:target="_blank"}
6. [PyTorch quantization tutorial [PyTorch tutorial]](https://pytorch.org/docs/stable/quantization.html){:target="_blank"}