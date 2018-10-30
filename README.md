# Anchor vector optimization

## Contents

* Introduction
* Binary classification
* Multi-class classification



---
## Introduction

A different approach to learning archetypes.

Suppose that you were not familiar with the Arabic numeral system. To teach you, I might neatly write an example of each of the digits 0 through 9 to serve as archetypes. If I asked you to identify another example of one of the digits, you would probably be able to do so reasonably well by judging how similar it looks to each of the 10 archetypal examples I first wrote for you, but you might find it a challenge if the new example were scribbled messily. After you made a guess, I would tell you what the numeral really was, and you would update the notion in your mind of how that numeral looks in general. As we repeat this process, you would gradually change your understanding of what it means for an example to look like each particular digit. You would change your archetypes to match your experiences and compare each new example against those new archetypes.  

Sometimes, your experience necessitates that you expand your collection of archetypes to encapsulate dramatically different examples of the same idea. Maybe you've only ever seen the numeral 4 written with an open top, and you need to update your understanding to recognize that a 4 with a closed top is still a 4. Maybe you have to learn that the Chinese character for 4 also means 4, despite the fact that it looks nothing like the Arabic numeral 4. In this circumstance, to determine whether a new example is a 4, you may need to compare the example to your several distinct archetypes of 4, such that if the example looks like _any_ of the numeral's archetypes, you identify the example with the numeral. This set of archetypes that you carry with you and modify over time through experience is what _anchors_ you the notion of 4.

![4s]

In the following exploration, I will be developing an approach to machine learning that takes literally this idea of learning via archetype comparison. We will begin with the simplified case of binary classification, i.e. distinguishing what something _is_ from what it _is not_. After that, we will generalize our analysis to multi-class classification, using the identification of hand-written digits as a case study. In the end, even if we don't develop a revolutionary learning algorithm, hopefully we will have a new perspective on how we might approach a problem in machine learning.  

---



We might expect that we can obtain reasonable results by computing the similarity between
Can we parametrize our data distribution using a relatively small number of anchor vectors?

---

Suppose we have a multi-class dataset
* pixel brightness

[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) is commonly used as a measure of similarity between two

Consider the standard problem of handwritten digit recognition.
* 28x28 square of grayscale pixels
* unroll this into a vector embedded in 784-dimensional brightness-space where each dimension corresponds to one pixel and the value of the vector in that dimension is the brightness of the pixel, scaled between 0 and 1.

## Binary classification

To begin translating our discussion of archetypes into a mathematical language compatible with machine learning, let us consider a toy problem of .

In its simplest form, the application of anchor vectors to the binary classification problem can be framed as the following: Is there some direction where the positive class of data is likely to be concentrated? If we assume that this is true,

### Loss Function

### "Charge" property

### Negative anchors

The work we have discussed so far applies well when our data lives in a [simply connected space](https://en.wikipedia.org/wiki/Simply_connected_space) or a union of several such spaces; in other words, blobs of data are good, but holes in those blobs cause problems. Because our distance function increases as we move away from any particular anchor, the only way to

---

## Multi-class classification: MNIST

### Loss Function

### Performance

### Comparison with simple Neural Network

### Visualizing learned decision boundaries with UMAP

---

## The Future

Hierarchical classification with multi-label classification. Change the softmax output into something else.

Specify different number of anchors per class.

Dynamically add or remove anchors during training with a kind of drop-out regularization.

---

## Conclusion

* Advantages
  * straightforward interpretation
  * intuitive initialization
  * better performance than vanilla NN with same number of parameters? [WRONG]

---

![1-anchor]




[1-anchor]:images/2018-06-08/3-anchors.png
[4s]: images/4s.png
