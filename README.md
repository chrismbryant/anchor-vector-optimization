# Anchor vector optimization

## Contents

* Introduction
* Binary classification
* Multi-class classification



---
## Introduction

Suppose that you were not familiar with the Arabic numeral system. To teach it to you, I might neatly write an example of each of the digits 0 through 9, then ask you to identify a new example of one of the digits. If my handwriting stayed relatively consistent, you would probably be able to answer reasonably well simply by observing which of my 10 archetypal examples this new example looks most similar to. You might find this task challenging, however, if I instead messily scribbled the new example. In circumstances like this, after you make a guess, I would tell you what the numeral really was, and you would update the notion in your mind of what that numeral generally looks like. As we repeat this process, you would gradually change your understanding of what it means for an example to _look like_ each particular digit. You would change your mental archetypes to match your experiences and compare each new example against those new archetypes.  

Sometimes, your experience necessitates that you expand your collection of archetypes to encapsulate dramatically different examples of the same idea. Maybe you've only ever seen the numeral 4 written with an open top, and you need to update your understanding to recognize that a 4 with a closed top is also a 4. Maybe you have to learn that the Chinese character for 4 also means 4, despite the fact that it looks nothing like the Arabic numeral 4. In this circumstance, to determine whether a new example is a 4, you may need to compare the example to your several _distinct_ archetypes of 4, such that if the example looks like _any_ of the numeral's archetypes, you identify the example with the numeral. This set of archetypes that you carry with you and modify over time through experience is what _anchors_ you to the notion of 4.

![4s]

In the following exploration, I will be developing an approach to machine learning that takes literally this idea of learning via repeated archetype comparison. We will begin with the simplified case of binary classification, i.e. distinguishing what something _is_ from what it _is not_. After that, we will generalize our analysis to multi-class classification, using the identification of hand-written digits as a case study.  

[Back to top](#Contents)

---

## Binary classification

To begin translating our discussion of archetypes into a mathematical language compatible with machine learning, let us consider a toy problem of distinguishing two classes of points on a sphere. In fact, let us make this more concrete by imagining that we have discovered an alien planet inhabited by two major civilizations. After some exploration, we learn that members of only one of these civilizations is friendly, while members of the other will attack any human they encounter. The belligerent species has spread out over most of the planet, but the friendly one has clustered around a few central bases where it can maintain control of the local resources. If humans are to live safely on this alien planet, we need to have a way of staying close to the areas predominantly occupied by the friendly civilization.

Thankfully, a team of explorers have surveyed the planet for us, marking green or orange dots on their map any time they encountered a member of the friendly or hostile civilization. Using the data gathered, can we shade the explorers' map according to how safe we estimate the region is for humans?

|Globe |Map Projection|
|:---:|:---:|
|![planet] | ![planet_flat]|

If we consider this measure of safety as "the probability that a certain location is safe," this map-shading task can be framed as a binary classification problem: safe vs. not safe. There are plenty of algorithms that we could employ to tackle this task (e.g. logistic regression, k-nearest-neighbors, support vector machines, neural networks, etc.), but let's see if we can approach the problem from a different perspective. We think that the friendly civilization has some central bases, so we might guess that those bases have the strongest defenses against the hostile civilization. As long as we are close to any one of these bases, no matter which base we choose, we will be safe. The farther away we get from any base, the weaker the defenses become, and the greater danger we place ourselves in.

Unfortunately, the explorers had a limited expedition, so we don't know for sure how many central bases there are or exactly where on the planet they are positioned. In the absence of such detailed information, we can start our mathematical inquiry by guessing that the friendly civilization has just _one_ central base. The question now becomes: how can we best estimate the location of that central base?

### Loss Function

To understand what kind of 

|Distance Product | Gaussian Kernel Density|
|:---:|:---:|
|![multiplicative_demo] | ![gaussian_demo]|


![multiplicative_pointy_demo]

![y_hat_binary]
![J_binary]
![blocks_binary]


We might expect that we can obtain reasonable results by computing the similarity between
Can we parametrize our data distribution using a relatively small number of anchor vectors?

---

Suppose we have a multi-class dataset
* pixel brightness

[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) is commonly used as a measure of similarity between two

Consider the standard problem of handwritten digit recognition.
* 28x28 square of grayscale pixels
* unroll this into a vector embedded in 784-dimensional brightness-space where each dimension corresponds to one pixel and the value of the vector in that dimension is the brightness of the pixel, scaled between 0 and 1.

???






### "Charge" property

### Negative anchors

The work we have discussed so far applies well when our data lives in a [simply connected space](https://en.wikipedia.org/wiki/Simply_connected_space) or a union of several such spaces; in other words, blobs of data are good, but holes in those blobs cause problems. Because our distance function increases as we move away from any particular anchor, the only way to

---

## Multi-class classification: MNIST

### Loss Function

![y_hat_multiclass]
![J_multiclass]
![blocks_multiclass]

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




[1-anchor]:images/2018-06-08/1-anchor.png
[4s]:images/4s.png
[multiplicative_demo]:images/distance_function_demo/multiplicative.png
[multiplicative_pointy_demo]:images/distance_function_demo/multiplicative_pointy.png
[gaussian_demo]:images/distance_function_demo/gaussian.png

[planet]:images/aliens/angle=020.png
[planet_flat]:images/aliens/flat.png

[y_hat_binary]:images/equations/y_hat_binary.png
[y_hat_multiclass]:images/equations/y_hat_multiclass.png  
[J_binary]:images/equations/J_binary.png
[J_multiclass]:images/equations/J_multiclass.png
[index_notation_binary]:images/equations/index_notation_binary.png
[index_notation_multiclass]:images/equations/index_notation_multiclass.png
[blocks_binary]:images/equations/blocks_binary.png
[blocks_multiclass]:images/equations/blocks_multiclass.png
