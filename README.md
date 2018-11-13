# Anchor vector optimization

## Contents

* [Introduction](#Introduction)
* [Binary Classification](#binary-classification)
  * [Distance Function](#distance-function)
  * [Building a Predictive Model](#building-a-predictive-model)
  * [Learning](#learning)
* [Multiclass Classification](#multiclass-classification:-mnistgit )



---
## Introduction

Suppose that you were not familiar with the Arabic numeral system. To teach it to you, I might neatly write an example of each of the digits 0 through 9, then ask you to identify a new example of one of the digits. If my handwriting stayed relatively consistent, you would probably be able to answer reasonably well simply by observing which of my 10 archetypal examples this new example looks most similar to. You might find this task challenging, however, if I instead messily scribbled the new example. In circumstances like this, after you make a guess, I would tell you what the numeral really was, and you would update the notion in your mind of what that numeral generally looks like. As we repeat this process, you would gradually change your understanding of what it means for an example to _look like_ each particular digit. You would change your mental archetypes to match your experiences and compare each new example against those new archetypes.  

Sometimes, your experience necessitates that you expand your collection of archetypes to encapsulate dramatically different examples of the same idea. Maybe you've only ever seen the numeral 4 written with an open top, and you need to update your understanding to recognize that a 4 with a closed top is also a 4. Maybe you have to learn that the Chinese character for 4 also means 4, despite the fact that it looks nothing like the Arabic numeral 4. In this circumstance, to determine whether a new example is a 4, you may need to compare the example to your several _distinct_ archetypes of 4, such that if the example looks like _any_ of the numeral's archetypes, you identify the example with the numeral. This set of archetypes that you carry with you and modify over time through experience is what **_anchors_** you to the notion of 4.

![4s]

In the following exploration, I will be developing an approach to machine learning that takes literally this idea of learning via repeated archetype comparison. We will begin with the simplified case of binary classification, i.e. distinguishing what something _is_ from what it _is not_. After that, we will generalize our analysis to multi-class classification, using the identification of hand-written digits as a case study.  

[Back to top](#Contents)

---

## Binary Classification

To begin translating our discussion of archetypes into a mathematical language compatible with machine learning, let us consider a toy problem of distinguishing two classes of points on a sphere. In fact, let us make this more concrete by imagining that we have discovered an alien planet inhabited by two major civilizations. After some exploration, we learn that members of only one of these civilizations is friendly, while members of the other will attack any human they encounter. The belligerent species has spread out over most of the planet, but the friendly one has clustered around a few central bases where it can maintain control of the local resources. If humans are to live safely on this alien planet, we need to have a way of staying close to the areas predominantly occupied by the friendly civilization.

Thankfully, a team of explorers have surveyed the planet for us, marking green or orange dots on their map any time they encountered a member of the friendly or hostile civilization. Using the data gathered, how can we shade the explorers' map according to where it is or isn't safe for a human to be?

|Globe |Map Projection|
|:---:|:---:|
|![planet] | ![planet_flat]|

This map-shading task can be framed as a probabilistic binary classification problem if we consider each spot on the globe as having some probability between 0 and 1 of being _safe_. There are plenty of algorithms that we could employ to tackle this problem (e.g. k-nearest-neighbors, support vector machines, neural networks, etc.), but let's see if we can approach the problem from a different perspective. We think that the friendly civilization has some central bases, so we might guess that those bases have the strongest defenses against the hostile civilization. As long as we are close to any one of these bases, we can reason, no matter which base we choose, we will be safe. The farther away we get from any base, the weaker the defenses become, and the greater danger we place ourselves in.

Unfortunately, the explorers' expedition was limited, so we don't know for sure how many central bases there are or exactly where on the planet they are positioned. In the absence of such detailed information, we can start our mathematical inquiry by guessing that the friendly civilization has just fixed number (e.g. 4) of central bases. The question now becomes: how can we best estimate the location of each of those central bases?

[Back to top](#Contents)

### Distance Function

From here on, I will assume that the reader has a basic knowledge of linear algebra and some multivariable calculus. Since all our pieces of data correspond to locations on the surface of the alien planet, the data can be represented as a collection of vectors pointing from the origin to the surface of a sphere centered on that origin. Choosing the location of a central base location is equivalent to choosing a vector pointing from the origin in some new direction. We'll call this an **_anchor vector_**.

The measure known as [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) provides us a good way of representing how far apart two locations on our planet are from each other. This similarity measure computes the cosine of the angle between the two vectors representing our locations, such that if the vectors exactly coincide, the cosine similarity equals 1, and if they point in opposite directions, the cosine similarity equals -1. To turn this into a distance measure, we can simply take 1 minus the cosine similarity so that the closest and farthest that two vectors can be from each other are a distance of 0 and 2, respectively. (**Note**: Cosine similarity is well-defined for vector pairs of any magnitude and in any number of dimensions, but since all our data lies on a sphere, there happens to be a one-to-one correspondence between cosine distance and [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance). Though this correspondence allows cosine distance to act as a true metric of our spherical space, in general, cosine distance cannot be considered a _true_ [distance metric](https://en.wikipedia.org/wiki/Metric_(mathematics) since it does not obey the triangle inequality. This is not important for the current discussion, but it's good to know.)

Cosine distance can tell us how far away we are from any single base on the alien planet, but to understand how far we are from _safety_, we need to create a new notion of distance that somehow represents how far away we are from all of the bases together. Remember that if we treat each base equally, we just need to be near any one of the bases to be safe. One simple way to encode this idea is to represent our effective distance from safety as the _minimum distance_ between us and each of the bases. However, if two bases are near to each other, we might guess that being close to both of them is safer than being close to a single isolated base. To account for the contributions of all the bases, then, we can take the _product of distances_ between us and each base. Notice that since the cosine distance goes to 0 as two vectors approach each other, as long as we are near any base, the product of distances will also be lose to 0. Furthermore, if we're near multiple bases, the product of distances will be even smaller. Below I've plotted an example of the distance product function in action, along with a version with the contour lines pushed down away from the tips to emphasize visually what's going on.

|Distance Product | Distance Product (Pointy)|
|:---:|:---:|
|![multiplicative_demo] | ![multiplicative_pointy_demo]|

For reference, below are visualizations of two alternate distance functions: the minimum distance function (which we discussed earlier) and a Gaussian Kernel Density estimation. The latter simply places a Gaussian blip at each anchor, which has the effect of capturing the density of anchors distributed throughout a region. While this technique is commonly used to estimate data distributions, it does not satisfy the property we desire that has the distance be _minimal_ at every anchor. Furthermore, it suffers from the fact that before constructing the density estimation, the width of the Gaussian blip must be chosen (if it's too big, not enough variation is captured, and if it's too small, the function is just a collection of equally shaped spikes).

| Minimum Distance| Gaussian Kernel Density|
|:---:|:---:|
| ![min_dist_demo]| ![gaussian_demo]|

Now that we have a distance function, generating a "probability" function is straightforward. If we treat the distance measure as some kind of [log-odds](https://en.wikipedia.org/wiki/Logit) quantity, or at least a linear transformation of log-odds, we can convert it into a probability by applying the sigmoid function. The [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) squishes the real number line into the space between 0 and 1, so it will ensure that our output behaves like a probability (which must be non-negative and no greater than 1). Readers familiar with [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) will recognize this move of treating our data as log-odds, since that is exactly the tactic logistic regression uses to apply the method of [linear regression](https://en.wikipedia.org/wiki/Linear_regression) to the problem of binary classification.

[Back to top](#Contents)

### Building a Predictive Model

With the distance function in place, we can now write out an expression to represent the probability ("_y-hat_") of being safe in a certain location on the planet, given a collection of P home bases:

![y_hat_binary]

On the first line, we have the definition of our probability prediction _y-hat_. The index _i_ on _y-hat_ indicates that this prediction refers to a single example location on the planet. The variables _b_ and _w_ are unknown parameters that linearly transform the distance measure _D_ into log-odds. The function _σ_ is the sigmoid function which transforms log-odds into probabilities, as indicated in the second line. The distance measure _D_ is a product of distances _d_ between the example location **_x_** and each of the anchors **_a_** (indexed by a small _p_). The last line shows the formula for cosine distance between two vectors **_a_** and **_x_**. (**Note**: Technically, since our data lies on the full sphere, we should divide the cosine distance by 2 to make sure that each pairwise distance is scaled between 0 and 1. This has the advantage that the distance product is also forced to stay scaled between 0 and 1. But this detail isn't really that important since the values of _D_ in each case end up just being off by a constant multiple of 2, which can be absorbed into the _w_ parameter.)

With the problem statement transformed into this mathematical model, we now need to estimate the unknown parameters _b_, _w_, and the vectors **_a<sub>p</sub>_** which give us the most accurate predictions (_y-hat_) given the data we were provided and a choice of number of anchors (_P_). To do this, we must first create a _cost function_ or [_loss function_](https://en.wikipedia.org/wiki/Loss_function) which penalizes our model when it fails to accurately predict whether a spot on the planet is inhabited by friendly or hostile aliens. We must then choose the model parameter values which _minimize_ that cost.

![J_binary]

In the set of equations above, I've represented the total model cost _J_ as a sum of average prediction loss _L_ (i.e. how badly does **_y-hat_** predict **_y_** for each of our _m_ training examples?) and regularization cost _J<sub>reg</sub>_. The equation for _L_ depicts the same [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) loss function that is used in logistic regression. To understand intuitively why this is a good loss function for our model output, we can plug in some numbers close to 0 and 1. If an example is truly labeled 0 (_y_ = 0), but our model predicts a value close to 1 (_y-hat_ = 0.99), the first term is 0, and the second term is large [ln(1 - 0.99) ~ -4.6]. If, on the other hand, our model predicts a value close to 0 (_y-hat_ = 0.01) for that example, the second term is small [ln(1 - 0.01) ~ -0.01]. A similar observation can be made when the true label is 1. Thus, when the ground truth and the model prediction agree, the cost is low, and when they disagree, the cost is high.

Typically, when the number of model parameters becomes large, we like to introduce [_regularization_](https://en.wikipedia.org/wiki/Regularization_(mathematics)) to prevent [overfitting](https://en.wikipedia.org/wiki/Overfitting). This regularization is coupled with "hyperparameters", whose values must be chosen to determine how much the regularization conditions should dominate the overall cost function. For logistic regression, we can introduce "L2 regularization" to penalize _w_ when it becomes large. However, since we just have a single _w_ parameter, such penalization is unnecessary, so I will set the regularization strength _λ_ to 0.

The second term in _J<sub>reg</sub>_ is unique to this problem of anchor vectors, and its purpose is not exactly to reduce overfitting. Inspired by physics, this condition penalizes anchor vectors from being close to each other in the same way that equal electric charges are discouraged from being near each other. The electrostatic energy of two charges, each with charge _q_, is proportional to the product of their charges and inversely proportional to the distance between them. When a system of charged particles moves, it seeks to find the configuration of particles which minimizes the system's energy. Here, we include a **"charge regularization"** to push anchor vectors apart so that they try not to learn the same thing. The main difference between the physics formula and this regularization condition is that I use cosine similarity, here, to represent the distance between anchors rather than Euclidean distance.

To understand the notation I used in the above formulas, it may be useful to see a visual representation of each of the vectorized objects referenced. Below, I've illustrated the objects **X**, **A**, and **y** as blocks of values. **X** is a _m_-by-_n_ matrix of data points with one example vector in each row (_m_ examples with _n_ features per example). The vector **y** is a collection of _m_ example labels which correspond to each of the _m_ example feature vectors. **A** is a _P_-by-_n_ matrix of anchor vectors, where each row is one anchor vector with _n_ features.    

![blocks_binary]

[Back to top](#Contents)

### Learning

What is really going on when the model learns its parameter values?

When trained on labeled examples, the anchor vector binary classification model tries to find optimal positions for its vectors that best represent the distribution of positive labels, subject to the constraint that the anchors shouldn't be very close together. The logistic regression portion of the model tries to find the best scaling of the distance function outputs which separates the two classes into regions of high probability and low probability for positive and negative examples, respectively.

Thinking back to the distance function contour plots shown [above](#distance-function), we can imagine the training procedure as a two-step process. In one step, the _shape_ of every 2D contour line is determined by the XY positions of all the anchors. In the other step, the contour lines are shifted up or down the Z dimension _without changing their shape_ so that the redder/higher the contour's Z height, the higher the probability it represents, and the lower/bluer the height, the lower the probability it represents. The center-most contour in the Z dimension is the typical {probability = 0.5} decision boundary used to separate the positive from the negative class.

In terms of the alien planet, our model searches for the location of home bases which best correspond with the observed scattering of friendly and hostile aliens across the planet, informed by the additional piece of intuition that two home bases are not likely to be very close together.

With this theory in place, we can now implement the model and cost function relatively easily in Google's Python-based machine learning framework [TensorFlow](https://www.tensorflow.org/), and see what happens when we choose various hyperparameters (such as the number of anchors or the charge of each anchor). When we set up our model as a computation graph in TensorFlow, the partial derivatives of the cost function with respect to every parameter are automatically computed through [backpropagation](https://en.wikipedia.org/wiki/Backpropagation). This simplifies the model training process for us, since with a cost function and parameter partial derivatives we can use any pre-written optimization function (I've chosen the [Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)) to search for the optimal parameters. In my analysis, I've performed an 80%/20% train/test split. [**Note:** In general, this optimization procedure is non-convex, so training may get stuck in local optima in our parameter space. In the current study, I have decided not to do a thorough complexity analysis since I do not want to compute the derivatives by hand right now.]  

#### 1 anchor

The simplest choice would be to guess that there is just a single home base. To train a single-anchor model, we initialize the anchor at a random position on the globe, and give the sigmoid weights random values. With each training iteration, the model moves the anchor to better fit the data distribution. Below, I've plotted a heat map of the final prediction distribution, where the yellow "X" marks the anchor position and the redder the heat map color, the safer the region. I have also plotted the error of the training and testing set over the course of the 5000-epoch model training period.


| Learned Color Map| Training and Testing Error Over Time|
|:---:|:---:|
|![binary-p01-q0-heatmap]| ![binary-p01-q0-cost]|

Since this is a binary classification problem, we can use [ROC curves](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) to measure the performance of the model. These plots show how the true positive rate (true positive / all positive) and false positive rate (false positive / all negative) change as we shift the probability threshold for determining what is and is not considered a "positive prediction". A completely non-discriminative model would like on the diagonal line connecting TPR = FPR = 0 to TPR = FPR = 1. A perfectly discriminative model woudl pull the ROC curve all the way up to the TPR = 1, FPR = 0 point. From the ROC curves below, plotted for both the training set and the test set, we can see that a single-anchor model isn't terrible, but it could be better.

| Performance on Training Set| Performance on Test Set|
|:---:|:---:|
| ![binary-p01-q0-ROC_train]|![binary-p01-q0-ROC_test]|

#### 2 anchors

To improve the classification performance, we can try again with two anchors instead of just one. For now, let's ignore anchor "charge" and just consider the anchors as non-interacting. Immediately, with just one more anchor, we can see dramatic improvement in classification accuracy. There is a clear visible separation between the green and orange dots as traced out by the white 50% probability threshold decision boundary.

| Learned Color Map| Training and Testing Error Over Time|
|:---:|:---:|
|![binary-p02-q0-heatmap]| ![binary-p02-q0-cost]|

When we plot ROC curves, we can confirm that the classification performance has dramatically improved from the single anchor case:

| Performance on Training Set| Performance on Test Set|
|:---:|:---:|
| ![binary-p02-q0-ROC_train]|![binary-p02-q0-ROC_test]|

We can peer into the black box of our model training procedure by rendering a heat map for the parameter states at intermediate points throughout the training. For this two-anchor case, it looks like the anchors first converged quickly from their random initializations to the same position around epoch 10. After jittering around together for some time, the two anchors separated around epoch 80, heading toward their final resting locations, which they settled on around epoch 400:

<center>

![binary-p02-q0-training]

</center>

#### 3 and 4 anchors

With 3 anchors, we again see improved performance:

| Learned Color Map| Training and Testing Error Over Time|
|:---:|:---:|
|![binary-p03-q0-heatmap]| ![binary-p03-q0-cost]|

| Performance on Training Set| Performance on Test Set|
|:---:|:---:|
| ![binary-p03-q0-ROC_train]|![binary-p03-q0-ROC_test]|

The 4-anchor model approaches limit at which this procedure can perform before we start seeing problems related to overfitting. To combat some crowding issues, I introduced a small amount of charge to these anchors:

| Learned Color Map| Training and Testing Error Over Time|
|:---:|:---:|
|![binary-p04-q0.001-heatmap]| ![binary-p04-q0.001-cost]|

| Performance on Training Set| Performance on Test Set|
|:---:|:---:|
| ![binary-p04-q0.001-ROC_train]|![binary-p04-q0.001-ROC_test]|

By watching the parameters change during training, we can see how the uncharged 3-anchor case differs from the charged 4-anchor case. Notice that, like in the 2-anchor case, the 3 anchors begin by quickly moving from their random initializations to the same location, then separate again to find their final resting locations. In the 4-anchor case, on the other hand, we see the anchors approach the same location, but never quite touch because of their mutual repulsion. As they get pushed back apart again, we see how this repulsion informs their movement until they reach an equilibrium:

<center>

| 3 anchors (_q_ = 0) | 4 anchors (_q_ = 0.001)|
|:---:|:---:|
|![binary-p03-q0-training]| ![binary-p04-q0.001-training]|

</center>
---

#### 10 anchors

To better investigate the effect of anchor charge on the learning process, let us know examine the extreme case of 10 anchors. Since the data distribution has only 3 or 4 visible green clusters, 10 anchors is obviously too many. Rather than overfitting to the training set, however, this excessive number just appears to weaken the overall model performance.

With no charge, the anchors have no option other than to merge and try their best to represent the data as one heavy anchor. Adding a small amount of charge (_q_ = 0.001) results in similar behavior, just with the anchors resting a small distance apart.

Setting _q_ = 0.01 yields an interesting change in behavior of the system. Unlike in all previous cases, when the anchor charge exceeds a certain value, the model learns to have the anchors represent the _negative_ class instead of the positive class. Initially, the anchors learn to spread out as far as possible across the globe, but when transitioning from epoch 7 to epoch 8, the anchors flip sign to reduce the training cost in a manner almost akin to a phase transition. The anchors remain mostly evenly distributed in this local minimum until they learn that they can minimize the error better by clustering together toward the bottom of the globe around epoch 80. For the next few thousand epochs, 9 of the anchors wander slowly around the south pole while one straggler ventures upward to capture a particularly dense region of orange points. At epoch 3,740, the anchor configuration snaps into its final configuration.

The _q_ = 0.1 scenario yields a similar result to the _q_ = 0.01 case, but because the charge is now so large, the anchors have a harder time pushing out of the state that has them evenly distributed across the globe. For the first 3,000 epochs, we see the anchors fidget collectively between a few equivalent configurations. For the next roughly 1,000 epochs, the anchors remain steady as the sigmoid parameters shift the probability distribution making the landscape bluer and bluer until the anchors finally snap to their final steady configuration.  

<center>

| Charge | Training Visualization | Training and Testing Error Over Time |
|:---:|:---:|:---:|
|_q_ = 0|![binary-p10-q0-training]|<img src="images/anchor_training/p=10,%20q=0,%20alpha=0.01/cost.png" width="350"/>|
|_q_ = 0.001|![binary-p10-q0.001-training]|<img src="images/anchor_training/p=10,%20q=0.001,%20alpha=0.01/cost.png" width="350"/>|
|_q_ = 0.01|![binary-p10-q0.01-training]|<img src="images/anchor_training/p=10,%20q=0.01,%20alpha=0.01/cost.png" width="350"/>|
|_q_ = 0.1|![binary-p10-q0.1-training]|<img src="images/anchor_training/p=10,%20q=0.1,%20alpha=0.01/cost.png" width="350"/>|

</center>
---

[Back to top](#Contents)

---

## Multi-class Classification: MNIST

Now that we've tried out the learning procedure on an easily-visualizable toy binary model, we are ready to generalize anchor-vector learning to the _multiclass classification_ problem. To examine this concretely, we will consider the classic problem of handwritten digit recognition using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Here we tackle directly the question I referenced in the introduction to this investigation: can we create a machine learning model which reflects our intuitive understanding of learning concepts through the refinement of archetypal ideals?

Suppose we have a multi-class dataset
* pixel brightness


Consider the standard problem of handwritten digit recognition.
* 28x28 square of grayscale pixels
* unroll this into a vector embedded in 784-dimensional brightness-space where each dimension corresponds to one pixel and the value of the vector in that dimension is the brightness of the pixel, scaled between 0 and 1.

???


### Distance Function

![y_hat_multiclass]
![J_multiclass]
![blocks_multiclass]
![index_notation_multiclass]
### Performance

### Comparison with simple Neural Network

### Visualizing learned decision boundaries with UMAP

---

## The Future

Hierarchical classification with multi-label classification. Change the softmax output into a collection of sigmoids with corresponding cross entropy losses.

Specify different number of anchors per class.

Negative anchors: The work we have discussed so far applies well when our data lives in a [simply connected space](https://en.wikipedia.org/wiki/Simply_connected_space) or a union of several such spaces; in other words, blobs of data are good, but holes in those blobs cause problems. Because our distance function increases as we move away from any particular anchor, the only way to

Grant a trainable weight to each anchor (to make some more important than others), and take a weighted geometric mean of cosine distances rather than a basic product of distances.

Dynamically add or remove anchors during training with a kind of drop-out regularization.

---

## Conclusion

* Advantages
  * straightforward interpretation
  * intuitive initialization
  * better performance than vanilla NN with same number of parameters? [WRONG]

---



[4s]:images/4s.png
[multiplicative_demo]:images/distance_function_demo/multiplicative.png
[multiplicative_pointy_demo]:images/distance_function_demo/multiplicative_pointy.png
[gaussian_demo]:images/distance_function_demo/gaussian.png
[min_dist_demo]:images/distance_function_demo/min_dist.png

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

[binary-p01-q0-heatmap]:images/anchor_training/p=01,%20q=0,%20alpha=0.01/learned_heatmap.png
[binary-p01-q0-cost]:images/anchor_training/p=01,%20q=0,%20alpha=0.01/cost.png
[binary-p01-q0-ROC_test]:images/anchor_training/p=01,%20q=0,%20alpha=0.01/ROC_test.png
[binary-p01-q0-ROC_train]:images/anchor_training/p=01,%20q=0,%20alpha=0.01/ROC_train.png

[binary-p02-q0-heatmap]:images/anchor_training/p=02,%20q=0,%20alpha=0.01/learned_heatmap.png
[binary-p02-q0-cost]:images/anchor_training/p=02,%20q=0,%20alpha=0.01/cost.png
[binary-p02-q0-ROC_test]:images/anchor_training/p=02,%20q=0,%20alpha=0.01/ROC_test.png
[binary-p02-q0-ROC_train]:images/anchor_training/p=02,%20q=0,%20alpha=0.01/ROC_train.png
[binary-p02-q0-training]:images/anchor_training/p=02,%20q=0,%20alpha=0.01/training_cropped.gif

[binary-p03-q0-heatmap]:images/anchor_training/p=03,%20q=0,%20alpha=0.01/learned_heatmap.png
[binary-p03-q0-cost]:images/anchor_training/p=03,%20q=0,%20alpha=0.01/cost.png
[binary-p03-q0-ROC_test]:images/anchor_training/p=03,%20q=0,%20alpha=0.01/ROC_test.png
[binary-p03-q0-ROC_train]:images/anchor_training/p=03,%20q=0,%20alpha=0.01/ROC_train.png
[binary-p03-q0-training]:images/anchor_training/p=03,%20q=0,%20alpha=0.01/training_cropped.gif

[binary-p04-q0.001-heatmap]:images/anchor_training/p=04,%20q=0.001,%20alpha=0.01/learned_heatmap.png
[binary-p04-q0.001-cost]:images/anchor_training/p=04,%20q=0.001,%20alpha=0.01/cost.png
[binary-p04-q0.001-ROC_test]:images/anchor_training/p=04,%20q=0.001,%20alpha=0.01/ROC_test.png
[binary-p04-q0.001-ROC_train]:images/anchor_training/p=04,%20q=0.001,%20alpha=0.01/ROC_train.png
[binary-p04-q0.001-training]:images/anchor_training/p=04,%20q=0.001,%20alpha=0.01/training_cropped.gif

[binary-p10-q0.1-training]:images/anchor_training/p=10,%20q=0.1,%20alpha=0.01/training_cropped.gif
[binary-p10-q0.01-training]:images/anchor_training/p=10,%20q=0.01,%20alpha=0.01/training_cropped.gif
[binary-p10-q0.001-training]:images/anchor_training/p=10,%20q=0.001,%20alpha=0.01/training_cropped.gif
[binary-p10-q0-training]:images/anchor_training/p=10,%20q=0,%20alpha=0.01/training_cropped.gif

[binary-p10-q0.1-cost]:images/anchor_training/p=10,%20q=0.1,%20alpha=0.01/cost.png
[binary-p10-q0.01-cost]:images/anchor_training/p=10,%20q=0.01,%20alpha=0.01/cost.png
[binary-p10-q0.001-cost]:images/anchor_training/p=10,%20q=0.001,%20alpha=0.01/cost.png
[binary-p10-q0-cost]:images/anchor_training/p=10,%20q=0,%20alpha=0.01/cost.png
