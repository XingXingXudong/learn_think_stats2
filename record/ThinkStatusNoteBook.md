# Think Stats, 2nd Edition

## Exploratory Data

**cross-sectional study**: 横断面研究
> Captures a snapshot of a group at a point in time.

**longitudinal study**: 纵向研究
> Observes a group repeatedly over a period of time.

# Distributions

**Distributions**
> One of the best ways to describe a variable is to report the values that appear in the dataset and how many times each value appears.
> 所谓变量的分布，即用来描述变量有哪些取值，以及这些取值出现的次数（密度）

**histogram**: 直方图
> The most common representation of a distribution is a **histogram**, whichis a graph that shows the **frequency** of each value. Here, the 'frequency' means the number of times the values appers.

**mode** 峰值
> The value that occurs most frequently in a given set of data.

### 一些差常用来描述分布的特征

> **central tendency**: 集中趋势 Do the values tend to cluster around a particular point 
> **modes**: 峰值 Is there more than one cluster?
> **spread** 松散 How much variability is there in the values?
> **tails** 尾 How quickly do the probabilities drop off as we move away from the modes?
> **outliers**: 异常值, Are there extreme values far from the modes?

# Probability Mass Functions

> **probability mass function(pmf)** is a function that gives the probability that a **discrete random variable** is exactly equal to some value. The probability mass function is often the primary means of defining a **discrete probability distributin**, and such functions exist for either **scalar** or **multivariate random variables** whose **domain** is dicrete.
> **probability density function(pdf)** is associated with continuous rather than dicrete random variable; the values of the pdf are not probabilities as such: a pdf must be integrated over an interval to yield a probability.

> Probability Mass Function 与直方图的：
>> Hist maps from values to integer **counters**;
>> Pmf maps from values to floating-point **probabilities**

