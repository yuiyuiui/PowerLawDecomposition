## 1. Basic
We have a input signal on a uniform grid of $[L_0=N_0h, L=Nh]$. The signal is 
$$f(x) = c_1x^{-a} + c_2x^{-a-d}+O(x^{-a-d-d_0})$$

$$a,d,d_0 > 0, ~d\geq d_0, c_j\in \mathbb{R}$$

Our purpose is to extract $c_1, c_2, a, d$ from the input.

Further, we will try to solve
$$f(x) = \sum_{k=1}^n c_k x^{-a_k} + O(x^{-a_n-d_0})$$

$$a_n>...>a_1>0, ~a_{k+1}-a_k\geq d_0>0, c_k\in\mathbb{R}$$

## 2. Method

1. Common Prony
2. Find leading order with high accuracy and then find other orders
3. Direct LSQ to find orders with dichotomy

## 3. Result

#### 1. Leading Item


$$f(x) = c_1x^{-a_1} + c_2x^{-a_2} + c_3x^{-a_3} * sin(x^2)$$

$$a_1 = 0.55, a_2 = 1.22, a_3 = 1.83, c1,c2,c3\approx 1$$

$$L0=1, L = 2^{14}, h = 0.1$$

| Method | Order error | Coff error |
|--------|-------------|------------|
| LogLog | 0.0072      | 0.071      |
| Shanks | 1.4e-5      | 7e-4       |

