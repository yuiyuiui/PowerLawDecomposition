(Unfinished project)
# 1. Basic
We have a input signal on a uniform grid of $[L_0=N_0h, L=Nh]$. The signal is 
$$f(x) = \sum_{k=1}^{\infty} c_k x^{-a_k}$$

$$\inf (a_{k+1}-a_k) = d >0$$

Our purpose is to extract several $c_k, a_k$ from the input.

Further, we will try to solve
$$f(x) = \sum_{k=1}^n c_k x^{-a_k} + O(x^{-a_n-d})$$

$$a_n>...>a_1>0, ~a_{k+1}-a_k=d>0, c_k\in\mathbb{R}$$
