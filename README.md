# 1. Basic
We have a input signal on a uniform grid of $[L_0=N_0h, L=Nh]$. The signal is 
$$f(x) = \sum_{k=1}^{\infty} c_k x^{-a_k}$$

$$\inf (a_{k+1}-a_k) = d >0$$

Our purpose is to extract several $c_k, a_k$ from the input.

Further, we will try to solve
$$f(x) = \sum_{k=1}^n c_k x^{-a_k} + O(x^{-a_n-d})$$

$$a_n>...>a_1>0, ~a_{k+1}-a_k=d>0, c_k\in\mathbb{R}$$

# 2. Method

### 2.1 Leading Item
1. General Log-Log L2 fit
2. Shanks transformation
3. Wynn's epsilon algorithm

### 2.2 Multiple Leading Items
1. Integral Annihilator Peeling + Global-back fitting
2. The Scaling Derivative Method
3. Varpro
4. SVD-ESPRIT / Matrix Pencil
5. Mellin Transform + Analytic continuation

# 3. Result

### 1. Leading Item


$$f(x) = c_1x^{-a_1} + c_2x^{-a_2} + c_3x^{-a_3}$$

$$a_1 = 0.55, a_2 = 1.22, a_3 = 1.83, c1,c2,c3\approx 1$$

$$L0=1, L = 2^{14}, h = 0.1$$

In the following table, $n$ means the order of Wynn's epsilon algorithm, $k$ means the geometric sampling rate.


| Method | Order error | Coff error |
|--------|-------------|------------|
| LogLog | 0.008      | 0.08       |
| Shanks | 1.7e-5      | 3e-4       |
| Wynn (n=3, k=2)  | 1.7e-5      | 3e-4       |
| Wynn Pola (n=21, k=1.3, F64)| 2e-11      | 1.2e-7       |
| Wynn Pola (n=41, k=1.2, F128)| 3e-14      | 3e-10       |
| Order-locked Wynn Pola (n=21, k=1.3, F64)| 2e-11      | 3e-10       |
| Order-locked Wynn Pola (n=21, k=1.3, F128)| 5.6e-13      | 1.1e-11       |


# 4. API
```julia
orders, coffs, _ = power_solve(f, grid, norder; method = WynnPola())
```

Default method is `WynnPola(; k=1.3, n=21, use_a_final=true, nc=5)`. If your data is noisy in near-field, please make sure that for the choosen `k,n`, when $x>L/k^n$, the signal $f(x)$ is pure enough.

`f` is the input signal, `grid` is the grid, `norder` is the order of the power law, `method` is the method to use.

Only $n=1$ is supported for now.

The returned `orders` and `coffs` are arrays of the same length as `norder`. The last item in the return value is an additional supplementary note that may appear and is related to the `method`. If there is no additional content, the third item is `nothing`.

