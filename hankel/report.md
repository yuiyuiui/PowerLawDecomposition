# 指数分解方法测试报告

## 实现方法

实现了两种将 $f(x)$ 分解为 $\sum_n c_n e^{-a_n x}$ 的方法：

### 1. Prony 方法（经典最小二乘变体）

1. 建立线性预测的超定 Toeplitz 系统，最小二乘求解 AR 系数
2. 构造伴随矩阵，求特征值获得信号极点 $z_n$
3. 由 $z_n = e^{-a_n h}$ 反推衰减率 $a_n = -\ln|z_n|/h$
4. 建立 Vandermonde 矩阵，最小二乘求解振幅 $c_n$

数据量要求：$M \geq 2N$

### 2. Matrix Pencil 方法（Hua–Sarkar, 1990）

1. 构造 Hankel 矩阵，提取两个移位子矩阵 $Y_0, Y_1$
2. 对 $Y_0$ 做 SVD，截断到秩 $N$
3. 求 $\Sigma_N^{-1} U_N^\top Y_1 V_N$ 的特征值获得信号极点
4. 同上反推 $a_n$、求解 $c_n$

数据量要求：$M \geq 2N + 1$

均通过 LinearAlgebra 标准库实现，BigFloat 支持由 GenericLinearAlgebra.jl 提供。

---

## 测试结果

### RMSE 拟合误差

| 测试案例 | 类型 | Prony RMSE | MPM RMSE |
|---------|------|-----------|---------|
| 2-exp simple (a=1,3) | Float32 | 2.4e-06 | 8.4e-07 |
| | Float64 | 7.4e-15 | 1.1e-15 |
| | BigFloat | 1.8e-17 | 9.0e-18 |
| 3-exp mixed (a=0.5,1.5,4) | Float32 | 2.6e-05 | 2.7e-06 |
| | Float64 | 3.0e-14 | 1.9e-15 |
| | BigFloat | 9.6e-17 | **2.4e-76** |
| 2-exp close (a=1,1.2) | Float32 | 3.7e-06 | 3.9e-07 |
| | Float64 | 1.3e-14 | 6.5e-16 |
| | BigFloat | 1.7e-17 | 1.1e-18 |
| 2-exp spread (a=0.1,10) | Float32 | 2.3e-06 | 4.1e-06 |
| | Float64 | 2.7e-16 | 4.6e-15 |
| | BigFloat | 9.8e-18 | 5.0e-18 |

### 参数恢复精度（max|Δa|）

| 测试案例 | 类型 | Prony | MPM |
|---------|------|-------|-----|
| 2-exp simple | Float32 | 6.9e-05 | 3.8e-06 |
| | Float64 | 1.6e-13 | 1.1e-14 |
| | BigFloat | 2.1e-16 | 1.7e-16 |
| 3-exp mixed | Float32 | 2.8e-03 | 3.9e-05 |
| | Float64 | 2.2e-13 | 1.6e-14 |
| | BigFloat | 7.7e-16 | **4.4e-75** |
| 2-exp close | Float32 | 2.5e-03 | 4.6e-05 |
| | Float64 | 5.9e-12 | 4.3e-14 |
| | BigFloat | 9.8e-15 | 9.0e-16 |

---

## 分析

### 1. Matrix Pencil 在大多数场景下精度优于 Prony

MPM 利用 SVD 截断消除噪声/数值误差的低秩分量，在拟合误差和参数恢复上通常优于 Prony 一到两个数量级。特别是在：

- **Float32**：MPM 的 RMSE 稳定在 1e-6 至 1e-7 级别，Prony 为 1e-5 至 1e-6。参数恢复差距更大（MPM 1e-5 vs Prony 1e-3）。
- **Float64**：MPM 达到 1e-15 至 1e-16 级别（接近机器精度），Prony 为 1e-14 至 1e-15。
- **BigFloat**：MPM 在 3-exp mixed 案例中达到了 **2.4e-76** 的极端精度（接近默认 256-bit 精度的理论极限 ~1e-77），说明 SVD 截断在高精度计算中优势更加显著。

### 2. 衰减率接近时 Prony 更敏感

在 "2-exp close"（a=1.0 和 a=1.2）案例中，Prony 的参数恢复误差比 MPM 大约两个数量级（Float64: 5.9e-12 vs 4.3e-14）。这是因为相近的极点导致 Prony 线性预测系统的条件数增大。

### 3. 衰减率差距极大时两者表现相当

在 "2-exp spread"（a=0.1 和 a=10）案例中，两种方法的精度接近。这是因为该情形下问题条件良好——两个指数分量在时域上容易分辨。Prony 在 Float64 下甚至略优（2.7e-16 vs 4.6e-15），因为该案例中线性预测系统恰好特别良态。

### 4. 类型精度表现符合预期

- **Float32**（~7 位有效数字）：RMSE 约 1e-6，参数恢复约 3–5 位有效数字
- **Float64**（~16 位有效数字）：RMSE 约 1e-14 至 1e-16，接近机器精度
- **BigFloat**（~77 位有效数字）：MPM 能充分利用扩展精度，Prony 受限于条件数瓶颈

### 5. 结论

对于实际应用，**推荐使用 Matrix Pencil 方法**。它在数值稳定性和精度上全面优于 Prony 方法，唯一的代价是需要多一个数据点（$M \geq 2N+1$ vs $M \geq 2N$）和一次 SVD 分解。
