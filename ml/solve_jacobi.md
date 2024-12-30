**proof**:

In numerical linear algebra, the Jacobi method (a.k.a. the Jacobi iteration method) is an iterative algorithm for determining the solutions of a strictly diagonally dominant system of linear equations. Each diagonal element is solved for, and an approximate value is plugged in.

Let $A \mathbf{x}=\mathbf{b}$ be a square system of $n$ linear equations, where:

$$
A=\left[\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 n} \\
a_{21} & a_{22} & \cdots & a_{2 n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n 1} & a_{n 2} & \cdots & a_{n n}
\end{array}\right], \quad \mathbf{x}=\left[\begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{array}\right], \quad \mathbf{b}=\left[\begin{array}{c}
b_1 \\
b_2 \\
\vdots \\
b_n
\end{array}\right] .
$$

When $A$ and $\mathbf{b}$ are known, and $\mathbf{x}$ is unknown, we can use the Jacobi method to approximate $\mathbf{x}$. The vector $\mathbf{x}^{(0)}$ denotes our initial guess for $\mathbf{x}$ often $\mathbf{x}_i^{(0)}=0$ for $i=1,2, \ldots, n$. We denote $\mathbf{x}^{(k)}$ as the $k$-th approximation or iteration of $\mathbf{x}$, and $\mathbf{x}^{(k+1)}$ is the next (or $k+1$ ) iteration of $\mathbf{x}$.

**Matrix-based formula**
Then $A$ can be decomposed into a diagonal component $D$, a lower triangular part $L$ and an upper triangular part $U$ :

$$
A=D+L+U \quad \text { where } \quad D=\left[\begin{array}{cccc}
a_{11} & 0 & \cdots & 0 \\
0 & a_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & a_{n n}
\end{array}\right] \text { and } L+U=\left[\begin{array}{cccc}
0 & a_{12} & \cdots & a_{1 n} \\
a_{21} & 0 & \cdots & a_{2 n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n 1} & a_{n 2} & \cdots & 0
\end{array}\right] \text {. }
$$

The solution is then obtained iteratively via

$$
\mathbf{x}^{(k+1)}=D^{-1}\left(\mathbf{b}-(L+U) \mathbf{x}^{(k)}\right) .
$$

**Element-based formula**
The element-based formula for each row $i$ is thus:

$$
x_i^{(k+1)}=\frac{1}{a_{i i}}\left(b_i-\sum_{j \neq i} a_{i j} x_j^{(k)}\right), \quad i=1,2, \ldots, n
$$

The computation of $x_i^{(k+1)}$ requires each element in $\mathbf{x}^{(k)}$ except itself. Unlike the Gauss-Seidel method, we can't overwrite $x_i^{(k)}$ with $x_i^{(k+1)}$, as that value will be needed by the rest of the computation. The minimum amount of storage is two vectors of size $n$.
