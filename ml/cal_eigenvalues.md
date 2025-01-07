Eigenvalues are a special set of scalars associated with a linear system of equations (i.e., a matrix equation) that are sometimes also known as characteristic roots, characteristic values (Hoffman and Kunze 1971), proper values, or latent roots (Marcus and Minc 1988, p. 144).

The determination of the eigenvalues and eigenvectors of a system is extremely important in physics and engineering, where it is equivalent to matrix diagonalization and arises in such common applications as stability analysis, the physics of rotating bodies, and small oscillations of vibrating systems, to name only a few. Each eigenvalue is paired with a corresponding so-called eigenvector (or, in general, a corresponding right eigenvector and a corresponding left eigenvector; there is no analogous distinction between left and right for eigenvalues).

The decomposition of a square matrix $\mathbf{A}$ into eigenvalues and eigenvectors is known in this work as eigen decomposition, and the fact that this decomposition is always possible as long as the matrix consisting of the eigenvectors of $A$ is square is known as the eigen decomposition theorem.

The Lanczos algorithm is an algorithm for computing the eigenvalues and eigenvectors for large symmetric sparse matrices.
Let $A$ be a linear transformation represented by a matrix $\mathbf{A}$. If there is a vector $\mathbf{X} \in \mathbb{R}^n \neq \mathbf{0}$ such that

$$
\mathbf{A} \mathbf{X}=\lambda \mathbf{X}
$$

for some scalar $\lambda$, then $\lambda$ is called the eigenvalue of A with corresponding (right) eigenvector $\mathbf{X}$.
Letting A be a $k \times k$ square matrix

$$
\left[\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 k} \\
a_{21} & a_{22} & \cdots & a_{2 k} \\
\vdots & \vdots & \ddots & \vdots \\
a_{k 1} & a_{k 2} & \cdots & a_{k k}
\end{array}\right]
$$

with eigenvalue $\lambda$, then the corresponding eigenvectors satisfy
$$
\left[\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 k} \\
a_{21} & a_{22} & \cdots & a_{2 k} \\
\vdots & \vdots & \ddots & \vdots \\
a_{k 1} & a_{k 2} & \cdots & a_{k k}
\end{array}\right]\left[\begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_k
\end{array}\right]=\lambda\left[\begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_k
\end{array}\right]
$$

which is equivalent to the homogeneous system

$$
\left[\begin{array}{cccc}
a_{11}-\lambda & a_{12} & \cdots & a_{1 k} \\
a_{21} & a_{22}-\lambda & \cdots & a_{2 k} \\
\vdots & \vdots & \ddots & \vdots \\
a_{k 1} & a_{k 2} & \cdots & a_{k k}-\lambda
\end{array}\right]\left[\begin{array}{c}
x_1 \\
x_2 \\
\vdots \\
x_k
\end{array}\right]=\left[\begin{array}{c}
0 \\
0 \\
\vdots \\
0
\end{array}\right]
$$

Equation (4) can be written compactly as

$$
(A-\lambda I) X=\mathbf{0}
$$

where I is the identity matrix. As shown in Cramer's rule, a linear system of equations has nontrivial solutions iff the determinant vanishes, so the solutions of equation (5) are given by

$$
\operatorname{det}(A-\lambda \mathbf{I})=0
$$

This equation is known as the characteristic equation of $A$, and the left-hand side is known as the characteristic polynomial.
For example, for a $2 \times 2$ matrix, the eigenvalues are

$$
\lambda_{ \pm}=\frac{1}{2}\left[\left(a_{11}+a_{22}\right) \pm \sqrt{4 a_{12} a_{21}+\left(a_{11}-a_{22}\right)^2}\right]
$$

which arises as the solutions of the characteristic equation

$$
x^2-x\left(a_{11}+a_{22}\right)+\left(a_{11} a_{22}-a_{12} a_{21}\right)=0
$$

If all $k$ eigenvalues are different, then plugging these back in gives $k-1$ independent equations for the $k$ components of each corresponding eigenvector, and the system is said to be nondegenerate. If the eigenvalues are $n$-fold degenerate, then the system is said to be degenerate and the eigenvectors are not linearly independent. In such cases, the additional constraint that the eigenvectors be orthogonal,

$$
\mathbf{X}_i \cdot \mathbf{X}_j=\left|\mathbf{X}_i\right|\left|\mathbf{X}_j\right| \delta_{i j}
$$

where $\delta_{i j}$ is the Kronecker delta, can be applied to yield $n$ additional constraints, thus allowing solution for the eigenvectors.
Eigenvalues may be computed in the Wolfram Language using Eigenvalues[matrix]. Eigenvectors and eigenvalues can be returned together using the command Eigensystem[matrix].

Assume we know the eigenvalue for

$$
\mathbf{A} \mathbf{X}=\lambda \mathbf{X}
$$

Adding a constant times the identity matrix to A ,

$$
(\mathbf{A}+c \mathbf{I}) \mathbf{X}=(\lambda+c) \mathbf{X} \equiv \lambda^{\prime} \mathbf{X}
$$

so the new eigenvalues equal the old plus $c$. Multiplying A by a constant $c$
$(c \mathbf{A}) \mathbf{X}=c(\lambda \mathbf{X}) \equiv \lambda^{\prime} \mathbf{X}$,
so the new eigenvalues are the old multiplied by $c$.
Now consider a similarity transformation of $A$. Let $|A|$ be the determinant of $A$, then

$$
\begin{aligned}
\left|Z^{-1} A Z-\lambda\right| \mid & =\left|Z^{-1}(A-\lambda I) Z\right| \\
& =|Z||A-\lambda I| | Z^{-1} \mid \\
& =|A-\lambda I|
\end{aligned}
$$

so the eigenvalues are the same as for $A$.

**Reference**:
[https://mathworld.wolfram.com/Eigenvalue.html#:~:text=Eigenvalues%20are%20a%20special%20set,Marcus%20and%20Minc%201988%2C%20p.](https://mathworld.wolfram.com/Eigenvalue.html#:~:text=Eigenvalues%20are%20a%20special%20set,Marcus%20and%20Minc%201988%2C%20p.)
