The laplace expansion, named after Pierre-Simon Laplace, also called cofactor expansion, is an expression of the determinant of an $n \times n$ matrix $B$ as as weighted sum of minors, which are the determinants of some $(n-1) \times (n-1)$ submatrices of $B$. Specifically, for every $i$, the Laplace expansion along the $i$-th row is the equality

$$
\operatorname{det}(B)=\sum_{j=1}^n(-1)^{i+j} b_{i, j} m_{i, j},
$$

where $b_{i, j}$ is the entry of the $i$ th row and $j$ th column of $B$, and $m_{i, j}$ is the determinant of the submatrix obtained by removing the $i$ th row and the $j$ th column of $B$. Similarly, the Laplace expansion along the jth column is the equality

$$
\operatorname{det}(B)=\sum_{i=1}^n(-1)^{i+j} b_{i, j} m_{i, j}
$$

(Each identity implies the other, since the determinants of a matrix and its transpose are the same.)

The coefficient $(-1)^{i+j} m_{i, j}$ of $b_{i, j}$ in the above sum is called the cofactor of $b_{i, j}$ in $B$.
The Laplace expansion is often useful in proofs, as in, for example, allowing recursion on the size of matrices. It is also of didactic interest for its simplicity and as one of several ways to view and compute the determinant. For large matrices, it quickly becomes inefficient to compute when compared to Gaussian elimination.
