Assume

$$
y=\theta_0+\theta_1 x_1+\theta_2 x_2+\theta_3 x_3+\cdots+\theta_n x_n
$$

it equals to 
$$
\left[\begin{array}{cccccc}
\theta_0 x_0^{(1)} & \theta_1 x_1^{(1)} & \theta_2 x_2^{(1)} & \theta_3 x_3^{(1)} & \ldots & \theta_n x_n^{(1)} \\
\theta_0 x_0^{(2)} & \theta_1 x_1^{(2)} & \theta_2 x_2^{(2)} & \theta_3 x_3^{(2)} & \ldots & \theta_n x_n^{(2)} \\
\theta_0 x_0^{(3)} & \theta_1 x_1^{(3)} & \theta_2 x_2^{(3)} & \theta_3 x_3^{(3)} & \cdots & \theta_n x_n^{(3)} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
\theta_0 x_0^{(m)} & \theta_1 x_1^{(m)} & \theta_2 x_2^{(m)} & \theta_3 x_3^{(m)} & \cdots & \theta_n x_n^{(m)}
\end{array}\right]=\left[\begin{array}{c}
y^{(1)} \\
y^{(2)} \\
y^{(3)} \\
\vdots \\
y^{(m)}
\end{array}\right]
$$

we can say
$$
\mathrm{X} \Theta=\mathrm{y}
$$

we know $X_0=1$, then $\mathrm{X}, \Theta$ shoule be
$$
\mathrm{X}=\left[\begin{array}{cccccc}
1 & x_1^{(1)} & x_2^{(1)} & x_3^{(1)} & \cdots & x_n^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & x_3^{(2)} & \cdots & x_n^{(2)} \\
1 & x_1^{(3)} & x_2^{(3)} & x_3^{(3)} & \cdots & x_n^{(3)} \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
1 & x_1^{(m)} & x_2^{(m)} & x_3^{(m)} & \cdots & x_n^{(m)}
\end{array}\right], \Theta=\left[\begin{array}{c}
\theta_0 \\
\theta_1 \\
\theta_2 \\
\vdots \\
\theta_n
\end{array}\right]
$$

we know Cost Function $J\left(\theta_0 \ldots \theta_n\right)$ should be
$$
\begin{aligned}
J\left(\theta_0 \ldots \theta_n\right) & =\frac{1}{2 m} \sum_{i=1}\left(h_\theta\left(x_i^{(i)}\right)-y^{(i)}\right) \\
& =\frac{1}{2 m}(X \Theta-y)^T(X \Theta-y) \\
& =\frac{1}{2 m}\left[(X \Theta)^T-y^T\right](X \Theta-y) \\
& =\frac{1}{2 m}\left[(X \Theta)^T(X \Theta)-y^T X \Theta-(X \Theta)^T y+y^T y\right] \\
& =\frac{1}{2 m}\left[(X \Theta)^T(X \Theta)-2(X \Theta)^T y+y^T y\right]
\end{aligned}
$$

then

$$
\begin{aligned}
\frac{\partial}{\partial \Theta} J(\Theta)= & \frac{1}{2 m} \frac{\partial}{\partial \Theta}\left[(X \Theta)^T(X \Theta)-2(X \Theta)^T y+y^T y\right] \\
= & \frac{1}{2 m} \frac{\partial}{\partial \Theta}\left[(X \Theta)^T(X \Theta)\right]-\frac{1}{m} \frac{\partial}{\partial \Theta}\left((X \Theta)^T y\right) \\
= & \frac{1}{2 m} \frac{\partial}{\partial \Theta}\left(\Theta^T X^T X \Theta\right)-\frac{1}{m}\left(X^T y\right) \\
= & \frac{1}{2 m} X^T X \frac{\partial}{\partial \Theta}\left(\Theta^T \Theta\right)-\frac{1}{m}\left(X^T y\right) \\
= & \frac{1}{m} X^T X \Theta-\frac{1}{m}\left(X^T y\right)=0 \\
& X^T X \Theta=X^T y \\
& \left(X^T X\right)^{-1} X^T X \Theta=\left(X^T X\right)^{-1} X^T y
\end{aligned}
$$

at the end, we derive
$$
\Theta=\left(X^T X\right)^{-1} X^T y
$$