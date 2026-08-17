# Time-delay methods

## Sign convention

For component \(k\),

$$
t_{k,\mathrm{shifted}} = t_k - \Delta_k,
\qquad \Delta_{\mathrm{reference}} = 0.
$$

A positive $\Delta_B$ means B is observed later than the reference A. Pair
tables report $\Delta_j-\Delta_i$.

## Linear-interpolation baseline

`Utility/time_delay_interpolation.py` independently robust-normalises each
curve, propagates endpoint uncertainties through piecewise-linear
interpolation, evaluates only valid overlaps and fits an optional constant flux
offset.

The preserved score variants are:

1. raw A against interpolated B;
2. interpolated A against raw B;
3. both curves interpolated on a common grid;
4. the mean of the three scores.

A coarse grid is followed by bounded scalar refinement. This method is quick
and transparent, making it a useful baseline and diagnostic.

## Adaptive shared P-spline

The shared latent curve is represented by a cubic B-spline:

$$
f(t)=\sum_j \beta_j B_j(t).
$$

For shifted observations, the estimator minimises a weighted data term plus a
second-difference roughness penalty:

$$
\sum_{k,i} w_{k,i}
\left[y_{k,i}-f(t_{k,i}-\Delta_k)-c_k\right]^2
+
\sum_j \lambda_j(\Delta^2\beta_j)^2.
$$

Key adaptations retained from the research notebook:

- knots are placed at time quantiles;
- candidate spline sizes $K$ are selected by fast leave-one-out error;
- measurement weights combine `flux_obs_error` and, optionally, a rolling
  global MAD estimated across all shifted curves;
- local $\lambda_j$ values adapt to the rolling MAD and are clipped relative
  to a base smoothing level;
- delay search uses coordinate scans and bounded refinement;
- overlap constraints prevent solutions supported by too few points or too
  little temporal span;
- LOO, BIC-like and overlap-aware objective variants remain available.

## Uncertainty

Both modules use measurement-error Monte Carlo propagation:

$$
y_i^{(m)}=y_i+\epsilon_i^{(m)},\qquad
\epsilon_i^{(m)}\sim\mathcal{N}(0,\,
\texttt{error\_scale}^2\sigma_i^2).
$$

The complete estimator is rerun for every draw. This is not an MCMC posterior
sampler on the delay. `force_same_K_as_base=True` keeps the spline size selected
on the original data, substantially reducing computation while isolating the
effect of photometric error. Set it to `False` when uncertainty in model
complexity must also be propagated.