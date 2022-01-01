---
layout: post
title: Mandelbrot set
author: fkdosilovic
categories: [python, numpy, math, fractals, wip]
---

I've finally read the first part of Nicolas Rougier's [From Python to Numpy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/).
And, since I did not like terse exposition of Mandelbrot's set and fractal,
discussed in the [Temporal vectorization](https://www.labri.fr/perso/nrougier/from-python-to-numpy/#temporal-vectorization)
subsection, I've decided to expand on the subject a bit more.

This blog post is organized as follows: first section describes the Mandelbrot's set theory and required prerequisites

## Theory

- Mandelbrot's set
- Mandelbrot's fractal
- \\(z_{n + 1} = z_n^2 + c\\), where \\(z_0 = 0\\)
- sampling \\(\mathbb{C}\\)

## Implementation

### Simple Python implementation

```python
def is_divergent(c, radius=2, niters=200):
    z = 0 + 0j
    for _ in range(niters):
        z = z * z + c
        if abs(z) > radius:
            return True
    return False
```

### Uniform vectorization with NumPy

TODO

### Temporal vectorization with NumPy

TODO
