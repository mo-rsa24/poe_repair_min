"""Feynman-Kac steering (Singhal et al., arXiv 2501.06848) over the plain PoE sampler.

Not the Feynman-Kac correctors of Skreta et al. (arXiv 2503.02819), which change the score.
This package changes nothing in the score: K particles run the plain product-of-experts
sampler and are resampled on a reward read off the Tweedie estimate of the clean image.
"""
