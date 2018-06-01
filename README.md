# Chainer-Fast-Neuralstyle web implementation with Flask

## Overview

Web application implementation for Chainer-Falst-Neuralstyle with Python/Flask

- https://github.com/yusuketomoto/chainer-fast-neuralstyle

## How to use

- (optional)Download/Clone/Generate extra models

    - https://github.com/gafr/chainer-fast-neuralstyle-models

    - Copy \*.model files under models/ into models/ folder.

- (optional)Edit settings.py for your environment

```
port = 5000          # port number for web application
padding = 50         # image padding
gpu = -1             # Use GPU number(-1: No GPU)
median_filter = 3    # Median Filter
keep_colors = False  # Keep original image color ?
leaveimage = False   # Leave Input/Generated image file ?
```

- Run

    - `$ python index.py`

## Reference

- Chainer

    - https://github.com/chainer/chainer

- Chainer-Gogh

    - https://github.com/mattya/chainer-gogh

- Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"

    - https://github.com/yusuketomoto/chainer-fast-neuralstyle

- Chainer-fast-neuralstyle-models

    - https://github.com/gafr/chainer-fast-neuralstyle-models

## Licensing

This code is licensed under MIT.

https://github.com/dotnsf/chainer-fast-neuralstyle-flask/blob/master/LICENSE

## Copyright

2018 [K.Kimura @ Juge.Me](https://github.com/dotnsf) all rights reserved.
