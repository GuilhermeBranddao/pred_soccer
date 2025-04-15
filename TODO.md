
# Criação de novas funcionalidades
- [ ] Criar função teste AB de modelos
- [ ] Criar App intuitiva para realizar predições de partidas utiliza do bons modelos
- [ ] Criar funções mais proficionais
- [ ] Criar função para ver quais as variaveis mais impactam positivamente no modelo
- [ ] Criar funçõo para de treinamento para varios modelos
- [ ]
- [ ]
- [ ]



# Estudo
- [ ] Implemntar modelos emsambles
- [ ] Como saber quais foram as variaveis decisivas que levaram o modelo a realizar a predição
- [ ]
- [ ]









def create_connection(a, shape, order):
    """
    
    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    shape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C', 'F', 'A'}, optional

    Returns
    -------

    Notes
    -----

    Examples
    --------
    >>> import numpy as np
    >>> a = [4, 3, 5, 7, 6, 8]
    >>> indices = [0, 1, 4]
    >>> np.take(a, indices)
    array([4, 3, 6])

    """