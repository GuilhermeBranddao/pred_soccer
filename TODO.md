
# Criação de novas funcionalidades
- [ ] Criar função teste AB de modelos
- [ ] Criar App intuitiva para realizar predições de partidas utiliza do bons modelos
- [ ] Criar funções mais proficionais
- [ ] Criar função para ver quais as variaveis mais impactam positivamente no modelo
- [ ] Criar funçõo para de treinamento para varios modelos
- [ ] Melhorar a estrurura de feedback de funções
    - Exemplo: 
        - err, df = base_pre_processing(df)
        - O 'err' deve ser um atributo que carrege consigo uma informação relevante caso ocorra algum tipo de erro

- [ ]
- [ ]



# Estudo ML
- [ ] Implemntar modelos emsambles
- [ ] Como saber quais foram as variaveis decisivas que levaram o modelo a realizar a predição
- [ ]
- [ ]



# Criação de testes automatizados
- [ ] Estar se as funções estão fazendo o que deveria ser feito
    - Exemplo: testar a função 'encode_categorical_features' ela deve realizar o encode de certas features e salvar em suas respectivas pastas




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