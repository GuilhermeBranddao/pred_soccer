from typing import Any, Optional

class FunctionResult:
    """
    Representa o resultado de uma função, contendo informações sobre sucesso, mensagem, dados e exceção (se houver).
    """

    def __init__(self, success: bool, message: str = "", data: Any = None, exception: Optional[Exception] = None):
        self.success = success
        self.message = message
        self.data = data
        self.exception = exception

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        status = "✅ Success" if self.success else "❌ Error"
        if self.exception:
            return f"{status}: {self.message} | {type(self.exception).__name__}: {self.exception}"
        return f"{status}: {self.message}"