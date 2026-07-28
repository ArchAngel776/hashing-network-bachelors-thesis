class TargetShapeException(Exception):
    def __init__(self, target, size):
        super().__init__()

        self._target = target
        self._size = size

    def __str__(self):
        return f"The target's shape: {self._target.ndim} does not match the desired shape: {self._size}"
