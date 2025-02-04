class Counter:
    def __init__(self):
        self.cnt = 0

    def count(self) -> int:
        self.cnt += 1
        return self.cnt

    def value(self) -> int:
        return self.cnt