class IncreaingAllocator:
    def __init__(self, first_value: int = 0):
        self.id = first_value - 1
    
    def allocate(self) -> int:
        self.id += 1
        return self.id