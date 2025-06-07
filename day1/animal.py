class Animal:
    def __init__(self,name,spacies):
        self.name = name
        self.spacies = spacies

    @property
    def info(self):
        return f"{self.name} ({self.spacies})"