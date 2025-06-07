from animal import Animal

class Mammal(Animal):
    def __init__(self,name,gestation_period):
        super().__init__(name,'mammal')
        self.gestation = gestation_period

    @property
    def info(self): # 重写父类方法
        return f"{super().info} | Gestation:{self.gestation} months"  