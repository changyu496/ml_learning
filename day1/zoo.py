from animal import Animal
from mammal import Mammal

def animal_sound(animals):
    for animal in animals:
        print(f"{animal.name} says {animal.make_sound()}")

class Bird(Animal):
    def make_sound(self):
        return "Chirp"

class Lion(Mammal):
    def make_sound(self):
        return "Roar"
    
b = Bird("小鸟","禽类")
l = Lion("狮子","猫科")

animals = [b,l]

animal_sound(animals=animals)