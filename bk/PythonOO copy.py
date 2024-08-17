import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self):
        return round(math.pi * self.radius ** 2, 2)



#==================================================================




class ObjectCounter:
     num_instances = 0
     def __init__(self):
         ObjectCounter.num_instances += 1

ObjectCounter()

ObjectCounter()

ObjectCounter.num_instances

counter = ObjectCounter()
counter.num_instances


#==================================================



from ScreenMain import Circle

circle_1 = Circle(42)
circle_2 = Circle(7)


#=================================================
class Car:
    def __init__(self, make, model, year, color):
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        self.started = False
        self.speed = 0
        self.max_speed = 200


from car import Car

toyota_camry = Car("Toyota", "Camry", 2022, "Red")
toyota_camry.make
toyota_camry.model
toyota_camry.color
toyota_camry.speed


ford_mustang = Car("Ford", "Mustang", 2022, "Black")
ford_mustang.make
ford_mustang.model
ford_mustang.year
ford_mustang.max_speed


#=================================================================
# robot.py

class IndustrialRobot:
    def __init__(self):
        self.body = Body()
        self.arm = Arm()

    def rotate_body_left(self, degrees=10):
        self.body.rotate_left(degrees)

    def rotate_body_right(self, degrees=10):
        self.body.rotate_right(degrees)

    def move_arm_up(self, distance=10):
        self.arm.move_up(distance)

    def move_arm_down(self, distance=10):
        self.arm.move_down(distance)

    def weld(self):
        self.arm.weld()

class Body:
    def __init__(self):
        self.rotation = 0

    def rotate_left(self, degrees=10):
        self.rotation -= degrees
        print(f"Rotating body {degrees} degrees to the left...")

    def rotate_right(self, degrees=10):
        self.rotation += degrees
        print(f"Rotating body {degrees} degrees to the right...")

class Arm:
    def __init__(self):
        self.position = 0

    def move_up(self, distance=1):
        self.position += 1
        print(f"Moving arm {distance} cm up...")

    def move_down(self, distance=1):
        self.position -= 1
        print(f"Moving arm {distance} cm down...")

    def weld(self):
        print("Welding...")

from robot import IndustrialRobot
robot = IndustrialRobot()
robot.rotate_body_left()
robot.move_arm_up(15)
robot.weld()

robot.rotate_body_right(20)
robot.move_arm_down(5)
robot.weld()


#====================================================================
