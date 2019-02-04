"""Consider a Car class as given in the code. Write a Service class as given
in the class diagram below which performs various activities on a list of cars.
Assume that the car_list is sorted by year in ascending order."""

class Car:
    def __init__(self,model,year,registration_number):
        self.__model=model
        self.__year=year
        self.__registration_number=registration_number

    def get_model(self):
        return self.__model

    def get_year(self):
        return self.__year

    def get_registration_number(self):
        return self.__registration_number

    def __str__(self):
        return(self.__model+" "+self.__registration_number+" "+(str)(self.__year))

class Service:
    def __init__(self, car_list):
        self.__car_list = car_list

    def get_car_list(self):
        return self.__car_list

    def find_cars_by_year(self, year):
        car_list_by_year = []
        for car in car_list:
            if(car.get_year() == year):
                car_list_by_year.append(car)
        if(len(car_list_by_year) > 0):
            return car_list_by_year
        else:
            return None

    def add_cars(self, new_car_list):
        for car in new_car_list:
            self.get_car_list().append(car)
        self.get_car_list().sort(key=lambda car: car.get_year())        

    def remove_cars_from_karnataka(self):
        for car in self.get_car_list():
            if("KA" in car.get_registration_number() or "ka" in
               car.get_registration_number()):
                self.get_car_list().remove(car)


        """Testing Code Begin"""
"""--------------------------------"""
car_list = []
new_car_list = []
car = Car("Camry", 2009, "HAK3192")
car_list.append(car)
car = Car("Nissan", 2009, "HOK3192")
car_list.append(car)
car = Car("Bugatti", 2019, "KAK3192")
car_list.append(car)
s = Service(car_list)


#New Car List
car = Car("Corolla", 2008, "PAK31934")
new_car_list.append(car)
car = Car("Tundra", 2018, "HAK3172")
new_car_list.append(car)
car = Car("Lexus", 2015, "HAK4422")
new_car_list.append(car)
car = Car("Mercedes", 2020, "HAK1192")
new_car_list.append(car)

cars = s.get_car_list()

print("Adding new cars")
s.add_cars(new_car_list)

print()
print("Before deletion")
for car in cars:
    print(car)

s.remove_cars_from_karnataka()
print()
print("After Deletion")
for car in cars:
    print(car)

"""Testing Code End"""
"""--------------------------------"""


