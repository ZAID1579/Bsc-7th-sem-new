"""
Week 2 - Object-Oriented Programming in Python (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - Car class: __init__, instance attributes, display_info()
# ----------------------------------------------------------------------
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def display_info(self):
        print(f"{self.make} {self.model} {self.year}")


car1 = Car("Toyota", "Camry", 2020)
car2 = Car("Honda", "Civic", 2022)
car1.display_info()
car2.display_info()

# ----------------------------------------------------------------------
# Q2 - BankAccount: _balance, deposit(), withdraw() with funds check
# ----------------------------------------------------------------------
class BankAccount:
    def __init__(self, account_number, balance=0):
        self.account_number = account_number
        self._balance = balance

    def deposit(self, amount):
        self._balance += amount
        print(f"Deposited {amount}. New balance: {self._balance}")

    def withdraw(self, amount):
        if amount <= self._balance:
            self._balance -= amount
            print(f"Withdrew {amount}. New balance: {self._balance}")
        else:
            print("Insufficient funds.")


account = BankAccount("ACC12345")
account.deposit(500)
account.withdraw(200)
account.withdraw(400)

# ----------------------------------------------------------------------
# Q3 - Student and GraduateStudent: inheritance, super(), overriding
# ----------------------------------------------------------------------
class Student:
    def __init__(self, name, student_id):
        self.name = name
        self.student_id = student_id

    def get_details(self):
        return f"Name: {self.name}, ID: {self.student_id}"


class GraduateStudent(Student):
    def __init__(self, name, student_id, thesis_topic):
        super().__init__(name, student_id)
        self.thesis_topic = thesis_topic

    def get_details(self):
        return f"Name: {self.name}, ID: {self.student_id}, " \
               f"Thesis: {self.thesis_topic}"


student = Student("Aarav", "S101")
grad_student = GraduateStudent("Meera", "G202", "AI in Healthcare")
print(student.get_details())
print(grad_student.get_details())
