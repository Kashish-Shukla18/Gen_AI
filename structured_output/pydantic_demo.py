from pydantic import BaseModel,EmailStr,Field
from typing import Optional


Class Student(BaseModel):
name: str
age: Optional[int] = None
email: EmailStr
cgpa:float=Field(gt=0,lt=10)

# new_student = Student(name="John")
new_student = {"name": "John", "age": 20, "email": "kashish@example.com", "cgpa": 9}
print(new_student)

student_dict=dict(new_student)
print(student_dict)
student_json=Student.model_dump_json()
print(student_json)
Student=Student(**new_student)
print(Student)