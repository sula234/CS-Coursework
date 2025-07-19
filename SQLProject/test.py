from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import engine, User, Member, Caregiver, Address, Job, JobApplication, Appointment


Session = sessionmaker(bind=engine)
session = Session()

# # Select all tuples from the User table
users0 = session.query(User).all()
users1 = session.query(Caregiver).all()
users2 = session.query(Address).all()
users3 = session.query(Member).all()
users4 = session.query(Job).all()
users5 = session.query(JobApplication).all()
users6 = session.query(Appointment).all()

print(len(users0))
print(len(users1))
print(len(users2))
print(len(users3))
print(len(users4))
print(len(users5))
print(len(users6))