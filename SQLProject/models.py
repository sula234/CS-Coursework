from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Date, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    user_id = Column(Integer, primary_key=True)
    email = Column(String)
    given_name = Column(String)
    surname = Column(String)
    city = Column(String)
    phone_number = Column(String)
    profile_description = Column(String)
    password = Column(String)

    #caregiver = relationship("Caregiver", cascade="all,delete", backref="User")
    #member = relationship("Member", cascade="all,delete", backref="User")



    def __repr__(self):
        return f'''User ID: {self.user_id}, Email: {self.email}, Given Name: {self.given_name}, 
            Surname: {self.surname}, Phone Number: {self.phone_number}, Profile Description: {self.profile_description}, 
            Password: {self.password}, City: {self.city}\n '''
    
class Caregiver(Base):
    __tablename__ = 'caregiver'
    caregiver_user_id = Column(Integer, ForeignKey('user.user_id', ondelete="CASCADE"), primary_key=True)
    photo = Column(String)
    gender = Column(String)
    caregiving_type = Column(String)
    hourly_rate = Column(Float)

    def __repr__(self):
        return f"ID: {self.caregiver_user_id}, Hourly Rate: {self.hourly_rate}"
    

class Member(Base):
    __tablename__ = 'member'
    member_user_id = Column(Integer, ForeignKey('user.user_id', ondelete="CASCADE"), primary_key=True)
    house_rules = Column(String)
    
    #adress = relationship("Adress", cascade="all,delete", backref="Member")
    #job = relationship("Job", cascade="all, delete", backref="member")

class Adress(Base):
    __tablename__ = 'adress'
    member_user_id = Column(Integer, ForeignKey('member.member_user_id', ondelete="CASCADE"), primary_key=True)
    house_number = Column(String)
    street = Column(String)
    town = Column(String, ForeignKey('user.city'))

    #member = relationship("Member", cascade="all, delete", back_populates="member")

class Job(Base):
    __tablename__ = 'job'
    job_id = Column(Integer, primary_key=True)
    member_user_id = Column(Integer, ForeignKey('member.member_user_id', ondelete="CASCADE"))
    required_caregiving_type = Column(String)
    other_requirements = Column(String)
    date_posted = Column(Date)

    def __repr__(self):
        return f'''Dates Posted: {self.date_posted}, Other reqirements: {self.other_requirements}, Required Caregiving Type: {self.required_caregiving_type} '''

class JobApplication(Base):
    __tablename__ = 'job_application'
    caregiver_user_id = Column(Integer, ForeignKey('caregiver.caregiver_user_id', ondelete="CASCADE"), primary_key=True)
    job_id = Column(Integer, ForeignKey('job.job_id'), primary_key=True)
    date_applied = Column(Date)
    


class Appointment(Base):
    __tablename__ = 'appointment'
    appointment_id = Column(Integer, primary_key=True)
    caregiver_user_id = Column(Integer, ForeignKey('caregiver.caregiver_user_id'))
    member_user_id = Column(Integer, ForeignKey('member.member_user_id'))
    appointment_date = Column(Date)
    appointment_time = Column(String)
    work_hours = Column(Float)
    status = Column(String)

# Create an SQLite database in memory for testing purposes
engine = create_engine('sqlite:///database.db')
Base.metadata.create_all(engine)