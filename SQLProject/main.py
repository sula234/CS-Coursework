from sqlalchemy.orm import sessionmaker
from models import engine, User, Member, Caregiver, Adress, Job, JobApplication, Appointment
from datetime import datetime

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

# Insert data into the User table

# # First is the caregivers
users = [User(email='user1@example.com', given_name='John', surname='Doe', city='City1', phone_number='123456789', profile_description='Profile 1', password='pass1'), 
                User(email='user2@example.com', given_name='Jane', surname='Doe', city='City2', phone_number='987654321', profile_description='Profile 2', password='pass2'),
                User(email='user3@example.com', given_name='Paul', surname='Paul', city='Paris', phone_number='123456789', profile_description='Profile 1', password='pass1'), 
                User(email='user4@example.com', given_name='Jane', surname='Doe', city='City2', phone_number='987654321', profile_description='Profile 2', password='pass2'),
                 # Askar Askarov  is needed for step 2
                User(email='askar@example.com', given_name='Askar', surname='Askarov', city='Almaty', phone_number='870000000', profile_description='Hello, world!', password='qwerty'), 
                User(email='hello@example.com', given_name='Jane', surname='Doe', city='City2', phone_number='987654321', profile_description='Profile 2', password='pass2'),
                User(email='help@example.com', given_name='Vladimir', surname='Put', city='Moscow', phone_number='1231231234', profile_description='Profile 1', password='pass1'), 
                User(email='bob@example.com', given_name='Bob', surname='Grey', city='Astana', phone_number='0303030303', profile_description=':)', password='password'),
                User(email='hey@example.com', given_name='John', surname='Doe', city='City1', phone_number='123456789', profile_description='>>>>>', password='pass1'), 
                User(email='wer@example.com', given_name='Sasha', surname='Kasha', city='City2', phone_number='987654321', profile_description='Profile 2', password='pass2'),

                # Members: 
                User(email='care1@example.com', given_name='Sultan', surname='Kasenov', city='Aktau', phone_number='987654321', profile_description='Profile 2', password='pass2'),
                User(email='care2@example.com', given_name='John', surname='ALL', city='City1', phone_number='123456789', profile_description='Profile 1', password='pass1'), 
                User(email='care3@example.com', given_name='Peter', surname='Parker', city='Aktau', phone_number='987654321', profile_description='Profile 2', password='pass2'),
                User(email='der@example.com', given_name='Askar', surname='Uly', city='Astana', phone_number='877777777', profile_description='Hello, KZ!', password='qwerty'), 
                User(email='sigma@example.com', given_name='Jane', surname='Doe', city='Astana', phone_number='987654321', profile_description='Profile 2', password='pass2'),

                # Bolat Bolatov is needed for step 4 (id 15 in users list)
                User(email='Bolat@example.com', given_name='Bolat', surname='Bolatov', city='Almaty', phone_number='123456789', profile_description='Profile 1', password='pass1'), 
                User(email='bob@example.com', given_name='Bob', surname='Grey', city='Astana', phone_number='0303030303', profile_description=':)', password='password'),
                User(email='>>>>@example.com', given_name='John', surname='Balck', city='Astana', phone_number='33333', profile_description='Profile 1', password='1234456'), 
                User(email='dont@example.com', given_name='Kairat', surname='Nurtas', city='Almaty', phone_number='987654321', profile_description='Profile 2', password='pass2'),
                User(email='yes@example.com', given_name='Dim', surname='Dimic', city='Astana', phone_number='987654321', profile_description='Profile 2', password='pass2'),
                 ]


session.add_all(users)
session.commit()

# Insert data into the Caregiver table
caregivers = [Caregiver(caregiver_user_id=users[0].user_id, photo='caregiver1.jpg', gender='Male', caregiving_type='Type1', hourly_rate=10.0),
                 Caregiver(caregiver_user_id=users[1].user_id, photo='caregiver2.jpg', gender='Female', caregiving_type='Type2', hourly_rate=25.0),
                 Caregiver(caregiver_user_id=users[2].user_id, photo='caregiver1.jpg', gender='Male', caregiving_type='Type1', hourly_rate=20.0),
                 Caregiver(caregiver_user_id=users[3].user_id, photo='caregiver2.jpg', gender='Female', caregiving_type='Type2', hourly_rate=25.0),
                 Caregiver(caregiver_user_id=users[4].user_id, photo='caregiver1.jpg', gender='Male', caregiving_type='Type1', hourly_rate=8.0),
                 Caregiver(caregiver_user_id=users[5].user_id, photo='caregiver2.jpg', gender='Female', caregiving_type='Type2', hourly_rate=25.0),
                 Caregiver(caregiver_user_id=users[6].user_id, photo='caregiver1.jpg', gender='Male', caregiving_type='Type1', hourly_rate=20.0),
                 Caregiver(caregiver_user_id=users[7].user_id, photo='caregiver2.jpg', gender='Male', caregiving_type='Type2', hourly_rate=2.0),
                 Caregiver(caregiver_user_id=users[8].user_id, photo='caregiver1.jpg', gender='FeMale', caregiving_type='Type1', hourly_rate=20.0),
                 Caregiver(caregiver_user_id=users[9].user_id, photo='caregiver2.jpg', gender='Male', caregiving_type='Type2', hourly_rate=25.0),
                 ]

session.add_all(caregivers)
session.commit()

# Insert data into the Member table
members = [Member(member_user_id=users[12].user_id, house_rules='House Rules 2'),
                 Member(member_user_id=users[13].user_id, house_rules='House Rules 2'),
                 Member(member_user_id=users[14].user_id, house_rules='House Rules 2'),
                 Member(member_user_id=users[15].user_id, house_rules='House Rules 2'),
                 Member(member_user_id=users[16].user_id, house_rules='House Rules 2'),
                 Member(member_user_id=users[17].user_id, house_rules='House Rules 2'),
                 Member(member_user_id=users[18].user_id, house_rules='House Rules 2'),
                 Member(member_user_id=users[19].user_id, house_rules='House Rules 2'),
                 Member(member_user_id=users[11].user_id,house_rules='House Rules 2'),
                 Member(member_user_id=users[10].user_id, house_rules='House Rules 2'),
                 ]
session.add_all(members)
session.commit()


session.add_all([Adress(member_user_id=members[0].member_user_id, house_number='123', street='Turan', town=users[10].city),
                 Adress(member_user_id=members[1].member_user_id, house_number='123', street='Street1', town=users[11].city),
                 Adress(member_user_id=members[2].member_user_id, house_number='123', street='Street1', town=users[12].city),
                 Adress(member_user_id=members[3].member_user_id, house_number='123', street='Street1', town=users[13].city),
                 Adress(member_user_id=members[4].member_user_id, house_number='123', street='Street1', town=users[14].city),
                 Adress(member_user_id=members[5].member_user_id, house_number='123', street='Street1', town=users[15].city),
                 Adress(member_user_id=members[6].member_user_id, house_number='123', street='Street1', town=users[16].city),
                 Adress(member_user_id=members[7].member_user_id, house_number='123', street='Kabanabai', town=users[17].city),
                 Adress(member_user_id=members[8].member_user_id, house_number='123', street='Street1', town=users[18].city),
                 Adress(member_user_id=members[9].member_user_id, house_number='123', street='Turan', town=users[19].city),
                 ])
session.commit()

# Insert data into the Job table
# be careful with 10, 15, 19
jobs = [Job(member_user_id=members[0].member_user_id, required_caregiving_type='Baby Sitter', other_requirements='gentle', date_posted=datetime(year=2023, month=4, day=15)),
                 Job(member_user_id=members[1].member_user_id, required_caregiving_type='Elderly Care', other_requirements='Requirements 1', date_posted=datetime(year=2023, month=1, day=1)),
                 Job(member_user_id=members[2].member_user_id, required_caregiving_type='Baby Sitter', other_requirements='gentle', date_posted=datetime(year=2023, month=9, day=13)),
                 Job(member_user_id=members[3].member_user_id, required_caregiving_type='Elderly Care', other_requirements='gentle', date_posted=datetime(year=2023, month=11, day=11)),
                 Job(member_user_id=members[4].member_user_id, required_caregiving_type='Baby Sitter', other_requirements='Requirements 2', date_posted=datetime(year=2023, month=3, day=10)),
                 Job(member_user_id=members[5].member_user_id, required_caregiving_type='Elderly Care', other_requirements='Be good', date_posted=datetime(year=2023, month=8, day=30)),
                 Job(member_user_id=members[6].member_user_id, required_caregiving_type='Type2', other_requirements='Requirements 2', date_posted=datetime(year=2023, month=10, day=23)),
                 Job(member_user_id=members[5].member_user_id, required_caregiving_type='Elderly Care', other_requirements='No pets', date_posted=datetime(year=2023, month=7, day=10)),
                 Job(member_user_id=members[8].member_user_id, required_caregiving_type='Baby Sitter', other_requirements='Requirements 2', date_posted=datetime(year=2023, month=6, day=13)),
                 Job(member_user_id=members[5].member_user_id, required_caregiving_type='Childern', other_requirements='Be nice', date_posted=datetime(year=2023, month=10, day=13)),
                 ]
session.add_all(jobs)
session.commit()

# Insert data into the JobApplication table
session.add_all([JobApplication(caregiver_user_id=caregivers[0].caregiver_user_id, job_id=jobs[0].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 JobApplication(caregiver_user_id=caregivers[3].caregiver_user_id, job_id=jobs[0].job_id, date_applied=datetime(year=2023, month=3, day=10)),
                 JobApplication(caregiver_user_id=caregivers[9].caregiver_user_id, job_id=jobs[3].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 JobApplication(caregiver_user_id=caregivers[7].caregiver_user_id, job_id=jobs[6].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 JobApplication(caregiver_user_id=caregivers[7].caregiver_user_id, job_id=jobs[9].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 JobApplication(caregiver_user_id=caregivers[1].caregiver_user_id, job_id=jobs[2].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 JobApplication(caregiver_user_id=caregivers[2].caregiver_user_id, job_id=jobs[1].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 JobApplication(caregiver_user_id=caregivers[6].caregiver_user_id, job_id=jobs[8].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 JobApplication(caregiver_user_id=caregivers[1].caregiver_user_id, job_id=jobs[7].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 JobApplication(caregiver_user_id=caregivers[2].caregiver_user_id, job_id=jobs[7].job_id, date_applied=datetime(year=2023, month=9, day=13)),
                 ])
session.commit()

# # Insert data into the Appointment table

# be careful with 10, 15, 19
session.add_all([Appointment(caregiver_user_id=users[0].user_id, member_user_id=users[13].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='14:00:00', work_hours=3.5, status='Accepted'),
                 Appointment(caregiver_user_id=users[1].user_id, member_user_id=users[17].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='14:00:00', work_hours=5, status='Accepted'),
                 Appointment(caregiver_user_id=users[2].user_id, member_user_id=users[17].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='15:00:00', work_hours=10, status='Accepted'),
                 Appointment(caregiver_user_id=users[3].user_id, member_user_id=users[18].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='14:00:00', work_hours=2, status='Accepted'),
                 Appointment(caregiver_user_id=users[4].user_id, member_user_id=users[13].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='13:00:00', work_hours=1, status='Accepted'),
                 Appointment(caregiver_user_id=users[5].user_id, member_user_id=users[12].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='14:00:00', work_hours=3.5, status='Rejected'),
                 Appointment(caregiver_user_id=users[6].user_id, member_user_id=users[13].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='14:00:00', work_hours=3.5, status='Rejected'),
                 Appointment(caregiver_user_id=users[7].user_id, member_user_id=users[16].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='21:00:00', work_hours=23, status='Rejected'),
                 Appointment(caregiver_user_id=users[0].user_id, member_user_id=users[14].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='14:00:00', work_hours=777, status='Accepted'),
                 Appointment(caregiver_user_id=users[9].user_id, member_user_id=users[11].user_id, appointment_date=datetime(year=2023, month=9, day=13), appointment_time='11:00:00', work_hours=1.5, status='Rejected'),
                 ])
session.commit()
