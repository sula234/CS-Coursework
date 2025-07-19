from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from models import engine, User, Member, Caregiver, Adress, Job, JobApplication, Appointment



engine = create_engine('sqlite:///database.db')

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()


########################################################################################################
# STEP 3: Update SQL Statement
########################################################################################################

# 3.1 
# Assuming you have a User object with the name "Askar Askarov" in your database
# askar_user = session.query(User).filter_by(given_name='Askar', surname='Askarov').first()
# print(askar_user)

# # Update the phone number
# askar_user.phone_number = '+77771010001'

# # 3.2
# # Select all caregivers from the database
# all_caregivers = session.query(Caregiver).all()

# # print all hour rates before query
# print([str(caregv.hourly_rate) for caregv in all_caregivers])

# # Update the hourly rates based on the specified conditions
# for caregiver in all_caregivers:
#     if caregiver.hourly_rate < 9:
#         # Add a fixed commission fee of $0.5
#         caregiver.hourly_rate += 0.5
#     else:
#         # Add a 10% commission fee
#         caregiver.hourly_rate += caregiver.hourly_rate * 0.1

# # print all hour rates after query
# all_caregivers = session.query(Caregiver).all()
# print([str(caregv.hourly_rate) for caregv in all_caregivers])
    
# # Commit the changes to the database
# session.commit()

########################################################################################################
# STEP 4: Delete SQL Statement
########################################################################################################

# # Find the user with the given name and surname
# bolat_user = session.query(User).filter_by(given_name='Bolat', surname='Bolatov').first()

# # Check if the user exists
# if bolat_user:
#     # Find and delete the jobs posted by the user
#     session.query(Job).filter_by(member_user_id=bolat_user.user_id).delete()

#     # Commit the changes to the database
#     session.commit()
# else:
#     print("User 'Bolat Bolatov' not found.")


# # Find the user IDs of members who live on Turan street
# turan_street_members = session.query(Member).join(Adress).filter(Adress.street == 'Turan').all()

# # Delete the associated addresses
# for member in turan_street_members:
#     session.query(Adress).filter_by(member_user_id=member.member_user_id).delete()

# # Delete the members who live on Turan street
# session.query(Member).filter(Member.member_user_id.in_([member.member_user_id for member in turan_street_members])).delete(synchronize_session=False)

# # Commit the changes to the database
# session.commit()

########################################################################################################
# STEP 5: Simple Queries
########################################################################################################

# # 5.1 Select caregiver and member names for the accepted appointments.
# apps = session.query(Appointment).filter_by(status='Accepted').all()
# members = []
# caregivers = []

# for app in apps:
#     members += session.query(User).filter_by(user_id=app.member_user_id).all()
#     caregivers += session.query(User).filter_by(user_id=app.caregiver_user_id).all()

# print([mem.given_name for mem in members])
# print([caregiver.given_name for caregiver in caregivers])

# # 5.2 List job ids that contain ‘gentle’ in their other requirements.

# # Define the query to list job IDs with 'gentle' in their other requirements
# query = select(Job.job_id).where(Job.other_requirements.like('%gentle%'))

# # Execute the query
# results = session.execute(query).fetchall()

# # Extract and print the job IDs
# job_ids = [result[0] for result in results]
# print("Job IDs with 'gentle' in other requirements:", job_ids)

# # 5.3 List the work hours of Baby Sitter positions.
# # Define the query to list work hours for Baby Sitter positions
# query = (
#     select(Job.job_id, Appointment.work_hours)
#     .join(Member, Job.member_user_id == Member.member_user_id)
#     .join(Appointment, Member.member_user_id == Appointment.member_user_id)
#     .where(Job.required_caregiving_type == 'Baby Sitter')
# )

# # Execute the query
# results = session.execute(query).fetchall()

# # Print the results
# for job_id, work_hours in results:
#     print(f"Job ID: {job_id}, Work Hours: {work_hours}")


########################################################################################################
# STEP 6: Complex Queries
########################################################################################################

# 6.1 Count the number of applicants for each job posted by a member (multiple joins with aggregation)
from sqlalchemy import func

# Define the query to count applicants for each job posted by a member
query = (
    select(
        Member.member_user_id,
        Job.job_id,
        func.count(JobApplication.caregiver_user_id).label("applicant_count")
    )
    .join(Job, Member.member_user_id == Job.member_user_id)
    .outerjoin(JobApplication, Job.job_id == JobApplication.job_id)
    .group_by(Member.member_user_id, Job.job_id)
)

# Execute the query
results = session.execute(query).fetchall()

# Print the results
for member_id, job_id, applicant_count in results:
    print(f"Member ID: {member_id}, Job ID: {job_id}, Applicant Count: {applicant_count}")


# 6.2 Total hours spent by care givers for all accepted appointments (multiple joins with aggregation)
# Define the query to calculate total hours spent by caregivers for all accepted appointments
query = (
    select(
        Caregiver.caregiver_user_id,
        func.sum(Appointment.work_hours).label("total_hours")
    )
    .join(Appointment, Caregiver.caregiver_user_id == Appointment.caregiver_user_id)
    .filter(Appointment.status == 'Accepted')
    .group_by(Caregiver.caregiver_user_id)
)

# Execute the query
results = session.execute(query).fetchall()

# Print the results
for caregiver_id, total_hours in results:
    print(f"Caregiver ID: {caregiver_id}, Total Hours: {total_hours}")


# 6.3 Average pay of caregivers based on accepted appointments (join with aggregation)

# Define the query to calculate the average pay of caregivers based on accepted appointments
query = (
    select(
        Caregiver.caregiver_user_id,
        func.avg(Caregiver.hourly_rate * Appointment.work_hours).label("average_pay")
    )
    .join(Appointment, Caregiver.caregiver_user_id == Appointment.caregiver_user_id)
    .filter(Appointment.status == 'Accepted')
    .group_by(Caregiver.caregiver_user_id)
)

# Execute the query
results = session.execute(query).fetchall()

# Print the results
for caregiver_id, average_pay in results:
    print(f"Caregiver ID: {caregiver_id}, Average Pay: {average_pay}")


# 6.4 Caregivers who earn above average based on accepted appointments (multiple join with aggregation and nested query)

# Subquery to calculate the average hourly rate for caregivers with accepted appointments
average_hourly_rate_subquery = (
    session.query(
        func.avg(Caregiver.hourly_rate).label('average_hourly_rate')
    )
    .join(Appointment, Appointment.caregiver_user_id == Caregiver.caregiver_user_id)
    .filter(Appointment.status == 'Accepted')
    .group_by(Caregiver.caregiver_user_id)
    .subquery()
)

# Select caregivers with hourly rates above the calculated average
above_average_caregivers = (
    session.query(Caregiver)
    .join(Appointment, Appointment.caregiver_user_id == Caregiver.caregiver_user_id)
    .filter(Appointment.status == 'Accepted')
    .join(average_hourly_rate_subquery, Caregiver.hourly_rate > average_hourly_rate_subquery.c.average_hourly_rate)
    .all()
)

for caregiver in above_average_caregivers:
    print(caregiver)

########################################################################################################
# STEP 7: Query with a Derived Attribute
########################################################################################################
# Calculate the total cost for all accepted appointments
total_cost = (
    session.query(func.sum(Caregiver.hourly_rate * Appointment.work_hours).label('total_cost'))
    .join(Appointment, Appointment.caregiver_user_id == Caregiver.caregiver_user_id)
    .filter(Appointment.status == 'Accepted')
    .scalar()
)

print(f'Total cost to pay for all accepted appointments: ${total_cost}')

########################################################################################################
# STEP 8: View Operation
########################################################################################################

# Query to view all job applications and the applicants with user information
job_applications_info = (
    session.query(
        JobApplication,
        User.given_name.label('user_given_name'),
        User.surname.label('user_surname'),
        Caregiver.caregiving_type,
        Job.required_caregiving_type,
        Job.other_requirements
    )
    .join(Caregiver, JobApplication.caregiver_user_id == Caregiver.caregiver_user_id)
    .join(User, User.user_id == Caregiver.caregiver_user_id)
    .join(Job, JobApplication.job_id == Job.job_id)
    .all()
)

# Print the results
for application, user_given_name, user_surname, caregiving_type, required_caregiving_type, other_requirements in job_applications_info:
    print(f"Job Application ID: {application.job_id}, Applicant: {user_given_name} {user_surname}, Caregiving Type: {caregiving_type}, Required Caregiving Type: {required_caregiving_type}, Other Requirements: {other_requirements}")

