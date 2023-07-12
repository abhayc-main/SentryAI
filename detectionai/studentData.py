import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import uuid

cred = credentials.Certificate("./firebase/serviceAccount.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

def create_new_school(name, location):
    school_id = str(uuid.uuid4())  # Generate a random school ID
    school_ref = db.collection("Schools").document(school_id)
    school_data = {
        "name": name,
        "location": location,
        "students": []
    }
    school_ref.set(school_data)
    print("New school created successfully.")
    return school_id

def add_new_person(school_id, name, grade):
    person_id = str(uuid.uuid4())  # Generate a random person ID
    person_data = {
        "name": name,
        "grade": grade,
        "last_attendance_time": ""
    }
    school_ref = db.collection("Schools").document(school_id)
    school_ref.collection("students").document(person_id).set(person_data)
    print("New person added successfully.")

# Example usage
new_school_name = "Example School"
new_school_location = "City, Country"
new_school_id = create_new_school(new_school_name, new_school_location)

new_person_name = "Washist Kolluru"
new_person_grade = 11
add_new_person(new_school_id, new_person_name, new_person_grade)
