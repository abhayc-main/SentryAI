import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import uuid

cred = credentials.Certificate("./firebase/serviceAccount.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': "sentryai-777b5.appspot.com"
})

ref = db.reference('Students')

def add_new_person(name, grade):
    person_id = str(uuid.uuid4())  # Generate a random person ID
    data = {
        "name": name,
        "grade": grade,
        "last_attendance_time": ""
    }
    ref.child(person_id).set(data)
    print("New person added successfully.")

# Example usage
add_new_person("Aditya Donkada", 11)

# Example Data
data = {
    "321654":
        {
            "name": "Abhay Chebium",
            "grade": 11,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "852741":
        {
            "name": "Washist Kolluru",
            "grade": 11,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "963852":
        {
            "name": "Pradyu Kandala",
            "grade": 11,
            "last_attendance_time": "2022-12-11 00:54:34"
        }
}

