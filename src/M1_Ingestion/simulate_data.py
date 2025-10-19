import csv
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()
NUM_RECORDS = 10000
OUTPUT_FILE = "raw_traffic_violations.csv"

VIOLATION_TYPES = ["Speeding", "Red Light", "Illegal Turn", "No Signal", "Illegal Parking"]
VEHICLE_TYPES = ["Sedan", "Truck", "Motorcycle", "Van", "Bus"]
SEVERITY_LEVELS = [1, 2, 3, 4, 5]
STANDARD_TS_FORMAT = "%Y-%m-%d %H:%M:%S"

def create_inconsistent_timestamp(dt_obj):
    if random.random() < 0.08:
        choice = random.choice([
            'N/A',
            dt_obj.strftime("%m/%d/%y %H:%M"),
            dt_obj.strftime("%Y/%d/%m %H:%M:%S"),
            '2026-Oct-30 11:00:00'
        ])
        return choice

    if random.random() < 0.05:
        return dt_obj.strftime(STANDARD_TS_FORMAT).replace('-', '_')

    return dt_obj.strftime(STANDARD_TS_FORMAT)

print(f"Starting data simulation for {NUM_RECORDS} records...")

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow([
        "Violation ID", "Timestamp", "Location", "Violation Type", "Vehicle Type", "Severity"
    ])

    for i in range(1, NUM_RECORDS + 1):
        violation_id = f"VIO-{i:06d}"

        start_date = datetime.now() - timedelta(days=365)
        dt_obj = fake.date_time_between(start_date=start_date, end_date="now")

        timestamp = create_inconsistent_timestamp(dt_obj)

        location = f"{fake.latitude():.4f},{fake.longitude():.4f}"

        violation_type = random.choice(VIOLATION_TYPES)
        vehicle_type = random.choice(VEHICLE_TYPES)

        severity = random.choice(SEVERITY_LEVELS)

        if random.random() < 0.10:
            if random.choice([True, False]):
                location = ""
            else:
                severity = ""

        if random.random() < 0.05:
            severity = random.choice(["HIGH", "LOW", ""])

        if random.random() < 0.03:
             violation_type = violation_type.lower()

        writer.writerow([violation_id, timestamp, location, violation_type, vehicle_type, severity])

print(f"Successfully generated {NUM_RECORDS} records to {OUTPUT_FILE}")
