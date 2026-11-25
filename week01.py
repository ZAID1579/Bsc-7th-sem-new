"""
Week 1 - Advanced Python Data Structures and Functional Programming (Lab VII, AI & ML)
"""

# ----------------------------------------------------------------------
# Q1 - Sensor data: list comprehension (Celsius to Fahrenheit, hot days)
# ----------------------------------------------------------------------
readings = [22.5, 23.1, 20.9, 24.0, 21.5, 19.8, 26.2, 23.5]

fahrenheit_readings = [c * 9 / 5 + 32 for c in readings]
hot_days = [c for c in readings if c > 23.0]

print("Celsius readings:", readings)
print("Fahrenheit readings:", fahrenheit_readings)
print("Hot days (> 23.0 C):", hot_days)

# ----------------------------------------------------------------------
# Q2 - User preferences: dictionary get, update, remove, loop
# ----------------------------------------------------------------------
user_preferences = {"theme": "dark", "notifications": True,
                    "language": "en-US", "font_size": 14}

currency = user_preferences.get("currency", "USD")
print("Currency:", currency)

user_preferences["font_size"] = 16
user_preferences.pop("notifications")

print("Updated preferences:")
for key, value in user_preferences.items():
    print(key, ":", value)

# ----------------------------------------------------------------------
# Q3 - Unique event IDs: set from a list, intersection, union
# ----------------------------------------------------------------------
log1_list = ["A101", "B202", "C303", "A101", "D404"]
log2_events = {"C303", "E505", "F606", "B202"}

log1_events = set(log1_list)
print("Unique events in log 1:", log1_events)
print("Common events:", log1_events & log2_events)
print("All unique events:", log1_events | log2_events)
