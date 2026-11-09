from gpiozero import MotionSensor
from datetime import datetime

PIR_PIN = 17  # change to your GPIO number

pir = MotionSensor(PIR_PIN)
print("Warming up... waiting for motion. Ctrl+C to stop.")

while True:
    pir.wait_for_motion()
    print(f"{datetime.now():%H:%M:%S}  Motion detected")
    pir.wait_for_no_motion()
    print(f"{datetime.now():%H:%M:%S}  Motion stopped")
