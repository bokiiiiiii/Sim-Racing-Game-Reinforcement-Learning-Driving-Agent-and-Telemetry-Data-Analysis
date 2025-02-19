import pyvjoy

joystick = pyvjoy.VJoyDevice(1)

if pyvjoy._sdk.vJoyEnabled():
    print("ACC Controller initialized successfully.")
else:
    print("ACC Controller initialization failed.")


def set_steering(value):  # -100 <= value <= 100
    steering_value = int((value + 100) / 200 * 32768)
    joystick.set_axis(pyvjoy.HID_USAGE_X, steering_value)


def set_throttle(value):  # 0 <= value <= 100
    throttle_value = int(value / 100 * 32768)
    joystick.set_axis(pyvjoy.HID_USAGE_SL1, throttle_value)


def set_brake(value):  # 0 <= value <= 100
    brake_value = int(value / 100 * 32768)
    joystick.set_axis(pyvjoy.HID_USAGE_SL0, brake_value)


def reset():
    set_steering(0)
    set_throttle(0)
    set_brake(0)
