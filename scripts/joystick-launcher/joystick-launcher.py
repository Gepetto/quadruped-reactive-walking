#!/usr/bin/env python3
"""Start and stop the odri-controller service."""

import dbus
from inputs import get_gamepad

SERVICE = "gepetto-quadruped-reactive-walking.service"
SYSTEMD = "org.freedesktop.systemd1"
SYSTEMDS = "/" + SYSTEMD.replace(".", "/")


def get_manager():
    sysbus = dbus.SystemBus()
    systemd1 = sysbus.get_object(SYSTEMD, SYSTEMDS)
    return dbus.Interface(systemd1, SYSTEMD + ".Manager")


def start(manager):
    manager.StartUnit(SERVICE, "replace")


def stop(manager):
    manager.StopUnit(SERVICE, "replace")


def main():
    manager = get_manager()
    security = False  # ensure no accidental trigger
    while True:
        for event in get_gamepad():
            if event.code == "BTN_TR":
                security = event.state == 1
            elif security and event.state == 1:
                if event.code == "BTN_NORTH":
                    start(manager)
                elif event.code == "BTN_WEST":
                    stop(manager)


if __name__ == "__main__":
    main()
