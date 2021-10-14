# Joystick Launcher

Start and stop the controller from the a joystick.

## Install

```bash
sudo cp *.service /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable joystick-launcher
sudo systemctl start joystick-launcher
```

## Use

To start the controller, press LR1 + north (triangle on a ps-like gamepad)
To stop the controller, press LR1 + west (square on a ps-like gamepad)
