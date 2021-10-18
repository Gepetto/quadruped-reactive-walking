#ifndef MQTT_INTERFACE_HPP
#define MQTT_INTERFACE_HPP

#include <mutex>


class MqttInterface {
 public:
  MqttInterface();
  void set(double current, double voltage, double energy, std::string & joystick);
  void setStatus(std::string & status);
  void start();
  void stop();
  bool getStop();
  bool getCalibrate();

 private:
  bool run_;
  double current_;
  double voltage_;
  double energy_;
  std::string status_;
  std::string joystick_;
  std::mutex run_m;
  std::mutex data_m;

  bool getRun();
  std::string getCurrent();
  std::string getVoltage();
  std::string getEnergy();
  std::string getStatus();
  std::string getJoystick();
};
#endif /* !MQTT_INTERFACE_HPP */
