#include "qrw/mqtt-interface.hpp"

#include <cstring>
#include <iostream>
#include <thread>

#include "MQTTClient.h"

#define ADDRESS "tcp://localhost:1883"
#define CLIENTID "Gepetto QRW MQTT Interface"

bool stop_ = false;
bool calibrate_ = false;
std::mutex commands_m;

void delivered(void * /*context*/, MQTTClient_deliveryToken /*dt*/) {}

int msgarrvd(void *context, char *topicName, int topicLen,
             MQTTClient_message *message) {
  std::string payload((char *)message->payload, message->payloadlen);
  if (payload == "stop") {
    std::lock_guard<std::mutex> guard(commands_m);
    stop_ = true;
  } else if (payload == "calibrate") {
    std::lock_guard<std::mutex> guard(commands_m);
    calibrate_ = true;
  } else {
    std::cerr << "Unknown Message on " << topicName << ": " << payload
              << std::endl;
  }

  MQTTClient_freeMessage(&message);
  MQTTClient_free(topicName);
  return 1;
}

void connlost(void *context, char *cause) {
  std::cerr << "Connection lost" << std::endl;
  std::cerr << "cause: " << cause << std::endl;
}

void MqttInterface::set(double current, double voltage, double energy,
                        std::string &joystick) {
  std::lock_guard<std::mutex> guard(data_m);
  current_ = current;
  voltage_ = voltage;
  energy_ = energy;
  joystick_ = joystick;
}

void MqttInterface::setStatus(std::string &status) {
  std::lock_guard<std::mutex> guard(data_m);
  status_ = status;
}

MqttInterface::MqttInterface()
    : run_(true), current_(0), voltage_(0), energy_(0) {}

void MqttInterface::start() {
  int rc;

  MQTTClient client;
  MQTTClient_connectOptions conn_opts = MQTTClient_connectOptions_initializer;
  MQTTClient_create(&client, ADDRESS, CLIENTID, MQTTCLIENT_PERSISTENCE_NONE,
                    NULL);
  conn_opts.keepAliveInterval = 20;
  conn_opts.cleansession = 1;
  MQTTClient_setCallbacks(client, NULL, connlost, msgarrvd, delivered);
  if ((rc = MQTTClient_connect(client, &conn_opts)) != MQTTCLIENT_SUCCESS) {
    printf("Failed to connect, return code %d\n", rc);
    exit(EXIT_FAILURE);
  }
  MQTTClient_subscribe(client, "/odri/commands", 0);
  MQTTClient_deliveryToken token;

  std::string current_s;
  std::string voltage_s;
  std::string energy_s;
  std::string status_s;
  std::string joystick_s;
  const char *current_c;
  const char *voltage_c;
  const char *energy_c;
  const char *status_c;
  const char *joystick_c;

  while (getRun()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(330));

    current_s = getCurrent();
    voltage_s = getVoltage();
    energy_s = getEnergy();
    status_s = getStatus();
    joystick_s = getJoystick();
    current_c = current_s.data();
    voltage_c = voltage_s.data();
    energy_c = energy_s.data();
    status_c = status_s.data();
    joystick_c = joystick_s.data();

    MQTTClient_publish(client, "/odri/current", std::strlen(current_c),
                       current_c, 0, 0, &token);
    MQTTClient_publish(client, "/odri/voltage", std::strlen(voltage_c),
                       voltage_c, 0, 0, &token);
    MQTTClient_publish(client, "/odri/energy", std::strlen(energy_c), energy_c,
                       0, 0, &token);
    MQTTClient_publish(client, "/odri/status", std::strlen(status_c), status_c,
                       0, 0, &token);
    MQTTClient_publish(client, "/odri/joystick", std::strlen(joystick_c),
                       joystick_c, 0, 0, &token);
  }

  MQTTClient_disconnect(client, 10000);
  MQTTClient_destroy(&client);
}

void MqttInterface::stop() {
  std::lock_guard<std::mutex> guard(run_m);
  run_ = false;
}

bool MqttInterface::getStop() {
  std::lock_guard<std::mutex> guard(commands_m);
  if (stop_) {
    stop_ = false;
    return true;
  }
  return false;
}

bool MqttInterface::getCalibrate() {
  std::lock_guard<std::mutex> guard(commands_m);
  if (calibrate_) {
    calibrate_ = false;
    return true;
  }
  return false;
}

bool MqttInterface::getRun() {
  std::lock_guard<std::mutex> guard(run_m);
  return run_;
}

std::string MqttInterface::getStatus() {
  std::lock_guard<std::mutex> guard(data_m);
  return status_;
}

std::string MqttInterface::getJoystick() {
  std::lock_guard<std::mutex> guard(data_m);
  return joystick_;
}

std::string MqttInterface::getCurrent() {
  std::lock_guard<std::mutex> guard(data_m);
  std::string ret = std::to_string(current_);
  return ret;
}

std::string MqttInterface::getVoltage() {
  std::lock_guard<std::mutex> guard(data_m);
  std::string ret = std::to_string(voltage_);
  return ret;
}

std::string MqttInterface::getEnergy() {
  std::lock_guard<std::mutex> guard(data_m);
  std::string ret = std::to_string(energy_);
  return ret;
}
