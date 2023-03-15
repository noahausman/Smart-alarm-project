#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <MPU9250_WE.h>
#include <Wire.h>
#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>

#define MPU9250_ADDR1 0x68
#define MPU9250_ADDR2 0x69

String user_id = "";

ESP8266WebServer server(80);

MPU9250_WE myMPU9250_1 = MPU9250_WE(MPU9250_ADDR1);
MPU9250_WE myMPU9250_2 = MPU9250_WE(MPU9250_ADDR2);

const char* ssid = "Linoys iPhone"; //Wifi SSID
const char* password = "linoy8629"; //Wifi Password

WiFiClient wifiClient;
const char* serverAt = "http://167.99.78.90:3237/";

void handleRoot() {
  server.send(200, "text/plain", "WeMOS on the Web!");
}

void handleNotFound(){
  String message = "File Not Found\n";
  message += "URI: ";
  message += server.uri();
  message += "\nMethod: ";
  message += (server.method() == HTTP_GET)?"GET":"POST";
  message += "\nArguments: ";
  message += server.args();
  message += "n";
  for (uint8_t i = 0; i < server.args(); i++){
    message += " " + server.argName(i) + ": " + server.arg(i) + "n";
  }
  server.send(404, "text/plain", message);
}

void setup(void){
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  Serial.println("");
  
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED){
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  if (MDNS.begin("esp8266")) {
    Serial.println("MDNS responder started");
  }

  server.on("/", handleRoot);
  server.on("/setId", [](){
    user_id = server.arg(0);
    Serial.println(user_id);
    server.send(200, "text/plain", "OK");
  });
  
  server.onNotFound(handleNotFound);
  server.begin();
  
  Wire.begin();
  // MPU 1
  if(!myMPU9250_1.init()){
    Serial.println("MPU9250 1 does not respond");
  }else{
    Serial.println("MPU9250 1 is connected");
  }
  
  delay(1000);
  myMPU9250_1.autoOffsets();
  myMPU9250_1.enableGyrDLPF();
  myMPU9250_1.setGyrDLPF(MPU9250_DLPF_6);
  myMPU9250_1.setSampleRateDivider(5);
  myMPU9250_1.setGyrRange(MPU9250_GYRO_RANGE_250);
  myMPU9250_1.setAccRange(MPU9250_ACC_RANGE_2G);
  myMPU9250_1.enableAccDLPF(true);
  myMPU9250_1.setAccDLPF(MPU9250_DLPF_6);
  delay(200);

  // MPU 2
  if(!myMPU9250_2.init()){
    Serial.println("MPU9250 2 does not respond");
  }else{
    Serial.println("MPU9250 2 is connected");
  }
  
  delay(1000);
  myMPU9250_2.autoOffsets();
  myMPU9250_2.enableGyrDLPF();
  myMPU9250_2.setGyrDLPF(MPU9250_DLPF_6);
  myMPU9250_2.setSampleRateDivider(5);
  myMPU9250_2.setGyrRange(MPU9250_GYRO_RANGE_250);
  myMPU9250_2.setAccRange(MPU9250_ACC_RANGE_2G);
  myMPU9250_2.enableAccDLPF(true);
  myMPU9250_2.setAccDLPF(MPU9250_DLPF_6);
  delay(200);
}

void loop(void){
  server.handleClient();
  
  Serial.print("Sending...");
  Serial.println();
   
  if (WiFi.status() == WL_CONNECTED){
    HTTPClient http;

    // GET VALUES FROM IMU 1
    xyzFloat gValue1 = myMPU9250_1.getGValues();
    xyzFloat gyr1 = myMPU9250_1.getGyrValues();
    float resultantG1 = myMPU9250_1.getResultantG(gValue1);
    // Acceleration data (x,y,z)
    float data_acc_x_1 = gValue1.x;
    float data_acc_y_1 = gValue1.y;
    float data_acc_z_1 = gValue1.z;
    // Gyroscope data in degrees (x,y,z)
    float data_gyr_x_1 = gyr1.x;
    float data_gyr_y_1 = gyr1.y;
    float data_gyr_z_1 = gyr1.z;

    // GET VALUES FROM IMU 2
    xyzFloat gValue2 = myMPU9250_2.getGValues();
    xyzFloat gyr2 = myMPU9250_2.getGyrValues();
    float resultantG2 = myMPU9250_2.getResultantG(gValue2);
    // Acceleration data (x,y,z)
    float data_acc_x_2 = gValue2.x;
    float data_acc_y_2 = gValue2.y;
    float data_acc_z_2 = gValue2.z;
    // Gyroscope data in degrees (x,y,z)
    float data_gyr_x_2 = gyr2.x;
    float data_gyr_y_2 = gyr2.y;
    float data_gyr_z_2 = gyr2.z;

    String url = serverAt;
    String str = "data?data0=" + user_id + "&data1=" + String(data_acc_x_1) + "&data2=" + String(data_acc_y_1) + "&data3=" + String(data_acc_z_1) + "&data4=" + String(data_gyr_x_1) + "&data5=" + String(data_gyr_y_1) + "&data6=" + String(data_gyr_z_1);
    str += "&data7=" + String(data_acc_x_2) + "&data8=" + String(data_acc_y_2) + "&data9=" + String(data_acc_z_2) + "&data10=" + String(data_gyr_x_2) + "&data11=" + String(data_gyr_y_2) + "&data12=" + String(data_gyr_z_2);
    url += str;

    // send the data and check if it sent successfully 
    http.begin(wifiClient,url);
    int returnCode = http.GET();   // perform an HTTP GET request 
    if (returnCode > 0){
      String ans = http.getString();
      Serial.println(ans);
    }
    
    http.end();   
  } else {
    Serial.println("WiFi disconnected");
  }
  
  delay(100);
}
