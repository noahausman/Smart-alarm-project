#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>

#define BUZZER D4

const char* ssid = "Linoys iPhone"; //Wifi SSID
const char* password = "linoy8629"; //Wifi Password

// Define WiFi Client
WiFiClient wifiClient;
const char* url = "http://167.99.78.90:3237/buzzer_state";

void setup(void){
  Serial.begin(115200);
  pinMode(BUZZER, OUTPUT);
  digitalWrite(BUZZER, LOW);
  WiFi.begin(ssid, password);
  Serial.println("");
   
  // Wait for connection
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void loop(void){
  Serial.print("Sending...");
  if (WiFi.status() == WL_CONNECTED){
    if (http_buzzer_on()){
      digitalWrite(BUZZER, HIGH);
    } else {
      digitalWrite(BUZZER, LOW);
    } 
      
  } else {
    Serial.println("WiFi disconnected");
  }
  
  delay(500); 
}

bool http_buzzer_on(void){
    // buzzer asks the server when to be on
    HTTPClient http;
    http.begin(wifiClient,url);
    int returnCode = http.GET();   // perform an HTTP GET request
    bool changeBuzzerToHigh = false;
    if (returnCode > 0){
      String stringFromServer = http.getString();
      if (stringFromServer == "True"){
        changeBuzzerToHigh = true;
      }
    }
    
    http.end();
    
    return changeBuzzerToHigh;
}
