#include "DHT.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <driver/i2s.h>
#include <math.h>

/* ========= WIFI ========= */
const char* ssid = "010";
const char* password = "aezakmii";

/* ========= SERVER URLS ========= */
const char* tempServerUrl  = "http://172.20.10.5:8080/sensors/temperature/4";
const char* noiseServerUrl = "http://172.20.10.5:8080/sensors/noisedetector/4";

/* ========= DHT ========= */
#define DHTPIN 7
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

/* ========= I2S MIC ========= */
#define I2S_BCLK  4
#define I2S_WS    5
#define I2S_SD    2

#define SAMPLE_RATE 16000
#define SAMPLES     256

int32_t samples[SAMPLES];
double noiseFloor = 1.0;
double smooth_dB = 0;

/* ========= TIMERS ========= */
unsigned long lastTempSend = 0;
unsigned long lastNoiseSend = 0;

const unsigned long TEMP_INTERVAL  = 5000; // 5 sec
const unsigned long NOISE_INTERVAL = 200;  // 200 ms

/* ========= CALIBRATION ========= */
double measureNoiseFloor() {
  const int rounds = 20;
  double rmsSum = 0;

  for (int r = 0; r < rounds; r++) {
    size_t bytes_read;
    i2s_read(I2S_NUM_0, samples, sizeof(samples), &bytes_read, portMAX_DELAY);

    double sum = 0;
    for (int i = 0; i < SAMPLES; i++) {
      int32_t s = samples[i] >> 8;
      sum += (double)s * s;
    }

    rmsSum += sqrt(sum / SAMPLES);
    delay(20);
  }

  return rmsSum / rounds;
}

void setup() {
  Serial.begin(115200);

  /* WIFI */
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");

  /* DHT */
  dht.begin();

  /* I2S */
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = SAMPLES,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_BCLK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);

  /* Noise floor */
  Serial.println("Calibrating noise...");
  delay(2000);
  noiseFloor = measureNoiseFloor();
  Serial.print("Noise floor = ");
  Serial.println(noiseFloor);
}

/* ================================================= */

void loop() {

  unsigned long now = millis();

  /* ============ FAST NOISE LOOP ============ */

  if (now - lastNoiseSend >= NOISE_INTERVAL) {

    lastNoiseSend = now;

    size_t bytes_read;
    i2s_read(I2S_NUM_0, samples, sizeof(samples), &bytes_read, portMAX_DELAY);

    double sum = 0;
    for (int i = 0; i < SAMPLES; i++) {
      int32_t s = samples[i] >> 8;
      sum += (double)s * s;
    }

    double rms = sqrt(sum / SAMPLES);

    double rel_dB = 20.0 * log10(rms / noiseFloor);
    rel_dB *= 2.5;
    if (rel_dB < 0) rel_dB = 0;

    smooth_dB = 0.8 * smooth_dB + 0.2 * rel_dB;

    Serial.print("Sound: ");
    Serial.print(smooth_dB, 1);
    Serial.println(" dB");

    HTTPClient http;
    http.begin(noiseServerUrl);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"noise\":" + String(smooth_dB, 1) + "}";

    http.POST(payload);
    http.end();
  }

  /* ============ SLOW DHT LOOP ============ */

  if (now - lastTempSend >= TEMP_INTERVAL) {

    lastTempSend = now;

    float t = dht.readTemperature();
    float h = dht.readHumidity();

    if (isnan(t) || isnan(h)) {
      Serial.println("DHT failed");
      return;
    }

    HTTPClient http;
    http.begin(tempServerUrl);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"temperature\":" + String(t) +
                     ",\"humidity\":" + String(h) + "}";

    http.POST(payload);

    Serial.print("Temp sent: ");
    Serial.println(payload);

    http.end();
  }
}
