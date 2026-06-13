# Kubernetes Deployment

Each camera runs as a separate pod. Deploy one instance per camera, each with its own `ConfigMap` and referencing the shared MQTT `Secret`.

## Prerequisites

- A running MQTT broker reachable from within the cluster
- A `PersistentVolume` (or NFS/hostPath mount) for image output, accessible at the path set in `output_dir`
- YOLOv8x weights baked into the Docker image (done automatically by the `Dockerfile`)

---

## Secret — MQTT Credentials

Create once and share across all camera instances.

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mqtt-credentials
  namespace: car-counter
type: Opaque
stringData:
  config.json: |
    {
      "host": "mqtt.example.com",
      "port": 1883,
      "username": "car-counter",
      "password": "changeme"
    }
```

---

## ConfigMap — Per-Camera App Config

One `ConfigMap` per camera instance.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: car-counter-driveway
  namespace: car-counter
data:
  config.yaml: |
    camera_name: driveway
    rtsps_url: rtsps://192.168.1.10/stream

    scan_regions:
      - {x: 100, y: 200, width: 400, height: 300}

    vehicle_classes: [car, truck, bus]
    detection_confidence: 0.4
    stationary_seconds: 3
    iou_threshold: 0.5

    night_enhancement: true
    target_fps: 1

    model_path: yolov8x.pt
    mqtt_topic: car-counter/driveway

    publish_interval_seconds: 5
    mqtt_timeout_seconds: 60

    output_dir: /output
    image_save_cooldown_seconds: 30
```

---

## Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: car-counter-driveway
  namespace: car-counter
spec:
  replicas: 1
  selector:
    matchLabels:
      app: car-counter
      camera: driveway
  template:
    metadata:
      labels:
        app: car-counter
        camera: driveway
    spec:
      containers:
        - name: car-counter
          image: your-registry/car-counter:latest
          env:
            - name: APP_CONFIG_PATH
              value: /config/app/config.yaml
            - name: MQTT_CONFIG_PATH
              value: /config/mqtt/config.json
            - name: LOG_LEVEL
              value: INFO
            - name: LIVENESS_FILE
              value: /tmp/healthy
          ports:
            - containerPort: 9600
              name: metrics
          volumeMounts:
            - name: app-config
              mountPath: /config/app
              readOnly: true
            - name: mqtt-config
              mountPath: /config/mqtt
              readOnly: true
            - name: output
              mountPath: /output
          livenessProbe:
            exec:
              command:
                - /bin/sh
                - -c
                - "test $(( $(date +%s) - $(date -r /tmp/healthy +%s) )) -lt 30"
            initialDelaySeconds: 30
            periodSeconds: 15
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /metrics
              port: 9600
            initialDelaySeconds: 10
            periodSeconds: 10
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          terminationMessagePolicy: FallbackToLogsOnError
      terminationGracePeriodSeconds: 90
      volumes:
        - name: app-config
          configMap:
            name: car-counter-driveway
        - name: mqtt-config
          secret:
            secretName: mqtt-credentials
        - name: output
          persistentVolumeClaim:
            claimName: car-counter-output
```

> **`terminationGracePeriodSeconds`** must exceed `mqtt_timeout_seconds` (default: 60s) to allow the MQTT queue to flush before the pod is killed.

---

## Multiple Cameras

Repeat the `ConfigMap` and `Deployment` above for each camera, changing:
- `ConfigMap` name and `camera_name` / `rtsps_url` / `mqtt_topic` fields
- `Deployment` name and `camera` label
- `ConfigMap` reference in the `volumes` section

All deployments share the same `Secret` and image.

---

## Prometheus Scraping

Add a `ServiceMonitor` (if using the Prometheus Operator) or annotate the pod for scraping:

```yaml
# Pod template annotations (without Prometheus Operator)
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "9600"
  prometheus.io/path: "/metrics"
```

All metrics include a `camera` label, so a single Prometheus job can scrape all instances.
