# Cloud Command Parser — Lightweight Model Distillation

## 1. Project Overview

This project explores the distillation of a larger language model into a smaller model for understanding simple natural-language commands.

The aim is to convert commands such as "Set an alarm for 7 AM tomorrow" into a structured JSON response containing an intent and its related slots.

The student model used in this work is EleutherAI Pythia-70M. Teacher-generated responses are used as training targets, and the student is trained on CPU.

## 2. Project Structure

```text
Cloud_Computing_Assignment/
|
|-- api/                  FastAPI application and Kubernetes manifest
|-- cluster/              k3d cluster configuration
|-- data/                 Command pool, validation data and schema
|-- models/               Local model checkpoints (not tracked by Git)
|-- network/              NACL configuration and sample packets
|-- pipeline/             Teacher labelling script
|-- results/              Evaluation, training and load-test evidence
|-- tools/                Evaluation, scoring, load testing and NACL tools
|-- training/              Training script and configuration
|
|-- Dockerfile
|-- .dockerignore
|-- .gitignore
|-- README.md
|-- seaweedfs-values.yaml
|-- s3-export.yaml
|-- s3-export-live.yaml
|-- s3-verify.yaml

## 3. Dataset and Teacher Labelling

The project uses natural-language commands paired with four target fields:

- `intent`
- `date`
- `time`
- `place`

A local Phi-3.5 3.8B model was used as the teacher through Ollama. The teacher-labelling pipeline generated training examples in two shards.

The combined dataset contains 2,972 structurally valid, unique records. This confirms structural validity, not that every teacher-generated label is semantically correct.

The validation dataset contains 100 examples and is kept separate from student training.

## 4. Student Model Training

The student model is `EleutherAI/pythia-70m`.

Training was performed on CPU using sequence-level targets. The prompt tokens are masked during loss calculation so that the model learns from the target JSON response.

The main training configuration is stored in `training/config.yaml`.

An experimental run was performed using all 2,972 labelled records for three epochs. The recorded training losses were:

| Epoch | Training Loss |
|---|---:|
| 1 | 0.4563 |
| 2 | 0.0897 |
| 3 | 0.0559 |

**Important:** The 2,972-record run is experimental. It does not satisfy the assignment's 1,500-record training limit and should not be presented as the compliant final training run.

## 5. Evaluation

The student model was evaluated on the 100-record validation set.

| Metric | Result |
|---|---:|
| Valid JSON | 100% |
| Intent accuracy | 77% |
| Date accuracy | 85% |
| Time accuracy | 84% |
| Place accuracy | 86% |
| Exact full-record accuracy | 50% |

The results show that valid JSON generation is not the same as correctly understanding every command. The exact full-record accuracy is lower because all four fields must be correct for a prediction to count as an exact match.

Evaluation outputs are available under `results/evaluation/`.

## 6. API Service

A FastAPI application is available in `api/app.py`.

The service provides:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Checks service and model status |
| POST | `/parse` | Converts a natural-language command into JSON |

Example request:

```json
{
  "text": "Set an alarm for 7 AM tomorrow"
}
```

The response headers include `X-Served-By` and `X-Model-Version`.

## 7. Docker Deployment

Build the Docker image:

docker build -t command-parser-api:1.0 .

Run the container:

docker run --rm -p 8080:8080 command-parser-api:1.0

The current Dockerfile copies the model checkpoint into the image. Retrieving model artefacts from external storage at startup remains an improvement.

## 8. Kubernetes Deployment

The Kubernetes manifest is available at `api/kubernetes.yaml`.

The application was deployed to a local k3d cluster in the `pipeline` namespace. The current configuration uses one replica and a ClusterIP service.

To access the service locally:

kubectl port-forward -n pipeline service/command-parser-api 8083:80

The API is then available at `http://127.0.0.1:8083`.

Ingress, autoscaling, multiple replicas, rolling-update testing and rollback testing remain incomplete.
## 9. Load Testing

Locust was used to test the API with four concurrent users for approximately two minutes.

Recorded results:

| Metric | Result |
|---|---:|
| Average response time | 3,605 ms |
| Median response time | 3,500 ms |
| 95th percentile | 4,400 ms |
| Requests per second | 1.09 |
| Failed requests | 0 |

Load-test evidence is stored in `results/`.

The statistics CSV and request-header count differ by one request. This discrepancy is retained rather than silently corrected.

These measurements are from a local single-replica environment and should not be treated as cloud-scale performance results.

## 10. Network Access Control

The network-related files are:

- `network/nacl.json`
- `network/packets_sample.jsonl`
- `tools/nacl_sim.py`

The simulator checks sample inbound and outbound packet rules. The current configuration demonstrates basic allow and deny decisions, but it does not yet implement the complete subnet allocation and NACL policy required by the assignment.

## 11. Project Limitations and Future Improvements

The current implementation demonstrates the basic teacher-student training workflow and local API deployment. However, several assignment requirements are not yet fully implemented.

The main areas for improvement are:

1. Run final student training using only the first 1,500 cleaned training records.
2. Complete dataset cleaning checks and save the required checksum.
3. Evaluate the teacher, untrained student and trained student on the same validation set.
4. Move model artefacts out of the Docker image and retrieve them from object storage at startup.
5. Implement the required health and readiness endpoints and response validation.
6. Complete Kubernetes ingress, autoscaling, rollout, rollback and recovery tests.
7. Complete subnet allocation and the required stateless NACL rules.
8. Add automated tests and a CI workflow.
9. Record resource usage and complete the required cost and SLO analysis.

These items are listed as limitations because they have not yet been demonstrated by the current project evidence.

## 12. Repository

GitHub: https://github.com/Abhimanyu-Space/cloud-command-parser-distillation
