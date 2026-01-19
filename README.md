# 🎙️ Voice Demo (Voice-Sandwich + VibeVoice)

This repository provides a **GPU-accelerated, containerized voice demo system** built on:

- **Voice-Sandwich** (web + Python backend)
- **VibeVoice** (text-to-speech / voice generation)
- Optional **LLM routing via Groq**
- HTTPS support using self-signed certificates

The project is designed to run **locally on a single machine** using Docker and NVIDIA GPUs.

---

## 🚀 Features

- End-to-end voice demo pipeline
- NVIDIA GPU acceleration (CUDA)
- Modular architecture (Web / Python / Models)
- HTTPS-enabled backend
- Hot-reload friendly for development

---

## 📦 Requirements

### System
- Linux (recommended) or WSL2
- NVIDIA GPU
- NVIDIA Driver + CUDA compatible with Docker


## Clone the Repository
git clone https://github.com/anycookie112/voice-demo.git
cd voice-demo

## Generate HTTPS Certificates
openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem \
  -keyout key.pem \
  -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=<IP_ADDRESS>"

## Build Docker Image
docker build -t voice-demo .


## Run the Container
docker compose up


