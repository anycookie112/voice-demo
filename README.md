Voice Demo (Voice Sandwich)

This repository demonstrates a GPU-accelerated voice AI application that integrates real-time speech processing, LLM inference, and a web frontend. The system is containerized using Docker and supports secure HTTPS communication via self-signed SSL certificates.

Features

🎙️ Real-time voice processing pipeline

🤖 LLM integration (Groq supported)

🌐 Web frontend built with PNPM

🐳 Fully containerized (Docker)

🔒 HTTPS support with self-signed certificates

⚡ GPU acceleration (CUDA-enabled Docker)

Prerequisites

Make sure you have the following installed:

Docker (with NVIDIA Container Toolkit)

NVIDIA GPU with compatible drivers

Node.js (for frontend build)

pnpm

OpenSSL

Repository Setup

Clone the repository:

git clone https://github.com/anycookie112/voice-demo.git
cd voice-demo

Build the Docker Image
docker build -t voice-demo .

Generate SSL Certificates

Create a self-signed certificate (replace CN with your server IP):

openssl req -x509 -newkey rsa:4096 -nodes \
  -out cert.pem \
  -keyout key.pem \
  -days 365 \
  -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=YOUR_SERVER_IP"


Example:

CN=172.20.8.110

Build the Web Frontend

Navigate to the web component:

cd voice-sandwich-demo/components/web


Install dependencies and build:

pnpm install --no-strict-peer-dependencies
pnpm build

Run the Application (Docker)

From the root of the repository, run:

docker run --gpus all -it \
  -p 8000:8000 \
  -w /app/voice-sandwich-demo/components/python \
  -e GROQ_API_KEY="YOUR_GROQ_API_KEY" \
  -e LLM_PROVIDER="groq" \
  -v $(pwd)/cert.pem:/app/cert.pem \
  -v $(pwd)/key.pem:/app/key.pem \
  -v /home/robust/models:/home/robust/models \
  -v $(pwd)/voice-sandwich-demo/components/python/src:/app/voice-sandwich-demo/components/python/src \
  -v /home/robust/voice_demo_docket/voice-demo/VibeVoice/demo/voices:/app/voice-demo/VibeVoice/demo/voices \
  voice-demo
