#!/bin/bash
# Optional: generate official *_pb2.py when grpcio-tools is installed.
set -e
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
python -m grpc_tools.protoc \
  -I "${ROOT}/lightx2v/server/ws/proto" \
  --python_out="${ROOT}/lightx2v/server/ws/pb" \
  "${ROOT}/lightx2v/server/ws/proto/lightx2v_v1.proto"
echo "generated ${ROOT}/lightx2v/server/ws/pb/lightx2v_v1_pb2.py"
