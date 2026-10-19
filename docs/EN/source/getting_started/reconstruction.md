# VGGT and TRELLIS.2

LightX2V provides native BF16 runners for VGGT-1B reconstruction (`recon`) and the TRELLIS.2-4B 512 image-to-3D pipeline (`i23d`). Model graphs live in `lightx2v/models/networks/`; TRELLIS conditioning, sampling, and decoding use the existing encoder/scheduler/decoder directories.

See the repository guide in `scripts/bench/reconstruction/README.md` for pinned asset downloads, CLI commands, environment versions, output formats, and the frozen A/B benchmark.

VGGT uses ordered parallel image preprocessing. TRELLIS selects a neighbor-map algorithm independently of its sampling and attention backends. Linear and attention modules use the existing operator registries; portable sparse operators do not import model or vendor packages.

The integration has been exercised on a single MLU590 per process. CUDA/A100 and MetaX performance, distributed inference, and CPU offload are not validated here. TRELLIS exports geometry GLB and returns texture voxel attributes; baked texture export and cascade resolutions are outside this integration. The source licenses included with the two models also apply.

Historical TRELLIS experiment results used a different sparse-convolution weight index, incomplete decoder BF16 storage, and FP32 BiRefNet deformable convolutions. The new implementation corrects these paths, using a portable sampled-feature/BF16-GEMM backend for deformable convolution. Validate this checkout's A/B outputs and measurements; do not substitute the historical mesh metrics.
