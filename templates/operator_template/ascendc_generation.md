# Ascend C Operator Generation Guide for {{ operator_name }}

## Using Cannbot Skills
1. Load the `ascendc-kernel-develop-workflow` skill
2. Follow the 7-stage workflow: environment → design → implement → test → debug → summary → document
3. Use `ascendc-direct-invoke-template` for the project skeleton

## Template Structure
```
operators/{{ operator_name }}/ascendc/
├── src/
│   ├── {{ operator_name }}_kernel.asc
│   ├── {{ operator_name }}_host.asc
│   ├── {{ operator_name }}_tiling.h
│   └── data_utils.h
├── CMakeLists.txt
├── artifact_manifest.json
├── build/
│   └── output/
└── scripts/
```

## Key Parameters
- NPU arch: dav-2201 (Ascend 910B)
- TILE_LENGTH: Choose based on operator (512-8192 for element-wise)
- Kernel type: Typically KERNEL_AIVEC for vector ops
- Kernels per call: Typically 1 for simple ops
