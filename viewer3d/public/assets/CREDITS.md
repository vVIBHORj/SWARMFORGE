# SwarmForge 3D Tactical Viewer — Asset Credits & Licenses

All external assets used in this application are sourced under permissive open-source licenses (MIT, Apache 2.0, CC0).

## 1. 3D glTF Models
- **Files**: `building-small-a.glb`, `building-small-b.glb`, `building-small-c.glb`, `building-small-d.glb`, `building-garage.glb`
  - **Source**: KenneyNL / Starter-Kit-City-Builder (https://github.com/KenneyNL/Starter-Kit-City-Builder)
  - **Author**: Kenney (Kenney.nl)
  - **License**: MIT License
  - **Usage**: Defended High-Value Asset (HVA) compounds, facility structures, and ground installations.

## 2. PBR & FX Textures
- **Files**:
  - `grasslight-big.jpg`, `grasslight-big-nm.jpg` (Terrain diffuse and normal maps)
  - `cloud.png` (Missile plume smoke and explosion dust particles)
  - `spark1.png` (Thruster glow and energy flash particles)
  - `lensflare0.png` (Engine afterburner and explosion core flares)
  - **Source**: Three.js Examples Repository (https://github.com/mrdoob/three.js/tree/r165/examples/textures)
  - **Author**: mrdoob & Three.js Contributors
  - **License**: MIT License
  - **Usage**: Tactical terrain surface, missile smoke ribbons, thruster exhausts, and intercept bursts.

## 3. Libraries & Decoders
- **Files**: `draco_decoder.js`, `draco_decoder.wasm`, `draco_wasm_wrapper.js`
  - **Source**: Google Draco 3D Data Compression (https://github.com/google/draco) via Three.js
  - **License**: Apache License 2.0
  - **Usage**: Accelerated client-side decoding of compressed 3D geometry.

## 4. Procedural Fallback Architectures
- When models or textures are missing, the viewer uses high-fidelity procedural geometry builders:
  - **Interceptors**: Delta-Wing Loitering Interceptor UAV with ogive nose, swept wings, and canted V-tails.
  - **Scouts**: High-Aspect-Ratio AWACS surveillance airframe with twin boom tails and dorsal rotodome radar dish.
  - **Missiles**: Hypersonic threat with sharp nosecone, cruciform strakes, control fins, and animated exhaust plumes.
  - **Decoys**: Compact EW munition with flare pods and corner reflectors.
  - **HVA Compounds**: Procedural command towers, hangars, cooling towers, and radar dishes with procedural window emissive textures.

