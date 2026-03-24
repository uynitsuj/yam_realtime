"""MuJoCo → Viser / Three.js mesh helpers
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Geom color helpers
# ---------------------------------------------------------------------------


def _get_geom_rgba(model: Any, geom_id: int) -> np.ndarray:
    matid = model.geom_matid[geom_id]
    if matid >= 0:
        return model.mat_rgba[matid].copy()
    rgba = model.geom_rgba[geom_id].copy()
    if np.all(rgba == 0):
        rgba = np.array([0.5, 0.5, 0.5, 1.0])
    return rgba


# ---------------------------------------------------------------------------
# MuJoCo geom → trimesh
# ---------------------------------------------------------------------------


def _create_primitive_mesh(model: Any, geom_id: int) -> Any:
    """Convert a MuJoCo primitive geom to trimesh (with texture support)."""
    import trimesh
    import trimesh.visual
    import trimesh.visual.material
    from PIL import Image

    size = model.geom_size[geom_id]
    geom_type = model.geom_type[geom_id]
    rgba = _get_geom_rgba(model, geom_id)
    rgba_uint8 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(radius=size[0], subdivisions=2)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2.0 * size)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        mesh = trimesh.creation.capsule(radius=size[0], height=2.0 * size[1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1])
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        mesh.apply_scale(size)
    elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        plane_x = 2.0 * size[0] if size[0] > 0 else 20.0
        plane_y = 2.0 * size[1] if size[1] > 0 else 20.0
        mesh = trimesh.creation.box((plane_x, plane_y, 0.001))
    else:
        return None

    # Check for textured material
    matid = model.geom_matid[geom_id]
    has_texture = False
    if matid >= 0 and matid < model.nmat:
        texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
        if texid < 0:
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
        if texid >= 0 and texid < model.ntex:
            has_texture = True
            mat_rgba = model.mat_rgba[matid]
            tex_w = model.tex_width[texid]
            tex_h = model.tex_height[texid]
            tex_nc = model.tex_nchannel[texid]
            tex_adr = model.tex_adr[texid]
            tex_data = model.tex_data[tex_adr:tex_adr + tex_w * tex_h * tex_nc]
            texrepeat = model.mat_texrepeat[matid]
            image: Any = None
            if tex_nc == 3:
                image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 3).astype(np.uint8)), mode="RGB")
            elif tex_nc == 4:
                image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 4).astype(np.uint8)), mode="RGBA")
            elif tex_nc == 1:
                image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w).astype(np.uint8)), mode="L")
            else:
                has_texture = False

    if has_texture and image is not None:
        verts = mesh.vertices
        uv = np.zeros((len(verts), 2))
        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        bb_range = bb_max - bb_min
        bb_range[bb_range == 0] = 1
        uv[:, 0] = (verts[:, 0] - bb_min[0]) / bb_range[0] * texrepeat[0]
        uv[:, 1] = (verts[:, 1] - bb_min[1]) / bb_range[1] * texrepeat[1]
        material = trimesh.visual.material.PBRMaterial(
            baseColorFactor=mat_rgba,
            baseColorTexture=image,
            metallicFactor=0.0,
            roughnessFactor=1.0,
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    else:
        vertex_colors = np.tile(rgba_uint8, (len(mesh.vertices), 1))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)

    return mesh


def _mujoco_mesh_to_trimesh(model: Any, geom_id: int) -> Any:
    """Convert a MuJoCo mesh geom to trimesh (with texture support)."""
    import trimesh
    import trimesh.visual
    import trimesh.visual.material
    from PIL import Image

    mesh_id = model.geom_dataid[geom_id]
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_count = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_start:vert_start + vert_count]
    faces = model.mesh_face[face_start:face_start + face_count]

    texcoord_adr = model.mesh_texcoordadr[mesh_id]
    texcoord_num = model.mesh_texcoordnum[mesh_id]

    if texcoord_num > 0:
        texcoords = model.mesh_texcoord[texcoord_adr:texcoord_adr + texcoord_num]
        face_texcoord_idx = model.mesh_facetexcoord[face_start:face_start + face_count]
        new_vertices = vertices[faces.flatten()]
        new_uvs = texcoords[face_texcoord_idx.flatten()]
        new_faces = np.arange(face_count * 3).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

        matid = model.geom_matid[geom_id]
        if matid >= 0 and matid < model.nmat:
            rgba = model.mat_rgba[matid]
            texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGB)])
            if texid < 0:
                texid = int(model.mat_texid[matid, int(mujoco.mjtTextureRole.mjTEXROLE_RGBA)])
            if texid >= 0 and texid < model.ntex:
                tex_w = model.tex_width[texid]
                tex_h = model.tex_height[texid]
                tex_nc = model.tex_nchannel[texid]
                tex_adr = model.tex_adr[texid]
                tex_data = model.tex_data[tex_adr:tex_adr + tex_w * tex_h * tex_nc]
                image = None
                if tex_nc == 1:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w)).astype(np.uint8), mode="L")
                elif tex_nc == 3:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 3)).astype(np.uint8), mode="RGB")
                elif tex_nc == 4:
                    image = Image.fromarray(np.flipud(tex_data.reshape(tex_h, tex_w, 4)).astype(np.uint8), mode="RGBA")
                if image is not None:
                    material = trimesh.visual.material.PBRMaterial(
                        baseColorFactor=rgba,
                        baseColorTexture=image,
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                    mesh.visual = trimesh.visual.TextureVisuals(uv=new_uvs, material=material)
                else:
                    rgba_255 = (rgba * 255).astype(np.uint8)
                    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
            else:
                rgba_255 = (rgba * 255).astype(np.uint8)
                mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
        else:
            rgba = _get_geom_rgba(model, geom_id)
            rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(rgba_255, (len(new_vertices), 1)))
    else:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        rgba = _get_geom_rgba(model, geom_id)
        rgba_255 = (np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(rgba_255, (len(mesh.vertices), 1)))

    return mesh


# ---------------------------------------------------------------------------
# Viser scene helpers
# ---------------------------------------------------------------------------


def _merge_geoms(model: Any, geom_ids: list[int]) -> Any:
    """Merge multiple geoms into a single trimesh, applying local transforms."""
    import trimesh
    import viser.transforms as vtf

    meshes = []
    for geom_id in geom_ids:
        geom_type = model.geom_type[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = _mujoco_mesh_to_trimesh(model, geom_id)
        else:
            mesh = _create_primitive_mesh(model, geom_id)
        if mesh is None:
            continue
        pos = model.geom_pos[geom_id]
        quat = model.geom_quat[geom_id]
        transform = np.eye(4)
        transform[:3, :3] = vtf.SO3(quat).as_matrix()
        transform[:3, 3] = pos
        mesh.apply_transform(transform)
        meshes.append(mesh)

    if not meshes:
        import trimesh as _trimesh
        return _trimesh.Trimesh()
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _is_fixed_body(model: Any, body_id: int) -> bool:
    """Check if a body is fixed (welded to world, not mocap)."""
    is_weld = model.body_weldid[body_id] == 0
    root_id = model.body_rootid[body_id]
    root_is_mocap = model.body_mocapid[root_id] >= 0
    return is_weld and not root_is_mocap


def _get_body_name(model: Any, body_id: int) -> str:
    from mujoco import mj_id2name, mjtObj
    name = mj_id2name(model, mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


def configure_default_camera(server: Any) -> None:
    """Set the initial Viser camera to an operator's-eye view."""
    @server.on_client_connect
    def _set_camera(client: Any) -> None:
        client.camera.position = (-0.4, 0.0, 1.3)
        client.camera.look_at = (0.45, 0.0, 0.85)
        client.camera.up_direction = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# VR streamer helpers
# ---------------------------------------------------------------------------


def _patch_glb_add_material(glb_path: str) -> None:
    """Patch GLB: add default PBR material for vertex-colored meshes that lack one."""
    import json as _json

    with open(glb_path, "rb") as f:
        f.read(12)  # header
        chunk0_len = struct.unpack("<I", f.read(4))[0]
        f.read(4)   # chunk type
        json_bytes = f.read(chunk0_len)
        rest = f.read()

    gltf = _json.loads(json_bytes)
    needs_patch = any(
        "material" not in prim or prim["material"] is None
        for mesh in gltf.get("meshes", [])
        for prim in mesh.get("primitives", [])
    )
    if not needs_patch:
        return

    if "materials" not in gltf:
        gltf["materials"] = []
    default_idx = len(gltf["materials"])
    gltf["materials"].append({
        "pbrMetallicRoughness": {
            "baseColorFactor": [1, 1, 1, 1],
            "metallicFactor": 0.1,
            "roughnessFactor": 0.7,
        }
    })
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "material" not in prim or prim["material"] is None:
                prim["material"] = default_idx

    new_json = _json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(new_json) % 4 != 0:
        new_json += b" "

    with open(glb_path, "wb") as f:
        f.write(b"glTF")
        f.write(struct.pack("<II", 2, 12 + 8 + len(new_json) + len(rest)))
        f.write(struct.pack("<I", len(new_json)))
        f.write(b"JSON")
        f.write(new_json)
        f.write(rest)


def export_body_glbs(model: Any, output_dir: Path) -> dict:
    """Export each body's merged mesh as a GLB. Returns body-info dict."""
    import trimesh
    import trimesh.visual
    from scipy.spatial.transform import Rotation

    output_dir.mkdir(parents=True, exist_ok=True)
    visible_groups = {0, 1, 2}
    body_geoms: dict[int, list[int]] = {}
    for i in range(model.ngeom):
        if int(model.geom_group[i]) not in visible_groups:
            continue
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        body_geoms.setdefault(int(model.geom_bodyid[i]), []).append(i)

    bodies: dict[int, dict] = {}
    for body_id, geom_ids in body_geoms.items():
        meshes = []
        for gid in geom_ids:
            geom_type = model.geom_type[gid]
            if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            mesh = (_mujoco_mesh_to_trimesh(model, gid)
                    if geom_type == mujoco.mjtGeom.mjGEOM_MESH
                    else _create_primitive_mesh(model, gid))
            if mesh is None:
                continue
            qw = model.geom_quat[gid]
            rot = Rotation.from_quat([qw[1], qw[2], qw[3], qw[0]]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = model.geom_pos[gid]
            mesh.apply_transform(T)
            # Z-up → Y-up for Three.js
            verts = np.array(mesh.vertices)
            new_verts = np.empty_like(verts)
            new_verts[:, 0] = verts[:, 0]
            new_verts[:, 1] = verts[:, 2]
            new_verts[:, 2] = -verts[:, 1]
            mesh.vertices = new_verts
            meshes.append(mesh)

        if not meshes:
            continue

        is_fixed = (model.body_weldid[body_id] == 0
                    and model.body_mocapid[model.body_rootid[body_id]] < 0)
        glb_path = output_dir / f"body_{body_id}.glb"
        has_texture = any(isinstance(m.visual, trimesh.visual.TextureVisuals) for m in meshes)
        if has_texture or len(meshes) > 1:
            trimesh.Scene(meshes).export(str(glb_path), file_type="glb")
        else:
            meshes[0].export(str(glb_path), file_type="glb")
        _patch_glb_add_material(str(glb_path))
        bodies[body_id] = {"file": f"body_{body_id}.glb", "is_fixed": is_fixed}

    return bodies


# ---------------------------------------------------------------------------
# VR HTML app (Three.js WebXR client)
# ---------------------------------------------------------------------------

VR_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>XDoF VR Teleop</title>
<style>
  body { margin: 0; overflow: hidden; background: #1a1a2e; }
  #info { position: absolute; top: 10px; width: 100%; text-align: center;
          color: #fff; font-family: monospace; font-size: 14px; z-index: 1; }
  #vr-btn { position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
            padding: 12px 24px; font-size: 18px; cursor: pointer; z-index: 1;
            background: #4CAF50; color: white; border: none; border-radius: 8px; }
</style>
</head>
<body>
<div id="info">Connecting...</div>
<button id="vr-btn" style="display:none">Enter VR</button>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const ambient = new THREE.AmbientLight(0xffffff, 0.35);
scene.add(ambient);
const dir = new THREE.DirectionalLight(0xffffff, 1.0);
dir.position.set(2, 4, 3);
scene.add(dir);
const dir2 = new THREE.DirectionalLight(0xffffff, 0.5);
dir2.position.set(-2, 3, -1);
scene.add(dir2);

const groundGeo = new THREE.PlaneGeometry(10, 10);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2a3e, roughness: 0.9 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.001;
scene.add(ground);

const grid = new THREE.GridHelper(4, 20, 0x444466, 0x333355);
scene.add(grid);

const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0.5, 1.5, 1.5);
camera.lookAt(0, 0.8, 0);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.xr.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.8, 0);
controls.update();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

const sceneRoot = new THREE.Group();
sceneRoot.rotation.y = Math.PI / 2;
scene.add(sceneRoot);

const info = document.getElementById('info');
const loader = new GLTFLoader();
const bodyMeshes = {};

async function loadScene() {
  info.textContent = 'Loading scene...';
  const resp = await fetch('/api/bodies');
  const bodyInfo = await resp.json();
  const cacheBust = '?t=' + Date.now();
  const total = Object.keys(bodyInfo).length;
  let loaded = 0;
  for (const [bid, bdata] of Object.entries(bodyInfo)) {
    try {
      const gltf = await new Promise((resolve, reject) => {
        loader.load('/meshes/' + bdata.file + cacheBust, resolve, undefined, reject);
      });
      const obj = gltf.scene;
      obj.traverse((child) => {
        if (child.isMesh) {
          child.geometry.computeVertexNormals();
          if (child.material.map) {
            child.material.side = THREE.DoubleSide;
          } else if (child.geometry.attributes.color) {
            child.material = new THREE.MeshStandardMaterial({
              vertexColors: true, side: THREE.DoubleSide, roughness: 0.5, metalness: 0.15,
            });
          } else {
            child.material.side = THREE.DoubleSide;
          }
        }
      });
      sceneRoot.add(obj);
      bodyMeshes[bid] = obj;
      loaded++;
    } catch (e) { console.error(`Failed to load body ${bid}:`, e); }
    info.textContent = `Loading meshes: ${loaded}/${total}`;
  }
  info.textContent = `Loaded ${loaded}/${total} bodies. Connecting...`;
  connectWS();
}

let ws = null;
let frameCount = 0;

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { info.textContent = 'Connected!'; };
  ws.onmessage = (event) => {
    const buf = new Float32Array(event.data);
    const n = buf.length / 8;
    for (let i = 0; i < n; i++) {
      const o = i * 8;
      const bid = Math.round(buf[o]).toString();
      const obj = bodyMeshes[bid];
      if (obj) {
        obj.position.set(buf[o+1], buf[o+2], buf[o+3]);
        obj.quaternion.set(buf[o+4], buf[o+5], buf[o+6], buf[o+7]);
      }
    }
    if (++frameCount === 1) info.textContent = '';
  };
  ws.onclose = () => { info.textContent = 'Disconnected. Reconnecting...'; setTimeout(connectWS, 1000); };
}

renderer.setAnimationLoop(() => { renderer.render(scene, camera); });
loadScene();
</script>
</body>
</html>
"""
