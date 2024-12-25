function buildSourceCommon({ uniformStruct, computeShaders, source }) {
    source.common = /*wgsl*/`
    ${uniformStruct}

    const OPTICAL_DENSITY = 200.0;
    const PI = radians(180);

    fn D(x: vec3f) -> vec3f { // By David Hoskins, MIT License https://www.shadertoy.com/view/4djSRW
        var p = fract(x * vec3(.1031, .1030, .0973));
        p += dot(p, p.yxz + 33.33);
        return fract((p.xxy + p.yxx) * p.zyx);
    }

    fn to_index(id: vec3u) -> u32 {
        return id.x + id.y * u.ux + id.z * u.ux * u.uy;
    }

    fn clamp_to_edge(id: vec3i) -> u32 {
        return to_index(vec3u(clamp(
            id,
            vec3i(0),
            vec3i(vec3u(u.ux, u.uy, u.uz)) - 1)
        ));
    }

    fn trilerp1(
        texture: ptr<storage, array<f32>, read>,
        pos: vec3f
    ) -> f32 {
        let base = vec3i(pos + 0.5) - 1; // To avoid negative rounding
        let frac = fract(pos + 0.5);
        return mix(
            mix(
                mix((*texture)[clamp_to_edge(base + vec3i(0, 0, 0))], (*texture)[clamp_to_edge(base + vec3i(1, 0, 0))], frac.x),
                mix((*texture)[clamp_to_edge(base + vec3i(0, 1, 0))], (*texture)[clamp_to_edge(base + vec3i(1, 1, 0))], frac.x),
                frac.y
            ),
            mix(
                mix((*texture)[clamp_to_edge(base + vec3i(0, 0, 1))], (*texture)[clamp_to_edge(base + vec3i(1, 0, 1))], frac.x),
                mix((*texture)[clamp_to_edge(base + vec3i(0, 1, 1))], (*texture)[clamp_to_edge(base + vec3i(1, 1, 1))], frac.x),
                frac.y
            ),
            frac.z
        );
    };

    fn trilerp4(
        texture: ptr<storage, array<vec4f>, read>,
        pos: vec3f
    ) -> vec4f {
    let base = vec3i(pos + 0.5) - 1; // To avoid negative rounding
    let frac = fract(pos + 0.5);
        return mix(
            mix(
                mix((*texture)[clamp_to_edge(base + vec3i(0, 0, 0))], (*texture)[clamp_to_edge(base + vec3i(1, 0, 0))], frac.x),
                mix((*texture)[clamp_to_edge(base + vec3i(0, 1, 0))], (*texture)[clamp_to_edge(base + vec3i(1, 1, 0))], frac.x),
                frac.y
            ),
            mix(
                mix((*texture)[clamp_to_edge(base + vec3i(0, 0, 1))], (*texture)[clamp_to_edge(base + vec3i(1, 0, 1))], frac.x),
                mix((*texture)[clamp_to_edge(base + vec3i(0, 1, 1))], (*texture)[clamp_to_edge(base + vec3i(1, 1, 1))], frac.x),
                frac.y
            ),
            frac.z
        );
    };

    fn rayBoxIntersect(origin: vec3<f32>, dir: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
        let invDir = 1.0 / dir;
        let tMinVec = (boxMin - origin) * invDir;
        let tMaxVec = (boxMax - origin) * invDir;
        let t1 = min(tMinVec, tMaxVec);
        let t2 = max(tMinVec, tMaxVec);
        let tMin = max(t1.x, max(t1.y, t1.z));
        let tMax = min(t2.x, min(t2.y, t2.z));
        if (tMax < tMin) {
            return vec2<f32>(-1.0, -1.0);
        }
        return vec2<f32>(tMin, tMax);
    }

    fn computeBasis(zDir: vec3<f32>) -> mat3x3<f32> {
        let z = normalize(zDir);
        let x = normalize(vec3<f32>(z.y, -z.x, 0.0));
        let y = normalize(cross(z, x));
        return mat3x3<f32>(x, y, z);
    }

    fn computeRayDirection(
        cameraMatrix: mat3x3<f32>,
        vFov: f32,
        screenPos: vec2<f32>
    ) -> vec3<f32> {
        let tanHalfFov = tan(0.5 * vFov);
        let rayDirCameraSpace = normalize(vec3<f32>(
            screenPos.x * tanHalfFov,
            screenPos.y * tanHalfFov,
            1.0
        ));
        let rayDirWorldSpace = cameraMatrix * rayDirCameraSpace;
        return normalize(rayDirWorldSpace);  // Return normalized ray direction
    }
    `;

    source.pressure = /*wgsl*/`
    fn to_index_dim(id: vec3u, dim: u32) -> u32 {
        return id.x + id.y * dim + id.z * dim * dim;
    }

    fn clamp_to_edge_dim(id: vec3i, dim: u32) -> u32 {
        return to_index_dim(vec3u(clamp(
            id,
            vec3i(0),
            vec3i(vec3u(dim - 1))
        )), dim);
    }
    `;
}
