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

    fn sinplex(p: vec3f) -> f32 {
        return .5
            + .5 * (sin(p.x)
            * sin(p.y * 1.5)
            * sin(p.z * 2.5));
    }

    fn add_smoke(t: vec4f, p: vec4f) -> vec4f {
        var w = p.a + t.a;
        return select(
            vec4(mix(p.rgb, t.rgb, t.a / w), w),
            t,
            w == 0
        );
    }

    fn simplex(p: vec3f) -> f32 { // Simplex 3D noise
        var s = floor(p + (p.x + p.y + p.z) / 3);
        var x = p - s + (s.x + s.y + s.z) / 6;
        var i = step(x.yzx, x);
        var j = i * (1. - i.zxy);
        i = 1 - i.zxy * (1 - i);
        var a = x - j + 1. / 6;
        var b = x - i + 1. / 3;
        var c = x - .5;
        var w = max(
                .5 - vec4f(
                dot(x, x),
                dot(a, a),
                    dot(b, b),
                    dot(c, c)
                )
            , vec4f(0));
        return
            .5 + 20 * dot(
            vec4f(
                dot(1-2*D(s), x),
                dot(1-2*D(s + j), a),
                dot(1-2*D(s + i), b),
                dot(1-2*D(s + 1), c)
            ),
            w * w * w * w
        );
        // var s = floor(p + (p.x + p.y + p.z) / 3);
        // var x = p - s + (s.x + s.y + s.z) / 6;
        // var i = step(x.yzx, x);
        // var i1 = i * (1. - i.zxy);
        // var i2 = 1 - i.zxy * (1 - i);
        // var x1 = x - i1 + 1. / 6;
        // var x2 = x - i2 + 1. / 3;
        // var x3 = x - .5;
        // var w = max(
        //         .5 - vec4f(
        //         dot(x, x),
        //         dot(x1, x1),
        //             dot(x2, x2),
        //             dot(x3, x3)
        //         )
        //     , vec4f(0));
        // return
        //     .5 + 20 * dot(
        //     vec4f(
        //         dot(2*D(s)-1, x),
        //         dot(2*D(s + i1)-1, x1),
        //         dot(2*D(s + i2)-1, x2),
        //         dot(2*D(s + 1)-1, x3)
        //     ),
        //     w * w * w * w
        // );
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

    struct CameraRay {
        pos: vec3f,
        dir: vec3f,
        rand: vec3f,
    };

    fn computeCameraRay(canvas: vec3u) -> CameraRay {
        var ray: CameraRay;
        ray.rand = D(vec3(vec2f(canvas.xy), u.t));
        let screen = (vec2f(canvas.xy) - 0.5 + ray.rand.xy - vec2f(u.canvasX, u.canvasY) * .5) / u.canvasX;
        let fov = radians(90.0);
        ray.pos = vec3f(u.camPosX, u.camPosY, u.camPosZ);
        let camTarget = vec3f(0, 0, 0);
        let forward = normalize(camTarget - ray.pos);
        let right = normalize(vec3f(forward.y, -forward.x, 0));
        let up = cross(right, forward);
        var uv = screen;
        uv.y = -uv.y;
        let orthoOffset = right * uv.x + up * uv.y - vec3(0, 0, (u.canvasX - u.canvasY) / u.canvasX * .5);
        let perspective_dir = normalize(forward + uv.x * right * tan(fov * 0.5) + uv.y * up * tan(fov * 0.5));
        ray.pos = mix(ray.pos, ray.pos + orthoOffset, u.orthoBlend);
        ray.dir = normalize(mix(perspective_dir, forward, u.orthoBlend));
        return ray;
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
