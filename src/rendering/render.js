function buildShadersRender({device, presentationFormat, computeShaders, source}) {
    source.commonBoxRayIntersection = /*wgsl*/`
    let random = D(vec3(vec2f(global_id.xy), u.t));
    let screen = (vec2f(global_id.xy) - 0.5 + random.xy - vec2f(u.canvasX, u.canvasY) * .5) / u.canvasX;

    let fov = radians(90.0);

    // let camPos = vec3f(sin(u.t * .25), cos(u.t * .25), 0.25) * 2.5;
    let camPos = vec3f(u.camPosX, u.camPosY, u.camPosZ);
    let camRot = computeBasis(vec3(0, 0, 0) - camPos);
    let camDir = computeRayDirection(camRot, fov, screen);

    let boxMin = vec3f(-.5);
    let boxMax = vec3f(.5);

    let invDir = 1.0 / camDir;
    let tMinVec = (boxMin - camPos) * invDir;
    let tMaxVec = (boxMax - camPos) * invDir;
    let t1 = min(tMinVec, tMaxVec);
    let t2 = max(tMinVec, tMaxVec);
    let tMin = max(t1.x, max(t1.y, t1.z));
    let tMax = min(t2.x, min(t2.y, t2.z));

    var color = vec3f(0);
    if (tMax >= tMin) { // Hit
        let hit = tMax * camDir + camPos; // Take the furtest hit

        var uvw = (hit - boxMin) / (boxMax - boxMin);

        uvw *= 16; // uvw *= vec3f(u.x, u.y, u.z);
        // uvw /= 4.0;
        color = vec3f(1, 0, 0);

        const lightGray = vec3f(.045);
        const darkGray = vec3f(.02);

        if (tMax == t2.x) { // x plane
            // normal = vec3<f32>(sign(dir.x), 0.0, 0.0);
            color = select(darkGray, lightGray, i32(uvw.y) % 2 == i32(uvw.z) % 2);
        } else if (tMax == t2.y) { // y plane
            // normal = vec3<f32>(0.0, sign(dir.y), 0.0);
            color = select(darkGray, lightGray, i32(uvw.z) % 2 == i32(uvw.x) % 2);
        } else if (tMax == t2.z) { // z plane
            // normal = vec3<f32>(0.0, 0.0, sign(dir.z));
            color = select(darkGray, lightGray, i32(uvw.x) % 2 == i32(uvw.y) % 2);
        }

        let volumeColor = marchRay(camPos, camDir, max(0.0, tMin) + u.stepLength * random.z, tMax);

        color = mix(color, volumeColor.rgb, volumeColor.a);
    }
    `;

    computeShaders.renderDefault = new ComputeShader("renderDefault", device, /*wgsl*/`
    ${source.common}

    fn marchRay(origin: vec3<f32>, direction: vec3<f32>, start: f32, stop: f32) -> vec4f {
        var t: f32 = start;
        var accumulated = vec4f(0);
        while (t < stop && accumulated.a < 0.995) {
            let pos = origin + direction * t;
            let volumePos = (pos + .5) * vec3f(u.x, u.y, u.z) + 0.5;
            var s = trilerp4(&lighting, volumePos);
            // var l = trilerp4(&lighting, volumePos);
            s.a *= u.stepLength * OPTICAL_DENSITY;
            s = vec4f(s.rgb * s.a , s.a); // Weight the added sample by density
            accumulated += s * (1.0 - accumulated.a);
            t += u.stepLength;
        }
        return accumulated;
    }

    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var tex: texture_storage_2d<${presentationFormat}, write>;
    // @group(0) @binding(2) var<storage, read> smoke : array<vec4f>;
    @group(0) @binding(2) var<storage, read> lighting : array<vec4f>;
    // @group(0) @binding(4) var<storage, read> temperature : array<f32>;
    @compute @workgroup_size(8,8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        ${source.commonBoxRayIntersection}
        color = pow(color, vec3f(1 / 1.8)) + random / 255.;
        textureStore(tex, vec2<i32>(i32(global_id.x), i32(global_id.y)), vec4f(color, 1));
    }`);

    computeShaders.renderFuel = new ComputeShader("renderFuel", device, /*wgsl*/`
    ${source.common}

    fn marchRay(origin: vec3<f32>, direction: vec3<f32>, start: f32, stop: f32) -> vec4f {
        let stepSize = 1. / u.x * .66; // TODO: Make adjustable
        var t: f32 = start;
        var accumulated = vec4f(0);
        while (t < stop && accumulated.a < 0.999) {
            let pos = origin + direction * t;
            let volumePos = (pos + .5) * vec3f(u.x, u.y, u.z) + 0.5;
            var s = trilerp4(&velocity, volumePos);
            var l = trilerp4(&lighting, volumePos);
            s.a *= stepSize * OPTICAL_DENSITY;
            s = vec4f(l.rgb * s.a , s.a); // Weight the added sample by density
            accumulated += s * (1.0 - accumulated.a);
            t += stepSize;
        }
        return accumulated;
    }

    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var tex: texture_storage_2d<${presentationFormat}, write>;
    @group(0) @binding(2) var<storage, read> velocity : array<vec4f>;
    @group(0) @binding(3) var<storage, read> lighting : array<vec4f>;
    @compute @workgroup_size(8,8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        ${source.commonBoxRayIntersection}
        color = pow(color, vec3f(1 / 1.8));
        textureStore(tex, vec2<i32>(i32(global_id.x), i32(global_id.y)), vec4f(color, 1));
    }`);


    computeShaders.renderFuelTemperature = new ComputeShader("renderFuelTemperature", device, /*wgsl*/`
    ${source.common}
    ${source.blackbody}

    fn marchRay(origin: vec3<f32>, direction: vec3<f32>, start: f32, stop: f32) -> vec4f {
        var t: f32 = start;
        var accumulated = vec4f(0);
        while (t < stop && accumulated.a < 0.999) {
            let pos = origin + direction * t;
            let volumePos = (pos + .5) * vec3f(u.x, u.y, u.z) + 0.5;
            var s = trilerp4(&velocity, volumePos);
            var temp = trilerp1(&temperature, volumePos);
            let col = blackbodyColor(temp);
            let l = 1.0; // TODO: Lighting amount based on temp
            s.a *= u.stepLength * OPTICAL_DENSITY;
            s = vec4f(col * l * s.a , s.a); // Weight the added sample by density
            accumulated += s * (1.0 - accumulated.a);
            t += u.stepLength;
        }
        return accumulated;
    }

    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var tex: texture_storage_2d<${presentationFormat}, write>;
    @group(0) @binding(2) var<storage, read> velocity : array<vec4f>;
    @group(0) @binding(3) var<storage, read> temperature : array<f32>;
    @compute @workgroup_size(8,8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        ${source.commonBoxRayIntersection}
        color = pow(color, vec3f(1 / 1.8));
        textureStore(tex, vec2<i32>(i32(global_id.x), i32(global_id.y)), vec4f(color, 1));
    }`);

    computeShaders.renderPressure = new ComputeShader("renderPressure", device, /*wgsl*/`
    ${source.common}

    fn marchRay(origin: vec3<f32>, direction: vec3<f32>, start: f32, stop: f32) -> vec4f {
        var t: f32 = start;
        var accumulated = vec4f(0);
        while (t < stop && accumulated.a < 0.9) {
            let pos = origin + direction * t;
            let volumePos = (pos + .5) * vec3f(u.x, u.y, u.z);
            var pres = pressure[clamp_to_edge(vec3i(volumePos))];
            // pres = fract(volumePos.z);
            pres *= 1000;
            var s = vec4f(max(0, pres), 0, max(0, -pres), 50.0);
            let f = fract(volumePos);
            const cellGap = 0.25;
            const cellGapInv = 1.0 - cellGap;
            if (f.x < cellGap || f.x > cellGapInv
                || f.y < cellGap || f.y > cellGapInv
                || f.z < cellGap || f.z > cellGapInv) {
                s.a = 0.0;
            }
            s.a *= u.stepLength;
            s = vec4f(s.rgb * s.a , s.a); // Weight the added sample by density
            accumulated += s * (1.0 - accumulated.a);
            t += u.stepLength;
        }
        return accumulated;
    }

    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var tex: texture_storage_2d<${presentationFormat}, write>;
    @group(0) @binding(2) var<storage, read> pressure : array<f32>;
    @compute @workgroup_size(8,8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        ${source.commonBoxRayIntersection}
        color = pow(color, vec3f(1 / 1.8));
        textureStore(tex, vec2<i32>(i32(global_id.x), i32(global_id.y)), vec4f(color, 1));
    }`);

    computeShaders.renderDivergence = new ComputeShader("renderDivergence", device, /*wgsl*/`
    ${source.common}

    fn marchRay(origin: vec3<f32>, direction: vec3<f32>, start: f32, stop: f32) -> vec4f {
        var t: f32 = start;
        var accumulated = vec4f(0);
        while (t < stop && accumulated.a < 0.999) {
            let pos = origin + direction * t;
            let volumePos = (pos + .5) * vec3f(u.x, u.y, u.z);
            var diverg = divergence[clamp_to_edge(vec3i(volumePos))];
            // diverg = log(abs(diverg)) * sign(diverg);
            diverg *= 0.1;
            var s = vec4f(max(0, diverg), 0, max(0, -diverg), 25.0);
            let f = fract(volumePos);
            const cellGap = 0.25;
            const cellGapInv = 1.0 - cellGap;
            if (f.x < cellGap || f.x > cellGapInv
                || f.y < cellGap || f.y > cellGapInv
                || f.z < cellGap || f.z > cellGapInv) {
                s.a = 0.0;
            }
            s.a *= u.stepLength;
            s = vec4f(s.rgb * s.a , s.a); // Weight the added sample by density
            accumulated += s * (1.0 - accumulated.a);
            t += u.stepLength;
        }
        return accumulated;
    }

    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var tex: texture_storage_2d<${presentationFormat}, write>;
    @group(0) @binding(2) var<storage, read> divergence : array<f32>;
    @compute @workgroup_size(8,8)
    fn main(@builtin(global_invocation_id) global_id: vec3u) {
        ${source.commonBoxRayIntersection}
        color = pow(color, vec3f(1 / 1.8));
        textureStore(tex, vec2<i32>(i32(global_id.x), i32(global_id.y)), vec4f(color, 1));
    }`);

}
