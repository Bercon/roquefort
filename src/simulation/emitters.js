function buildShadersEmitters({ device, computeShaders, source }) {
    // source.colorConverters = /*wgsl*/`
    // fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    //     let k = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    //     let p = abs(fract(c.xxx + k.xyz) * 6.0 - k.www);
    //     return c.z * mix(k.xxx, clamp(p - k.xxx, 0.0, 1.0), c.y);
    // }
    // `;

    computeShaders.rotatingSmokeEmitter = new ComputeShader("rotatingSmokeEmitter", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read_write> velocity : array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> smoke : array<vec4f>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        if (global_id.x == 0 || global_id.y == 0 || global_id.z == 0
            || global_id.x == u32(u.x - 1) || global_id.y == u32(u.y - 1) || global_id.z == u32(u.z - 1)) {
            return;
        }
        let index = to_index(global_id);
        let worldPos = (vec3f(global_id) / vec3f(u.x, u.y, u.z)) - 0.5;
        let spherePos = vec3(
            sin(u.t * .75) * .2,
            cos(u.t * .75) * .2,
            -.2 + .1 * sin(u.t * 1.127)
        );
        let dist = length(worldPos - spherePos);
        let color = vec3f(u.emitterR, u.emitterG, u.emitterB);
        let spot = max(0., (1. - dist * 10.)) * (.2 + .8 *sin(u.t * .5) * sin(u.t * .5));
        velocity[index] += vec4(0, 0, 0.00001, 0);
        let old = smoke[index];
        let added = vec4f(color, spot * 6. * u.dt);
        const epsilon = 1e-10;
        smoke[index] = vec4(
            mix(added.rgb, old.rgb, old.a / (added.a + old.a + epsilon)),
            old.a + added.a
        );
    }`);

    computeShaders.rotatingFireEmitter = new ComputeShader("rotatingFireEmitter", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read_write> velocity : array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> temperature : array<f32>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        if (global_id.x == 0 || global_id.y == 0 || global_id.z == 0
            || global_id.x == u32(u.x - 1) || global_id.y == u32(u.y - 1) || global_id.z == u32(u.z - 1)) {
            return;
        }
        let index = to_index(global_id);
        let worldPos = (vec3f(global_id) / vec3f(u.x, u.y, u.z)) - 0.5;
        let spherePos = vec3(
            sin(u.t * .75) * .2,
            cos(u.t * .75) * .2,
            -.35 + .05 * sin(u.t * 1.127));
        let dist = length(worldPos - spherePos);
        let color = vec3f(global_id) / vec3f(u.x, u.y, u.z);
        let asf = sin(u.t * 2.3) * (.5 + .5 * sin(u.t * 5.3));
        let spot = max(0., sqrt(1. - dist * 10.)) * (.6 + .4 * asf * asf);
        let direction = vec3(
            sin(u.t * 1.33),
            cos(u.t * 1.33),
            sin(u.t * 1.127));
        velocity[index] += vec4(direction * spot * 1., spot * 5.) * u.dt;
        temperature[index] += 5000. * spot * u.dt;
    }`);


    computeShaders.updateMouse = new ComputeShader("updateMouse", device, /*wgsl*/`
    ${source.common}

    fn palette(t : f32, a : vec3<f32>, b : vec3<f32>, c : vec3<f32>, d : vec3<f32> ) -> vec3<f32> {
        return a + b * cos(6.28318 * (c * t + d));
    }

    fn distanceToLine(a: vec3f, b: vec3f, pos: vec3f) -> f32 {
        let pa = pos - a;
        let ba = b - a;
        let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
        return length(pa - ba * h);
    }

    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read_write> velocity : array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> smoke : array<vec4f>;
    @group(0) @binding(3) var<storage, read_write> temperature : array<f32>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

        if (global_id.x == 0 || global_id.y == 0 || global_id.z == 0
            || global_id.x == u32(u.x - 1) || global_id.y == u32(u.y - 1) || global_id.z == u32(u.z - 1)) {
            return;
        }

        let index = to_index(global_id);

        let worldPos = (vec3f(global_id) / vec3f(u.x, u.y, u.z)) - 0.5;

        let a = (vec2f(u.mouseStartX * u.canvasX, u.mouseStartY * u.canvasY) - vec2f(u.canvasX, u.canvasY) * .5) / u.canvasX;
        let b = (vec2f(u.mouseEndX * u.canvasX, u.mouseEndY * u.canvasY) - vec2f(u.canvasX, u.canvasY) * .5) / u.canvasX;

        let boxMin = vec3f(-.5);
        let boxMax = vec3f(.5);
        let fov = radians(90.0);

        let camPos = vec3f(u.camPosX, u.camPosY, u.camPosZ);
        let camRot = computeBasis(vec3(0, 0, 0) - camPos);

        let aDir = computeRayDirection(camRot, fov, a);
        let bDir = computeRayDirection(camRot, fov, b);

        let aIntersections = rayBoxIntersect(camPos, aDir, boxMin, boxMax);
        let bIntersections = rayBoxIntersect(camPos, bDir, boxMin, boxMax);

        let aDist = (aIntersections.x + aIntersections.y) * 0.5;
        let bDist = (bIntersections.x + bIntersections.y) * 0.5;

        if (aDist < 0 || bDist < 0) { return; }

        let aPos = camPos + aDir * aDist;
        let bPos = camPos + bDir * bDist;

        let dist = distanceToLine(aPos, bPos, worldPos);

        let spot = sqrt(max(0., u.brushSize - dist));
        // let spot = max(0, 1. - length(worldPos) * 5.);

        let col_incr = 0.15;
        let color = palette(u.t / 8., vec3(1), vec3(0.5), vec3(1), vec3(0, col_incr, col_incr*2.));
        let old = smoke[index];
        let added = vec4f(color, spot * 2. * u.dt * u.brushSmokeAmount);
        const epsilon = 1e-10;
        smoke[index] = vec4(
            mix(added.rgb, old.rgb, old.a / (added.a + old.a + epsilon)),
            old.a + added.a
        );
        velocity[index] += vec4f((bPos - aPos) * u.brushVelocityAmount * 5, u.brushFuelAmount) * spot;
        temperature[index] += 500. * u.brushTemperatureAmount * spot;
    }`);
}
