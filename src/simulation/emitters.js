function buildShadersEmitters({ device, computeShaders, source }) {
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
        smoke[index] = add_smoke(old, added);
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

    computeShaders.fiveSpheres = new ComputeShader("fiveSpheres", device, /*wgsl*/`
    ${source.common}
    fn hue_to_rgb(h: f32) -> vec3<f32> {
        // return abs(fract(h + vec3(0, 2. / 3., 1. / 3.)) * 6. - 3.);
        return 0.5 + 0.5 * cos(h + vec3f(0,2,4));
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
        let worldPos = (vec3f(global_id) / vec3f(u.x, u.y, u.z)) * 2. - 1.;
        var newSmoke = smoke[index];
        var newVelocity = velocity[index];
        var newTemperature = temperature[index];
        for (var i = 0; i < 5; i++) {
            let phase = f32(i) / 5 * PI * 2;
            let phaseZ = f32(i) * PI / 2.5 * 3.;
            let radius = .5 + sin(u.t * 0.3) * .3;
            let spherePos = vec3(
                sin(u.t * .5 + phase) * radius,
                cos(u.t * .5 + phase) * radius,
                .6 * sin(u.t * 1.5 + phaseZ));
            let prevSpherePos = vec3(
                sin(u.t * .5 - u.dt + phase) * radius,
                cos(u.t * .5 - u.dt + phase) * radius,
                .6 * sin(u.t * 1.5 + phaseZ));
            let dist = length(worldPos - spherePos);
            let spot = sqrt(max(0., u.brushSize * 2. - dist)) * u.dt;
            let col_incr = 0.15;
            // let color = palette(u.t / 8., vec3(1), vec3(0.5), vec3(1), vec3(0, col_incr, col_incr*2.));
            let color = hue_to_rgb(u.t * .2 + f32(i) * 1.);
            let added = vec4f(color, spot * 2. * u.brushSmokeAmount);
            newSmoke = add_smoke(newSmoke, added);
            newVelocity += vec4f((spherePos - prevSpherePos) * u.brushVelocityAmount * 5, u.dt * 60. * u.brushFuelAmount) * spot * 60.;;
            newTemperature += 500. * u.brushTemperatureAmount * spot * 60.;
        }
        smoke[index] = newSmoke;
        velocity[index] = newVelocity;
        temperature[index] = newTemperature;
    }`);

    computeShaders.vortex = new ComputeShader("vortex", device, /*wgsl*/`
    ${source.common}
    fn hue_to_rgb(h: f32) -> vec3<f32> {
        return 0.5 + 0.5 * cos(h + vec3f(0,2,4));
    }
    fn distanceToLine(a: vec3f, b: vec3f, pos: vec3f) -> f32 {
        let pa = pos - a;
        let ba = b - a;
        return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
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
        var worldPos = (vec3f(global_id) / vec3f(u.x, u.y, u.z)) * 2. - 1.;
        var vortexCenter = vec2f(sin(u.t * 0.1 + worldPos.z), cos(u.t * 0.1 + worldPos.z)) * sin(u.t * 0.3) * .3;
        var toCenter = worldPos.xy - vortexCenter;
        var clockwise = vec2f(toCenter.y, -toCenter.x);

        var newSmoke = smoke[index];
        var newTemp = temperature[index];
        var newVel = velocity[index];

        newVel += vec4f(clockwise * 0.1, .8 * length(toCenter), 0) * u.dt
            // * 0.
        ;

        // let randHighSpace = D(vec3f(global_id) + u.t % 1);


        // let spot = max(0, .4 - length(worldPos - vec3f(vortexCenter, -1.))) * .5 * u.dt;
        // let spot = max(0, (size - stickFigure)) * 1.5;
        // velocity[index] *= exp(-10000. * spot * u.dt);

        // // 5 spheres
        // for (var i = 0; i < 5; i++) {
        //     let phase = 2 * 3.141 * f32(i) / 5;
        //     let radius = .5 + sin(u.t * 0.3) * .3;
        //     var t = u.t;
        //     let spherePos = vec3(
        //         sin(t * 2 + phase) * radius,
        //         cos(t * 2 + phase) * radius,
        //         sin(t * 1 + phase) * .6 );
        //     t -= u.dt; // Not necessarily truly previous if speed != 1
        //     let prevSpherePos = vec3(
        //         sin(t * 2 + phase) * radius,
        //         cos(t * 2 + phase) * radius,
        //         sin(t * 1 + phase) * .6);
        //     let dist = length(worldPos - spherePos);
        //     let spot = sqrt(max(0., .1 - dist)) * u.dt;
        //     let col_incr = 0.15;
        //     let color = hue_to_rgb(u.t * .2); // + f32(i) * 1.
        //     let old = smoke[index];
        //     var added = vec4f(color, spot * 5.);

        //     // added.a += simplex(worldPos * 2.) * 0.001 * u.dt;
        //     smoke[index] = vec4(
        //         mix(added.rgb, old.rgb, old.a / (added.a + old.a + 1e-10)),
        //         old.a + added.a
        //     );
        //     velocity[index] += vec4f((spherePos - prevSpherePos) * 200, 0) * spot;
        //     // temperature[index] += 500. * u.brushTemperatureAmount * spot * 60.;
        // }
        // let randHighFrequencySpace = D(vec3f(global_id) + u.t % 1);
        // let randHf = D(vec3f(1, 1, u.t));

        // Colliding rings
        // var t = u.t;
        // t *= .5;
        // let rand = D(vec3f(1, 1, floor(t)));
        // for (var i = 0; i < 2; i++) {
        //     let rand2 = D(vec3f(1, 1, floor(t + 100 + 100 * f32(i))));
        //     if (i == 1 && rand2.x < .5) { break; }
        //     var spherePos = (1. - 2. * rand) * .8;
        //     spherePos *= select(-1., 1., i == 0);
        //     let dist = length(worldPos - spherePos);
        //     var spot = sqrt(max(0., .2 - dist)) * u.dt;
        //     let color = hue_to_rgb(t * .6 + 4. * f32(i));

        //     let velSpot = spot * smoothstep(.97, 1., pow(sin(t * PI), 2));
        //     spot *= smoothstep(.98, 1., pow(sin(t * PI), 2));

        //     let old = smoke[index];
        //     let added = vec4f(color, spot * 10.);
        //     smoke[index] = vec4(
        //         mix(added.rgb, old.rgb, old.a / (added.a + old.a + 1e-10)),
        //         old.a + added.a
        //     );
        //     velocity[index] += vec4f(normalize(-spherePos + rand2 * .2) * 25, 0) * velSpot * (1. - .5 * rand2.x);
        //     temperature[index] += -20000 * spot;
        // }

        // Explosions
        // var t = u.t + 10.;
        // t *= .5;
        // let rand = D(vec3f(9, 9, floor(t)));
        // // let randb = D(vec3f(3, 3, floor(t)));
        // var spherePos = (1. - 2. * rand) * .5;
        // spherePos.z = -.5;

        // for (var i = 0; i < 100; i++) {
        //     let t2 = fract(t);
        //     let t3 = 1 - smoothstep(.9, 1, 1 - t2);
        //     var dir = D(vec3f(3, f32(i), floor(t))) - .5;
        //     // dir.z = abs(dir.z) * .8;
        //     let rand2 = D(vec3f(7, f32(i), floor(t)));
        //     let dist = length(worldPos - spherePos - dir * t3 * .1 * (3. + rand2.y));
        //     // var centerDistance = max(0., .3 - length(worldPos - spherePos));
        //     var spot = max(0., .08 - dist)  * u.dt;
        //     // spot = centerDistance * u.dt;
        //     let color = hue_to_rgb(floor(t) * 9.9);
        //     // let velSpot = spot;
        //     spot *= smoothstep(.0, 1, 1 - t3); // Impulse
        //     let added = vec4f(color, spot * 50.);
        //     // newSmoke = vec4(
        //     //     mix(added.rgb, newSmoke.rgb, newSmoke.a / (added.a + newSmoke.a + 1e-10)),
        //     //     newSmoke.a + added.a
        //     // );
        //     newVel += vec4f(dir * 500, 50) * spot; // * (.2 + .8 * rand2.x);
        //     // velocity[index] += vec4f(normalize(worldPos-spherePos) * 150, 5) * velSpot * (1. - .5 * rand2.x);
        //     newTemp += 50000 * spot;
        //     // newTemp += 5000 * select(0., 1., max(0., .08 - dist) != 0.) * u.dt;
        // }
        smoke[index] = newSmoke;
        temperature[index] = newTemp;
        velocity[index] = newVel;

        // let old = smoke[index];
        // let added = vec4f(vec3f(1), spot * u.dt);
        // smoke[index] = vec4(
        //     mix(added.rgb, old.rgb, old.a / (added.a + old.a + 1e-10)),
        //     old.a + added.a
        // );

        // worldPos = vec3f(worldPos.x, worldPos.y, 0);
        // // worldPos.z = 0.0;
        // // for (var i = 0; i < 5; i++) {
        //     let radius = .5 + sin(u.t * 0.3) * .3;
        //     let spherePos = vec3(
        //         sin(u.t * 1.75) * radius,
        //         .6 * sin(u.t * 1.5),
        //         0);
        //     let prevSpherePos = vec3(
        //         sin(u.t * 1.75 - u.dt) * radius,
        //         .6 * sin(u.t * 1.5),
        //         0);
        //     let dist = length(worldPos - spherePos);
        //     let spot = sqrt(max(0., u.brushSize * .5 - dist)) * u.dt;
        //     let col_incr = 0.15;
        //     // let color = palette(u.t / 8., vec3(1), vec3(0.5), vec3(1), vec3(0, col_incr, col_incr*2.));
        //     let color = hue_to_rgb(u.t * .2);
        //     let old = smoke[index];
        //     let added = vec4f(color, spot * 2. * u.brushSmokeAmount) * 5.;
        //     smoke[index] = vec4(
        //         mix(added.rgb, old.rgb, old.a / (added.a + old.a + 1e-10)),
        //         old.a + added.a
        //     );
        //     velocity[index] += vec4f((spherePos - prevSpherePos) * u.brushVelocityAmount * 5, u.dt * 60. * u.brushFuelAmount) * spot * 60.;
        //     // temperature[index] += 500. * u.brushTemperatureAmount * spot * 60.;
        // // }
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

        // if (abs(f32(global_id.z - u.ux / 2)) > .1) { return; }

        let index = to_index(global_id);

        let worldPos = (vec3f(global_id) / vec3f(u.x, u.y, u.z)) - 0.5;

        // let a = (vec2f(u.mouseStartX * u.canvasX, u.mouseStartY * u.canvasY) - vec2f(u.canvasX, u.canvasY) * .5) / u.canvasX;
        // let b = (vec2f(u.mouseEndX * u.canvasX, u.mouseEndY * u.canvasY) - vec2f(u.canvasX, u.canvasY) * .5) / u.canvasX;

        let rayA = computeCameraRay(vec3u(vec3f(u.mouseStartX * u.canvasX, u.mouseStartY * u.canvasY, 0)));
        let rayB = computeCameraRay(vec3u(vec3f(u.mouseEndX * u.canvasX, u.mouseEndY * u.canvasY, 0)));

        let boxMin = vec3f(-.5);
        let boxMax = vec3f(.5);
        // let fov = radians(90.0);

        // let camPos = vec3f(u.camPosX, u.camPosY, u.camPosZ);
        // let camRot = computeBasis(vec3(0, 0, 0) - camPos);

        // let aDir = computeRayDirection(camRot, fov, a);
        // let bDir = computeRayDirection(camRot, fov, b);

        let aIntersections = rayBoxIntersect(rayA.pos, rayA.dir, boxMin, boxMax);
        let bIntersections = rayBoxIntersect(rayB.pos, rayB.dir, boxMin, boxMax);

        let aDist = (aIntersections.x + aIntersections.y) * 0.5;
        let bDist = (bIntersections.x + bIntersections.y) * 0.5;

        if (aDist < 0 || bDist < 0) { return; }

        let aPos = rayA.pos + rayA.dir * aDist;
        let bPos = rayB.pos + rayB.dir * bDist;

        let dist = distanceToLine(aPos, bPos, worldPos);

        let spot = sqrt(max(0., u.brushSize - dist)) * u.dt;
        // let spot = max(0, 1. - length(worldPos) * 5.);

        let col_incr = 0.15;
        // let color = palette(u.t / 8., vec3(1), vec3(0.5), vec3(1), vec3(0, col_incr, col_incr*2.));
        let color = vec3f(.9,.1,.1);
        let old = smoke[index];
        let added = vec4f(color, spot * 2. * u.brushSmokeAmount);
        smoke[index] = add_smoke(old, added);
        velocity[index] += vec4f((bPos - aPos) * u.brushVelocityAmount * 5, u.brushFuelAmount) * spot * 60.;
        temperature[index] += 500. * u.brushTemperatureAmount * spot * 60.;
    }`);
}
