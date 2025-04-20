function buildShadersLighting({device, computeShaders, lightUniformStruct, source}) {
    source.blackbody = /*wgsl*/`
    fn blackbodyColor(temp: f32) -> vec3<f32> {
        const colors: array<vec3<f32>, 9> = array<vec3<f32>, 9>(
            vec3<f32>(0.0, 0.0, 0.0), // 1000K
            vec3<f32>(0.5, 0.3, 0.1), // 1500K
            vec3<f32>(0.8, 0.5, 0.2), // 2000K
            vec3<f32>(1.0, 0.6, 0.3), // 2500K
            vec3<f32>(1.0, 0.7, 0.4), // 3000K
            vec3<f32>(1.0, 0.8, 0.5), // 3500K
            vec3<f32>(1.0, 0.9, 0.6), // 4000K
            vec3<f32>(1.0, 1.0, 0.8), // 4500K
            vec3<f32>(1.0, 1.0, 1.0), // 5000K (white, not much change from 5000K onward)
        );
        let t = clamp(temp - 1000., 0, 4000.);
        let idx = i32(t / 500.);
        return mix(colors[idx], colors[idx + 1], (t % 500. / 500.0));
    }
    `;

    computeShaders.lighting = new ComputeShader("lighting", device, /*wgsl*/`
    ${lightUniformStruct}
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<uniform> light : Light;
    @group(0) @binding(2) var<storage, read> smoke : array<vec4f>;
    @group(0) @binding(3) var<storage, read_write> lighting : array<vec4f>;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        var currentLight = vec3f(light.lightR, light.lightG, light.lightB);
        var pos = vec3i(global_id);
        var lightDir = light.lightDir;
        // if (i32(u.t * 4) % 4 == 0) { lightDir = 0; }
        // if (i32(u.t * 4) % 4 == 1) { lightDir = 5; }
        // if (i32(u.t * 4) % 4 == 2) { lightDir = 3; }
        // if (i32(u.t * 4) % 4 == 3) { lightDir = 2; }
        // // lightDir = 0;
        pos = array<vec3i, 6>(
            vec3(0, pos.x, pos.y), // left
            vec3(pos.x, 0, pos.y), // front
            vec3(pos.x, pos.y, 0), // bottom
            vec3(i32(u.x) - 1, pos.x, pos.y), // right
            vec3(pos.x, i32(u.y) - 1, pos.y, ), // back
            vec3(pos.x, pos.y, i32(u.z) - 1) // top
        )[lightDir];
        var dir = array<vec3i, 6>(
            vec3(1, 0, 0), // left
            vec3(0, 1, 0), // front
            vec3(0, 0, 1), // bottom
            vec3(-1, 0, 0), // right
            vec3(0, -1, 0), // back
            vec3(0, 0, -1) // top
        )[lightDir];
        var stepLength = 1. / u.x;
        // let dist = abs(f32(pos.x) - u.x * (.5 + .5 * sin(u.t * .5))) / u.x;
        // let dist = fract(f32(pos.x) * .1);
        // let dist = abs((f32(pos.x) - u.x * .5 * sin(u.t)) / (u.x * .5));
        // currentLight *= max(0, .2 - dist) * 25.;
        for (var i = 0; i < i32(u.x); i++) {
            let index = to_index(vec3u(pos + dir * i));
            var s = smoke[index];
            let absorption = 1 - vec3f(s.x, s.y, s.z);
            let density = s.w * stepLength * OPTICAL_DENSITY;
            let attenuation = exp(-absorption * density);
            currentLight = currentLight * attenuation;
            // if (density > 0.02) { currentLight = vec3(0,0,0); }
            lighting[index] = lighting[index] * light.lightPrev + vec4f(currentLight, 0);
        }
    }`);

    computeShaders.bakeLighting = new ComputeShader("bakeLighting", device, /*wgsl*/`
    ${source.common}
    ${source.blackbody}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> smoke : array<vec4f>;
    @group(0) @binding(2) var<storage, read> temperature : array<f32>;
    @group(0) @binding(3) var<storage, read_write> lighting : array<vec4f>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let index = to_index(global_id);
        var temp = temperature[index];
        var blackbody = blackbodyColor(temp) * (max(temp, 1000.) - 1000.) * 0.01;
        var s = smoke[index];
        lighting[index] = vec4(
            s.rgb * (lighting[index].rgb + blackbody * u.blackbodyBrightness),
            s.a
        );
    }`);

    computeShaders.lightingFuel = new ComputeShader("lighting_fuel", device, /*wgsl*/`
    ${lightUniformStruct}
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<uniform> light : Light;
    @group(0) @binding(2) var<storage, read> velocity : array<vec4f>;
    @group(0) @binding(3) var<storage, read_write> lighting : array<vec4f>;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        var currentLight = vec3f(light.lightR, light.lightG, light.lightB);
        var orig = vec3i(global_id);
        var dir: vec3i;
        // left, right, top, bottom, front, back
        if (light.lightDir == 0) {
            orig.x = 0;
            dir = vec3i(0, 0, 1);
        } else if (light.lightDir == 2) {
            orig.z = i32(u.z) - 1;
            dir = vec3i(0, 0, -1);
        } else {
            orig.z = 0;
            dir = vec3i(0, 0, 1);
        }
        var stepLength = 1. / u.x;
        for (var i = 0; i < i32(u.x); i++) {
            let index = to_index(vec3u(orig + dir * i));
            var s = velocity[index]; // 4th channel is fuel amount
            let absorption = vec3f(1.0);
            let density = s.w * stepLength * OPTICAL_DENSITY;
            let attenuation = exp(-absorption * density);
            currentLight = currentLight * attenuation;
            lighting[index] = lighting[index] * light.lightPrev + vec4f(currentLight, 0);
        }
    }`);
}