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
        var orig = vec3i(global_id);
        var dir: vec3i;
        // left, right, top, bottom, front, back
        if (light.lightDir == 0.) {
            orig = vec3i(0, orig.x, orig.y);
            dir = vec3i(1, 0, 0);
        } else if (light.lightDir == 1.) {
            orig = vec3i(i32(u.z) - 1, orig.x, orig.y);
            dir = vec3i(-1, 0, 0);
        } else if (light.lightDir == 2.) {
            orig = vec3i(orig.x, orig.y, i32(u.z) - 1);
            dir = vec3i(0, 0, -1);
        } else {
            // TODO: IMPLEMENT MISSTING
            orig.z = 0;
            dir = vec3i(0, 0, 0);
        }
        var stepLength = 1. / u.x;
        for (var i = 0; i < i32(u.x); i++) {
            let index = to_index(vec3u(orig + dir * i));
            var s = smoke[index];
            let absorption = vec3f(s.x, s.y, s.z);
            let density = s.w * stepLength * OPTICAL_DENSITY;
            let attenuation = exp(-absorption * density);
            currentLight = currentLight * attenuation;
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