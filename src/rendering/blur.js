function buildShadersBlur({device, computeShaders, presentationFormat, source}) {
    source.blackbody = /*wgsl*/`
    // fn blackbodyColor(temp: f32) -> vec3<f32> {
    //     const colors: array<vec3<f32>, 9> = array<vec3<f32>, 9>(
    //         vec3<f32>(0.0, 0.0, 0.0), // 1000K
    //         vec3<f32>(0.5, 0.3, 0.1), // 1500K
    //         vec3<f32>(0.8, 0.5, 0.2), // 2000K
    //         vec3<f32>(1.0, 0.6, 0.3), // 2500K
    //         vec3<f32>(1.0, 0.7, 0.4), // 3000K
    //         vec3<f32>(1.0, 0.8, 0.5), // 3500K
    //         vec3<f32>(1.0, 0.9, 0.6), // 4000K
    //         vec3<f32>(1.0, 1.0, 0.8), // 4500K
    //         vec3<f32>(1.0, 1.0, 1.0), // 5000K (white, not much change from 5000K onward)
    //     );
    //     let t = clamp(temp - 1000., 0, 4000.);
    //     let idx = i32(t / 500.);
    //     return mix(colors[idx], colors[idx + 1], (t % 500. / 500.0));
    // }
    `;

    computeShaders.boxBlurHorizontal = new ComputeShader("boxBlurHorizontal", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> original : array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> blurred : array<vec4f>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        if (global_id.x >= u32(u.canvasiY)) { return; }
        var sum = vec4f(0);
        var size = u.glowRadius;
        var y = i32(global_id.x) * u.canvasiX;
        for (var i = 0; i < min(u.glowRadius, u.canvasiX); i++) {
            sum += original[i + y];
        }
        for (var i = 0; i < u.canvasiX; i++) {
            let l = i - u.glowRadius - 1;
            let r = i + u.glowRadius;
            if (l >= 0) {
                sum -= original[l + y];
                size--;
            }
            if (r < u.canvasiX) {
                sum += original[r + y];
                size++;
            }
            blurred[i + y] = sum / f32(size);
        }
    }`);

    computeShaders.boxBlurVertical = new ComputeShader("boxBlurVertical", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> original : array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> blurred : array<vec4f>;
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        if (global_id.x >= u32(u.canvasiX)) { return; }
        var sum = vec4f(0);
        var size = u.glowRadius;
        var x = i32(global_id.x);
        for (var i = 0; i < min(u.glowRadius, u.canvasiY); i++) {
            sum += original[i * u.canvasiX + x];
        }
        for (var i = 0; i < u.canvasiY; i++) {
            let l = i - u.glowRadius - 1;
            let r = i + u.glowRadius;
            if (l >= 0) {
                sum -= original[l * u.canvasiX + x];
                size--;
            }
            if (r < u.canvasiY) {
                sum += original[r * u.canvasiX + x];
                size++;
            }
            blurred[i * u.canvasiX + x] = sum / f32(size);
        }
    }`);

    computeShaders.composite = new ComputeShader("composite", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> original : array<vec4f>;
    @group(0) @binding(2) var<storage, read> blurred : array<vec4f>;
    @group(0) @binding(3) var tex: texture_storage_2d<${presentationFormat}, write>;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        if (global_id.x >= u32(u.canvasiX) || global_id.y >= u32(u.canvasY)) { return; }
        let base = vec2i(global_id.xy);
        let idx = base.x + base.y * u.canvasiX;
        var color = blurred[idx];
        color = original[idx] + pow(color, vec4f(u.glowGamma)) * u.glowStrength;
        // color *= 4;
        // color = color / (color + 1);
        var luminance = dot(color, vec4(0.2126, 0.7152, 0.0722, 0));
        var saturation = 1.2;
        color = mix(vec4(luminance), color, saturation);
        color = pow(color, vec4f(1 / 1.8));
        // color *= 1.1;
        textureStore(tex, vec2i(global_id.xy),
            vec4f(color.rgb + D(vec3(vec2f(global_id.xy), u.t)).x / 255., 1));
    }`);

    // computeShaders.blurHorizontal = new ComputeShader("blurHorizontal", device, /*wgsl*/`
    // ${source.common}
    // @group(0) @binding(0) var<uniform> u : U;
    // @group(0) @binding(1) var<storage, read> original : array<vec4f>;
    // @group(0) @binding(2) var<storage, read_write> blurred : array<vec4f>;
    // @compute @workgroup_size(8, 8)
    // fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    //     if (global_id.x >= u32(u.canvasX) || global_id.y >= u32(u.canvasY)) { return; }
    //     let base = vec2i(global_id.xy);
    //     var color = vec4f(0);
    //     var weight = 0.;
    //     for (var i = -u.glowRadius; i <= u.glowRadius; i++) {
    //         let w = f32(abs(i) - u.glowRadius + 1);
    //         color += w * original[clamp(base.x + i, 0, u.canvasiX - 1) + base.y * u.canvasiX];
    //         weight += w; // TODO: Precompute
    //     }
    //     blurred[base.x + base.y * u.canvasiX] = color / weight;
    // }`);

    // computeShaders.blurVertical = new ComputeShader("blurVertical", device, /*wgsl*/`
    // ${source.common}
    // @group(0) @binding(0) var<uniform> u : U;
    // @group(0) @binding(1) var<storage, read> original : array<vec4f>;
    // @group(0) @binding(2) var<storage, read> blurred : array<vec4f>;
    // @group(0) @binding(3) var tex: texture_storage_2d<${presentationFormat}, write>;
    // @compute @workgroup_size(8, 8)
    // fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    //     if (global_id.x >= u32(u.canvasX) || global_id.y >= u32(u.canvasY)) { return; }
    //     let base = vec2i(global_id.xy);
    //     var color = vec4f(0);
    //     var weight = 0.;
    //     for (var i = -u.glowRadius; i <= u.glowRadius; i++) {
    //         let w = f32(abs(i) - u.glowRadius + 1);
    //         color += w * blurred[base.x + clamp(base.y + i, 0, u.canvasiY - 1) * u.canvasiX];
    //         weight += w; // TODO: Precompute
    //     }
    //     color /= weight;
    //     // color = max(vec4f(0), color - u.glowCutoff);
    //     // color = vec4f(0);
    //     color = original[base.x + base.y * u.canvasiX] + pow(color, vec4f(u.glowGamma)) * u.glowStrength;
    //     color = pow(color, vec4f(1 / 1.8));
    //     textureStore(tex, vec2<i32>(i32(global_id.x), i32(global_id.y)),
    //         vec4f(color.rgb + D(vec3(vec2f(global_id.xy), u.t)).x / 255., 1));

    // }`);

}