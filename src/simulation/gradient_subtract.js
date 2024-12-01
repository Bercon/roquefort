function buildShadersGradientSubtract({ device, computeShaders, source }) {
    computeShaders.gradientSubtract = new ComputeShader("gradientSubtract", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> pressure_in : array<f32>;
    @group(0) @binding(2) var<storage, read_write> velocity : array<vec4f>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let index = to_index(global_id);
        let base = vec3i(global_id);
        let le = pressure_in[clamp_to_edge(base - vec3(1,0,0))];
        let ri = pressure_in[clamp_to_edge(base + vec3(1,0,0))];
        let fr = pressure_in[clamp_to_edge(base - vec3(0,1,0))];
        let ba = pressure_in[clamp_to_edge(base + vec3(0,1,0))];
        let to = pressure_in[clamp_to_edge(base - vec3(0,0,1))];
        let bo = pressure_in[clamp_to_edge(base + vec3(0,0,1))];
        velocity[index] -= vec4f(
            ri - le,
            ba - fr,
            bo - to,
            0) * .5 * u.rdx;
    }`);
}
