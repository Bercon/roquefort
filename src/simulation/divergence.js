function buildShadersDivergence({ device, computeShaders, source }) {
    computeShaders.divergence = new ComputeShader("divergence", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> velocity_in : array<vec4f>;
    @group(0) @binding(2) var<storage, read_write> div : array<f32>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let base = vec3i(global_id);
        var le = velocity_in[clamp_to_edge(base - vec3(1,0,0))].x;
        var ri = velocity_in[clamp_to_edge(base + vec3(1,0,0))].x;
        var fr = velocity_in[clamp_to_edge(base - vec3(0,1,0))].y;
        var ba = velocity_in[clamp_to_edge(base + vec3(0,1,0))].y;
        var to = velocity_in[clamp_to_edge(base - vec3(0,0,1))].z;
        var bo = velocity_in[clamp_to_edge(base + vec3(0,0,1))].z;
        if (u.enclosed != 0) {
            if (global_id.x == 0) { le *= -1.; }
            if (global_id.y == 0) { fr *= -1.; }
            if (global_id.z == 0) { to *= -1.; }
            if (global_id.x == u32(u.x - 1)) { ri *= -1.; }
            if (global_id.y == u32(u.y - 1)) { ba *= -1.; }
            if (global_id.z == u32(u.z - 1)) { bo *= -1.; }
        }
        // Add to existing divergence computed in advect step after burning gas expanding
        div[to_index(global_id)] += 0.5 * u.rdx * ((ri - le) + (ba - fr) + (bo - to));
    }`);
}
