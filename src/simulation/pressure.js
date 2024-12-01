function buildShadersPressure({ device, computeShaders, iterUniformStruct, source }) {
    computeShaders.jacobi = new ComputeShader("jacobi", device, /*wgsl*/`
    ${iterUniformStruct}
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> pressure_in : array<f32>;
    @group(0) @binding(2) var<storage, read> divergence : array<f32>;
    @group(0) @binding(3) var<storage, read_write> pressure_out : array<f32>;
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
        let alpha = -(u.dx * u.dx);
        // let rBeta = .25;
        const rBeta = 1. / 6.0;
        pressure_out[index] = (le + ri + fr + ba + to + bo + alpha * divergence[index]) * rBeta;
        // TODO: Over relaxation
        // let omega = 1.;
        // let new_pressure = (le + ri + fr + ba + to + bo + alpha * divergence[index]) * rBeta;
        // let old_pressure = pressure_in[index];
        // let correction = new_pressure - old_pressure;
        // pressure_out[index] = old_pressure + correction * omega;
        // pressure_out[index] = (1.0 - omega) * pressure_in[index] + omega * new_pressure;
    }`);

    computeShaders.jacobiRedBlack = new ComputeShader("jacobiRedBlack", device, /*wgsl*/`
    ${iterUniformStruct}
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read_write> pressure : array<f32>;
    @group(0) @binding(2) var<storage, read> divergence : array<f32>;
    @group(0) @binding(3) var<uniform> iter : Iter;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let offset = (global_id.x + global_id.y + global_id.z + iter.i) % 2 ;
        let global_id_w_offset = vec3(global_id.x, global_id.y, global_id.z * 2 + offset);
        let index = to_index(global_id_w_offset);
        let base = vec3i(global_id_w_offset);
        let le = pressure[clamp_to_edge(base - vec3(1,0,0))];
        let ri = pressure[clamp_to_edge(base + vec3(1,0,0))];
        let fr = pressure[clamp_to_edge(base - vec3(0,1,0))];
        let ba = pressure[clamp_to_edge(base + vec3(0,1,0))];
        let to = pressure[clamp_to_edge(base - vec3(0,0,1))];
        let bo = pressure[clamp_to_edge(base + vec3(0,0,1))];
        let alpha = -(u.dx * u.dx);
        const rBeta = 1. / 6.0;
        pressure[index] = (le + ri + fr + ba + to + bo + alpha * divergence[index]) * rBeta;
    }`);

    computeShaders.pressureClear = new ComputeShader("pressureClear", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read_write> pressure : array<f32>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        pressure[to_index(global_id)] *= exp(-u.pressureDecay * u.dt);
    }`);
}
