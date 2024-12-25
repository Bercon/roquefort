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

    // MULTIGRID

    computeShaders.restrict = new ComputeShader("restrict", device, /*wgsl*/`
    ${iterUniformStruct}
    ${source.pressure}
    @group(0) @binding(0) var<storage, read_write> buffer : array<f32>;
    @group(0) @binding(1) var<uniform> iter : Iter;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let base = global_id * 2;
        buffer[iter.offsetNext + to_index_dim(global_id, iter.next)] =
            0.125 * (
            buffer[iter.offsetCurrent + to_index_dim(base, iter.current)]
            + buffer[iter.offsetCurrent + to_index_dim(base + vec3u(1,0,0), iter.current)]
            + buffer[iter.offsetCurrent + to_index_dim(base + vec3u(0,1,0), iter.current)]
            + buffer[iter.offsetCurrent + to_index_dim(base + vec3u(1,1,0), iter.current)]
            + buffer[iter.offsetCurrent + to_index_dim(base + vec3u(0,0,1), iter.current)]
            + buffer[iter.offsetCurrent + to_index_dim(base + vec3u(1,0,1), iter.current)]
            + buffer[iter.offsetCurrent + to_index_dim(base + vec3u(0,1,1), iter.current)]
            + buffer[iter.offsetCurrent + to_index_dim(base + vec3u(1,1,1), iter.current)]
            );
    }`);

    computeShaders.relax = new ComputeShader("relax", device, /*wgsl*/`
    ${iterUniformStruct}
    ${source.pressure}
    // @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(0) var<storage, read_write> buffer : array<f32>;
    @group(0) @binding(1) var<uniform> iter : Iter;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let base = global_id * 2;
        let coarse = buffer[iter.offsetNext + to_index_dim(global_id, iter.next)];
        buffer[iter.offsetCurrent + to_index_dim(base, iter.current)] = coarse;
        buffer[iter.offsetCurrent + to_index_dim(base + vec3u(1,0,0), iter.current)] = coarse;
        buffer[iter.offsetCurrent + to_index_dim(base + vec3u(0,1,0), iter.current)] = coarse;
        buffer[iter.offsetCurrent + to_index_dim(base + vec3u(1,1,0), iter.current)] = coarse;
        buffer[iter.offsetCurrent + to_index_dim(base + vec3u(0,0,1), iter.current)] = coarse;
        buffer[iter.offsetCurrent + to_index_dim(base + vec3u(1,0,1), iter.current)] = coarse;
        buffer[iter.offsetCurrent + to_index_dim(base + vec3u(0,1,1), iter.current)] = coarse;
        buffer[iter.offsetCurrent + to_index_dim(base + vec3u(1,1,1), iter.current)] = coarse;
    }`);

    source.pressureSolver = /*wgsl*/`
    ${iterUniformStruct}
    ${source.pressure}
    // @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(0) var<storage, read_write> pressure : array<f32>;
    @group(0) @binding(1) var<storage, read> divergence : array<f32>;
    @group(0) @binding(2) var<uniform> iter : Iter;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let offset = (global_id.x + global_id.y + global_id.z + iter.i) % 2 ;
        let global_id_w_offset = vec3(global_id.x, global_id.y, global_id.z * 2 + offset);
        let index = iter.offsetCurrent + to_index_dim(global_id_w_offset, iter.current);
        let base = vec3i(global_id_w_offset);
        let le = pressure[iter.offsetCurrent + clamp_to_edge_dim(base - vec3(1,0,0), iter.current)];
        let ri = pressure[iter.offsetCurrent + clamp_to_edge_dim(base + vec3(1,0,0), iter.current)];
        let fr = pressure[iter.offsetCurrent + clamp_to_edge_dim(base - vec3(0,1,0), iter.current)];
        let ba = pressure[iter.offsetCurrent + clamp_to_edge_dim(base + vec3(0,1,0), iter.current)];
        let to = pressure[iter.offsetCurrent + clamp_to_edge_dim(base - vec3(0,0,1), iter.current)];
        let bo = pressure[iter.offsetCurrent + clamp_to_edge_dim(base + vec3(0,0,1), iter.current)];
        let alpha = -(iter.dx * iter.dx);
        const rBeta = 1. / 6.0;
        pressure[index] = (le + ri + fr + ba + to + bo + alpha * divergence[index]) * rBeta;
    }`

    computeShaders.solve = new ComputeShader("solve", device,
        source.pressureSolver);
    computeShaders.solveGroupSize442 = new ComputeShader("solveDim4", device,
        source.pressureSolver.replace("@workgroup_size(4,4,4)", "@workgroup_size(4,4,2)"));
}
