function buildShadersAdvect({device, computeShaders, source}) {
    computeShaders.advect = new ComputeShader("advect", device, /*wgsl*/`
    ${source.common}
    @group(0) @binding(0) var<uniform> u : U;
    @group(0) @binding(1) var<storage, read> velocity_in : array<vec4f>;
    @group(0) @binding(2) var<storage, read> smoke_in : array<vec4f>;
    @group(0) @binding(3) var<storage, read> temperature_in : array<f32>;
    @group(0) @binding(4) var<storage, read_write> velocity_out : array<vec4f>;
    @group(0) @binding(5) var<storage, read_write> smoke_out : array<vec4f>;
    @group(0) @binding(6) var<storage, read_write> temperature_out : array<f32>;
    @group(0) @binding(7) var<storage, read_write> pressure : array<f32>;
    @compute @workgroup_size(4,4,4)
    fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let index = to_index(global_id);
        let worldPos = vec3f(global_id.xyz) + .5;
        let k1 = velocity_in[index].xyz ;
        // let k2 = trilerp4(&velocity_in, worldPos - 0.5 * u.dt * u.rdx * k1).xyz ;
        // let k3 = trilerp4(&velocity_in, worldPos - 0.5 * u.dt * u.rdx * k2).xyz ;
        // let k4 = trilerp4(&velocity_in, worldPos -       u.dt * u.rdx * k3).xyz ;
        // let pos2 = worldPos - u.dt * u.rdx * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        // let k2 = trilerp4(&velocity_in, worldPos - u.dt * u.rdx * k1);
        // var pos2 = worldPos - u.dt * u.rdx * (k1.xyz + k2.xyz) * 0.5;

        let pos2 = worldPos - u.dt * u.rdx * k1;

        if (u.enclosed == 0 && (pos2.x < -.5 || pos2.y < -.5 || pos2.z < -.5
            || pos2.x > u.x - .5
            || pos2.y > u.y - .5
            || pos2.z > u.z - .5)) {
            velocity_out[index] = vec4f(0);
            smoke_out[index] = vec4f(0);
            temperature_out[index] = 0;
            return;
        }

        let vel2 = trilerp4(&velocity_in, pos2);

        var smoke = trilerp4(&smoke_in, pos2) * vec4(1, 1, 1, exp(-u.smokeDecay * u.dt));
        var temperature = trilerp1(&temperature_in, pos2) * exp(-u.temperatureDecay * u.dt);
        let burnAmount = select(0.0, u.burnRate * u.dt, temperature > u.ignitionTemperature);
        let velOut = vec4(vel2.xyz * exp(-u.velocityDecay * u.dt), max(0.0, vel2.w - burnAmount));
        let burntFuelAmount = vel2.w - velOut.w;
        let emittedSmoke = burntFuelAmount * u.burnSmokeEmit;

        temperature += burntFuelAmount * u.burnHeatEmit;
        smoke = add_smoke(smoke, vec4(u.smokeR, u.smokeG, u.smokeB, emittedSmoke));
        smoke_out[index] = smoke;
        const gravityVec = vec4(0,0,1,0);

        temperature_out[index] = temperature;
        velocity_out[index] = velOut + gravityVec * temperature * u.boyancy * 1e-3 * u.dt;
        pressure[index] *= exp(-u.pressureDecay * u.dt);
    }`);
}
