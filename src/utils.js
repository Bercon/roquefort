function uniqueId() {
    return crypto.randomUUID();
}

function createStorageBuffer(device, label, channels, width, height, depth = 1) {
    return device.createBuffer({
        label,
        size: 4 * channels * width * height * depth,
        usage: GPUBufferUsage.STORAGE,
    })
};

class DoubleStorageBuffer {
    constructor(device, label, channels, width, height, depth = 1) {
        this.read = createStorageBuffer(device, label + "_1", channels, width, height, depth);
        this.write = createStorageBuffer(device, label + "_2", channels, width, height, depth);
    }
    swap() {
        let tmp = this.read;
        this.read = this.write;
        this.write = tmp;
    }
    destroy() {
        this.read.destroy();
        this.write.destroy();
    }
}

class UniformBuffer {
    constructor(device, label, name, propDefinitions, props = undefined) {
        this.label = label;
        this.propDefinitions = propDefinitions;
        this.values = {};
        let s = "";
        this.fields = [];
        for (const [key, value] of Object.entries(propDefinitions)) {
            if (!["f32", "u32", "i32"].includes(value)) throw new Error(`Invalid type for ${key}: ${value}`)
            s += `\t${key} : ${value},`
            this.fields.push({ 'name': key, 'type': value });
            this.values[key] = null;
        }
        this.struct = `struct ${name} {
            ${s}
        }`;
        var a = new Uint32Array([0x12345678]);
        var b = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
        this.littleEndian = !(b[0] == 0x12);
        this.gpuBuffer = device.createBuffer({
            label: this.label,
            size: 4 * this.fields.length,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: !!props,
        });
        if (props) {
            this.updateProps(props);
            const buffer = this.gpuBuffer.getMappedRange();
            this.updateView(new DataView(buffer, 0));
            this.gpuBuffer.unmap();
        }
    }

    updateProps(newProps) {
        for (const key of Object.keys(newProps)) {
            if (!(key in this.values)) throw new Error(`Invalid uniform prop ${key}`)
            if (newProps[key] === undefined) throw new Error(`Undefined value for prop ${key}`)
            this.values[key] = newProps[key]
        }
    }

    updateView(view) {
        for (let i = 0; i < this.fields.length; i++) {
            const t = this.fields[i].type;
            const v = this.values[this.fields[i].name];
            if (v === null || v === undefined) throw new Error(`Undefined value for ${this.fields[i].name}`);
            if (t === "f32") view.setFloat32(i * 4, parseFloat(v), this.littleEndian);
            else if (t === "u32") view.setUint32(i * 4, v, this.littleEndian);
            else view.setInt32(i * 4, v, this.littleEndian);
        }
    }

    async update(device, newProps) {
        if (!this.arrayBuffer) {
            this.arrayBuffer = new ArrayBuffer(this.fields.length * 4);
            this.dataView = new DataView(this.arrayBuffer, 0);
        }
        this.updateProps(newProps);
        this.updateView(this.dataView);
        device.queue.writeBuffer(this.gpuBuffer, 0, this.arrayBuffer, this.arrayBuffer.byteOffset, this.arrayBuffer.byteLength);
    }
}

class ComputeShader {
    constructor(label, device, code) {
        this.label = label;
        this.pipeline = device.createComputePipeline({
            label,
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    label,
                    code
                })
            },
        });

    }

    computePass(device, pass, entries, dispatchX, dispatchY, dispatchZ) {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, device.createBindGroup({
            label: this.label,
            layout: this.pipeline.getBindGroupLayout(0),
            entries: entries.map((element, i) => ({
                binding: i,
                resource: element
            }))
        }));
        pass.dispatchWorkgroups(dispatchX, dispatchY, dispatchZ);
    }
}

class Timer {
    constructor() {
        this.reset();
    }

    reset() {
        this.startTime = performance.now();
        this.prev = this.startTime;
        this.currentTime = 0.0;
    }

    getTimeAndDelta(speed) {
        let now = performance.now();
        let delta = Math.min(1000 / 1, now - this.prev) * speed;
        this.prev = now;
        this.currentTime += delta;
        return [this.currentTime / 1000, delta / 1000];
    }
}

function sphericalToCartesian(spherical) {
    return [
        spherical[0] * Math.sin(spherical[1]) * Math.cos(spherical[2]),
        spherical[0] * Math.sin(spherical[1]) * Math.sin(spherical[2]),
        spherical[0] * Math.cos(spherical[1])
    ];
}

class RollingAverage {
    constructor(size = 100) {
        this.total = 0;
        this.cursor = 0;
        this.size = size;
        this.samples = [];
    }
    addSample(v) {
        this.total += v - (this.samples[this.cursor] || 0);
        this.samples[this.cursor] = v;
        this.cursor = (this.cursor + 1) % this.size;
    }
    get() {
        return this.total / this.samples.length;
    }
}

function hsvToRgb([h, s, v]) {
    let r = 0, g = 0, b = 0;
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return [r, g, b];
}
