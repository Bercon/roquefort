
class Configuration {
    constructor() {
        this.config = {
            'camPosRadius': 3.5,
            'camPosTheta': Math.PI / 2. - 0.3,
            'camPosPhi': Math.PI * 0.25
        }
        this.userConfigs = {};
        this.listeners = {};
        this.storedConfig = window.localStorage.getItem('config');
        this.autosave = true;
    }

    addListener(key, fn) {
        if (!this.listeners[key]) this.listeners[key] = []
        this.listeners[key].push(fn);
    }

    update(key, val) {
        if (this.config[key] === val) return;
        this.config[key] = val;
        if (this.listeners[key]) {
            for (let fn of this.listeners[key]) {
                fn(val);
            }
        }
        if (this.autosave && this.userConfigs.hasOwnProperty(key)) {
            this.saveConfig();
        }
    }

    updateNumber(key, val) {
        let n = Number(val);
        if (isFinite(n)) this.update(key, Number(val));
    }

    addTheta(val) {
        let theta = this.config.camPosTheta + val;
        if (theta > Math.PI * .99) theta = Math.PI * .99;
        if (theta < 0.01) theta = 0.01;
        this.updateNumber('camPosTheta', theta);
    }

    addPhi(val) {
        let phi = this.config.camPosPhi + val;
        while (phi > Math.PI) phi -= Math.PI * 2;
        while (phi < -Math.PI) phi += Math.PI * 2;
        this.updateNumber('camPosPhi', phi);
    }

    addRadius(val) {
        let radius = this.config.camPosRadius + val;
        if (radius < 0.1) radius = 0.1;
        if (radius > 6.0) radius = 6.0;
        this.updateNumber('camPosRadius', radius);
    }

    addUserConfig(key, defaultValue) {
        this.userConfigs[key] = defaultValue;
        this.update(key, defaultValue);
    }

    addUserConfigNumber(key, defaultValue) {
        this.userConfigs[key] = defaultValue;
        this.updateNumber(key, defaultValue);
    }

    resetConfig() {
        this.autosave = false;
        Object.entries(this.userConfigs).forEach(([key, value]) => this.update(key, value));
        this.autosave = true;
        this.saveConfig();
    }

    saveConfig() {
        const c = Object.fromEntries(Object.entries(this.config).filter(([key, value]) => this.userConfigs[key] !== undefined));
        // console.log('save', c);
        window.localStorage.setItem('config', JSON.stringify(c));
    }

    loadConfig() {
        if (this.storedConfig) {
            try {
                const parsedConfig = JSON.parse(this.storedConfig);
                const isValid = Object.keys(this.userConfigs).every(key => key in parsedConfig);
                if (isValid) {
                    // console.log('load', parsedConfig);
                    Object.entries(parsedConfig).forEach(([key, value]) => this.update(key, value));
                    return;
                } else {
                    console.warn('Loaded config is missing some required fields. Reverting to defaults.');
                    this.saveConfig();
                    return;
                }
            } catch (e) {
                console.error('Failed to parse the stored config. Reverting to defaults.', e);
                this.saveConfig();
                return;
            }
        }
    }
}

class ConfigurationPresets {
    constructor(configuration, presetConfigs, onChange) {
        this.configuration = configuration;
        this.presets = {};
        this.onChange = onChange;
        this.noMatching = presetConfigs[0];
        const affectedKeys = new Set();
        presetConfigs.forEach(p => {
            if (p.props) {
                Object.keys(p.props).forEach(k => affectedKeys.add(k));
                this.presets[p.label] = p;
            }
        });
        affectedKeys.forEach(k => configuration.addListener(k, this.onUpdate));
        this.onUpdate();
    }
    isPresetActive(preset) {
        for (const [key, value] of Object.entries(preset.props)) {
            if (this.configuration.config[key] !== value) {
                return false;
            }
        }
        return true;
    }

    onUpdate = () => {
        for (const preset of Object.values(this.presets)) {
            if (this.isPresetActive(preset)) {
                this.onChange(preset);
                return;
            }
        };
        this.onChange(this.noMatching);
    }

    apply = (label) => {
        if (!this.presets.hasOwnProperty(label)) {
            this.onUpdate();
            return;
        }
        const props = this.presets[label].props;
        for (const [key, value] of Object.entries(props)) {
            this.configuration.update(key, value);
        };
    }

}
