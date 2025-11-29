/**
 * Mishkah Lite - Emergency Restoration (Correct CSS Loading)
 */
(function (window) {
    'use strict';

    console.log('🔵 [Mishkah] Starting initialization...');

    // 1. Core & Utils
    const M = window.Mishkah = window.Mishkah || {};
    M.utils = M.utils || {};
    M.UI = M.UI || {};
    M.DSL = M.DSL || {};

    console.log('🔵 [Mishkah] Created base structure');

    // Utils implementation
    M.utils.JSON = {
        parseSafe: (v) => {
            try { return JSON.parse(v); }
            catch (e) { return null; }
        },
        clone: (v) => JSON.parse(JSON.stringify(v)),
        stableStringify: JSON.stringify
    };

    console.log('🔵 [Mishkah] Utils initialized');

    // 2. CSS Auto-Loader (Correct Implementation)
    function loadCSS() {
        const config = window.MishkahAutoConfig || {};
        const scriptTag = document.querySelector('script[data-css]');
        const cssType = scriptTag ? scriptTag.getAttribute('data-css') : (config.css || null);

        console.log('🎨 [CSS Loader] CSS type:', cssType);

        if (cssType === 'mi' || cssType === 'tailwind') {
            // Use Tailwind Play CDN (Script, not Link!)
            if (!document.querySelector('script[src*="tailwindcss.com"]')) {
                const tailwindScript = document.createElement('script');
                tailwindScript.src = 'https://cdn.tailwindcss.com';
                document.head.appendChild(tailwindScript);
                console.log('✅ [CSS Loader] Tailwind Play CDN loaded');
            }
        }

        // Load custom CSS if specified (optional)
        if (config.paths && config.paths.css) {
            const customLink = document.createElement('link');
            customLink.rel = 'stylesheet';
            customLink.href = config.paths.css;
            document.head.appendChild(customLink);
            console.log('🎨 [CSS Loader] Custom CSS:', config.paths.css);
        }
    }

    // Load CSS immediately
    loadCSS();

    // 3. App Implementation
    class MishkahApp {
        constructor() {
            console.log('🟢 [MishkahApp] Constructor');
            this.state = {
                data: {
                    stats: { models: 0, experiments: 0, patterns: 0 },
                    recentExperiments: []
                }
            };
            this.template = null;
            this.mountPoint = null;
        }

        init() {
            console.log('🟢 [MishkahApp.init] Starting...');

            // Find template
            const templateEl = document.getElementById('dna-dashboard');
            if (!templateEl) {
                console.error('❌ Template #dna-dashboard NOT FOUND');
                return;
            }

            // Load state from inside template
            const dataScript = templateEl.content
                ? templateEl.content.querySelector('script[data-m-data]')
                : templateEl.querySelector('script[data-m-data]');

            if (dataScript) {
                try {
                    const parsedData = JSON.parse(dataScript.textContent);
                    this.state.data = { ...this.state.data, ...parsedData };
                    console.log('✅ State loaded:', this.state.data);
                } catch (e) {
                    console.error('❌ State parse error:', e);
                }
            }

            // Get template HTML
            this.template = templateEl.innerHTML;
            console.log('✅ Template loaded, length:', this.template.length);

            // Mount point
            this.mountPoint = document.getElementById('app');
            if (!this.mountPoint) {
                console.error('❌ Mount point #app NOT FOUND');
                return;
            }

            console.log('🟢 Rendering...');
            this.render();
        }

        setState(callback) {
            console.log('🟡 [setState] Called');
            try {
                const newState = callback(this.state);
                if (newState) this.state = newState;
                this.render();
            } catch (e) {
                console.error('❌ setState error:', e);
            }
        }

        render() {
            if (!this.template || !this.mountPoint) return;

            let html = this.template;

            // Interpolation {state.data.xxx}
            html = html.replace(/\{([^}]+)\}/g, (match, key) => {
                const trimmed = key.trim();

                // Skip JSON objects
                if (trimmed.includes('\n') || trimmed.includes('{')) {
                    return match;
                }

                const path = trimmed.split('.');
                let val = this.state;
                for (let p of path) {
                    if (p === 'state') continue;
                    val = val ? val[p] : '';
                }
                return val !== undefined && val !== null ? val : '';
            });

            // Inject
            this.mountPoint.innerHTML = html;

            // Process directives
            this.processDirectives(this.mountPoint);

            // Plotly hydration
            if (M.UI.Plotly && M.UI.Plotly.hydrate) {
                M.UI.Plotly.hydrate(this.mountPoint);
            }
        }

        processDirectives(root) {
            // x-if
            root.querySelectorAll('[x-if]').forEach(el => {
                const cond = el.getAttribute('x-if');
                try {
                    const val = new Function('state', 'return ' + cond)(this.state);
                    if (!val) el.remove();
                } catch (e) { }
            });

            // x-for
            root.querySelectorAll('[x-for]').forEach(el => {
                const expr = el.getAttribute('x-for');
                try {
                    const [item, source] = expr.split(' in ').map(s => s.trim());
                    const listPath = source.replace('state.', '').split('.');
                    let list = this.state;
                    for (let p of listPath) list = list ? list[p] : [];

                    if (!Array.isArray(list) || list.length === 0) {
                        el.remove();
                    }
                } catch (e) { }
            });

            // x-class
            root.querySelectorAll('[x-class]').forEach(el => {
                const expr = el.getAttribute('x-class');
                try {
                    const val = new Function('state', 'exp', 'return ' + expr)(this.state, {});
                    if (val) el.className = val;
                } catch (e) { }
            });
        }

        rebuild() { this.render(); }
    }

    // 4. App Factory
    M.app = {
        make: function () {
            console.log('🟣 [Mishkah.app.make] Creating app...');
            const app = new MishkahApp();

            setTimeout(() => {
                try {
                    app.init();
                } catch (e) {
                    console.error('❌ Init error:', e);
                }
            }, 0);

            return app;
        }
    };

    console.log('🔵 App factory created');

    // 5. Auto Loader
    window.MishkahAuto = {
        ready: function (cb) {
            if (document.readyState === 'complete') {
                try { cb(M); } catch (e) { console.error('❌ Callback error:', e); }
            } else {
                window.addEventListener('load', () => {
                    try { cb(M); } catch (e) { console.error('❌ Callback error:', e); }
                });
            }
        }
    };

    console.log('✅ [Mishkah] Initialization complete!');

})(window);
