/**
 * Mishkah Lite - Emergency Restoration (with CSS Auto-Loading)
 */
(function (window) {
    'use strict';

    console.log('🔵 [Mishkah] Starting initialization...');

    // 1. Core & Utils
    const M = window.Mishkah = window.Mishkah || {};
    M.utils = M.utils || {};
    M.UI = M.UI || {};
    M.DSL = M.DSL || {};

    console.log('🔵 [Mishkah] Created base structure:', M);

    // Utils implementation
    M.utils.JSON = {
        parseSafe: (v) => {
            try { return JSON.parse(v); }
            catch (e) {
                console.error('❌ [Mishkah.utils.JSON.parseSafe] Parse error:', e);
                return null;
            }
        },
        clone: (v) => JSON.parse(JSON.stringify(v)),
        stableStringify: JSON.stringify
    };

    console.log('🔵 [Mishkah] Utils initialized');

    // 2. CSS Auto-Loader
    function loadCSS() {
        const config = window.MishkahAutoConfig || {};
        const cssType = config.css;

        console.log('🎨 [CSS Loader] Config css type:', cssType);

        if (cssType === 'mi' || cssType === 'tailwind') {
            // Load Tailwind CSS
            const tailwindLink = document.createElement('link');
            tailwindLink.rel = 'stylesheet';
            tailwindLink.href = 'https://cdn.jsdelivr.net/npm/tailwindcss@3.4.0/dist/tailwind.min.css';
            document.head.appendChild(tailwindLink);
            console.log('✅ [CSS Loader] Tailwind CSS loaded from CDN');
        }

        // Load custom CSS if specified
        if (config.paths && config.paths.css) {
            const customLink = document.createElement('link');
            customLink.rel = 'stylesheet';
            customLink.href = config.paths.css;
            document.head.appendChild(customLink);
            console.log('✅ [CSS Loader] Custom CSS loaded:', config.paths.css);
        }
    }

    // Load CSS immediately
    loadCSS();

    // 3. App Implementation
    class MishkahApp {
        constructor() {
            console.log('🟢 [MishkahApp] Constructor called');
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
            console.log('🟢 [MishkahApp.init] Starting initialization...');

            // Find template first
            const templateEl = document.getElementById('dna-dashboard');
            console.log('🟢 [MishkahApp.init] Looking for template #dna-dashboard:', templateEl);

            if (!templateEl) {
                console.error('❌ [MishkahApp.init] Template #dna-dashboard NOT FOUND');
                return;
            }

            // Load state from data-m-data INSIDE template
            const dataScript = templateEl.content
                ? templateEl.content.querySelector('script[data-m-data]')
                : templateEl.querySelector('script[data-m-data]');

            console.log('🟢 [MishkahApp.init] Looking for data-m-data inside template:', dataScript);

            if (dataScript) {
                try {
                    const dataText = dataScript.textContent;
                    console.log('🟢 [MishkahApp.init] Script content:', dataText.substring(0, 100) + '...');
                    const parsedData = JSON.parse(dataText);
                    // Merge with default state
                    this.state.data = { ...this.state.data, ...parsedData };
                    console.log('✅ [MishkahApp.init] State loaded:', this.state.data);
                } catch (e) {
                    console.error('❌ [MishkahApp.init] State parse error:', e);
                }
            } else {
                console.warn('⚠️ [MishkahApp.init] No data-m-data script found, using defaults');
            }

            // Get template HTML
            this.template = templateEl.innerHTML;
            console.log('✅ [MishkahApp.init] Template loaded, length:', this.template.length);

            // Mount point
            this.mountPoint = document.getElementById('app');
            console.log('🟢 [MishkahApp.init] Looking for mount point #app:', this.mountPoint);

            if (!this.mountPoint) {
                console.error('❌ [MishkahApp.init] Mount point #app NOT FOUND');
                return;
            }

            console.log('🟢 [MishkahApp.init] All checks passed, calling render()...');
            this.render();
        }

        setState(callback) {
            console.log('🟡 [MishkahApp.setState] Called');
            try {
                const newState = callback(this.state);
                if (newState) {
                    this.state = newState;
                    console.log('✅ [MishkahApp.setState] State updated:', this.state);
                }
                this.render();
            } catch (e) {
                console.error('❌ [MishkahApp.setState] Error:', e);
            }
        }

        render() {
            console.log('🔷 [MishkahApp.render] Starting render...');

            if (!this.template || !this.mountPoint) {
                console.error('❌ [MishkahApp.render] Missing template or mount point');
                return;
            }

            let html = this.template;

            // Interpolation {state.data.xxx}
            console.log('🔷 [MishkahApp.render] Starting interpolation...');
            let replacements = 0;
            html = html.replace(/\{([^}]+)\}/g, (match, key) => {
                const trimmed = key.trim();

                // Skip JSON objects (multi-line)
                if (trimmed.includes('\n') || trimmed.includes('{')) {
                    return match;
                }

                const path = trimmed.split('.');
                let val = this.state;
                for (let p of path) {
                    if (p === 'state') continue;
                    val = val ? val[p] : '';
                }
                replacements++;
                if (replacements <= 5) {
                    console.log(`  🔹 Replace "${match}" with "${val}"`);
                }
                return val !== undefined && val !== null ? val : '';
            });
            console.log(`✅ [MishkahApp.render] ${replacements} replacements made`);

            // Inject
            console.log('🔷 [MishkahApp.render] Injecting HTML...');
            try {
                this.mountPoint.innerHTML = html;
                console.log('✅ [MishkahApp.render] HTML injected');
            } catch (e) {
                console.error('❌ [MishkahApp.render] Injection error:', e);
                return;
            }

            // Process directives
            console.log('🔷 [MishkahApp.render] Processing directives...');
            try {
                this.processDirectives(this.mountPoint);
                console.log('✅ [MishkahApp.render] Directives processed');
            } catch (e) {
                console.error('❌ [MishkahApp.render] Directive error:', e);
            }

            // Plotly hydration
            if (M.UI.Plotly && M.UI.Plotly.hydrate) {
                console.log('🔷 [MishkahApp.render] Calling Plotly.hydrate()...');
                M.UI.Plotly.hydrate(this.mountPoint);
            }

            console.log('✅ [MishkahApp.render] Render complete!');
        }

        processDirectives(root) {
            console.log('🔶 [processDirectives] Starting...');

            // x-if
            const xIfElements = root.querySelectorAll('[x-if]');
            console.log(`🔶 [processDirectives] Found ${xIfElements.length} x-if elements`);

            xIfElements.forEach((el, i) => {
                const cond = el.getAttribute('x-if');
                try {
                    const val = new Function('state', 'return ' + cond)(this.state);
                    if (!val) el.remove();
                } catch (e) {
                    console.warn(`  ⚠️ x-if[${i}] error:`, e.message);
                }
            });

            // x-for
            const xForElements = root.querySelectorAll('[x-for]');
            console.log(`🔶 [processDirectives] Found ${xForElements.length} x-for elements`);

            xForElements.forEach((el, i) => {
                const expr = el.getAttribute('x-for');
                try {
                    const [item, source] = expr.split(' in ').map(s => s.trim());
                    const listPath = source.replace('state.', '').split('.');
                    let list = this.state;
                    for (let p of listPath) list = list ? list[p] : [];

                    if (!Array.isArray(list) || list.length === 0) {
                        el.remove();
                    }
                } catch (e) {
                    console.warn(`  ⚠️ x-for[${i}] error:`, e.message);
                }
            });

            // x-class
            const xClassElements = root.querySelectorAll('[x-class]');
            xClassElements.forEach((el) => {
                const expr = el.getAttribute('x-class');
                try {
                    const val = new Function('state', 'exp', 'return ' + expr)(this.state, {});
                    if (val) el.className = val;
                } catch (e) { }
            });

            console.log('✅ [processDirectives] Complete');
        }

        rebuild() {
            console.log('🔄 [MishkahApp.rebuild] Called');
            this.render();
        }
    }

    // 4. App Factory
    M.app = {
        make: function () {
            console.log('🟣 [Mishkah.app.make] Creating new app instance...');
            const app = new MishkahApp();

            setTimeout(() => {
                console.log('⏰ [Mishkah.app.make] Calling app.init()...');
                try {
                    app.init();
                } catch (e) {
                    console.error('❌ [Mishkah.app.make] Init error:', e);
                }
            }, 0);

            console.log('🟣 [Mishkah.app.make] Returning app instance');
            return app;
        }
    };

    console.log('🔵 [Mishkah] App factory created');

    // 5. Auto Loader
    window.MishkahAuto = {
        ready: function (cb) {
            console.log('🟤 [MishkahAuto.ready] Called, readyState:', document.readyState);
            if (document.readyState === 'complete') {
                console.log('🟤 [MishkahAuto.ready] Calling callback immediately');
                try {
                    cb(M);
                } catch (e) {
                    console.error('❌ [MishkahAuto.ready] Callback error:', e);
                }
            } else {
                console.log('🟤 [MishkahAuto.ready] Waiting for window.load...');
                window.addEventListener('load', () => {
                    console.log('⏰ [MishkahAuto.ready] Load event fired');
                    try {
                        cb(M);
                    } catch (e) {
                        console.error('❌ [MishkahAuto.ready] Callback error:', e);
                    }
                });
            }
        }
    };

    console.log('✅ [Mishkah] Initialization complete!');

})(window);
