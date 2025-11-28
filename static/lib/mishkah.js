/**
 * Mishkah Lite - Emergency Restoration (with Debug Logging)
 * A minimal implementation of the Mishkah framework to restore functionality.
 */
(function (window) {
    'use strict';

    console.log('🔵 [Mishkah] Starting initialization...');

    // 1. Core & Utils
    const M = window.Mishkah = window.Mishkah || {};
    M.utils = M.utils || {};
    M.UI = M.UI || {};
    M.DSL = M.DSL || {}; // Placeholder

    console.log('🔵 [Mishkah] Created base structure:', M);

    // Utils implementation for Plotly bridge
    M.utils.JSON = {
        parseSafe: (v) => {
            try {
                return JSON.parse(v);
            } catch (e) {
                console.error('❌ [Mishkah.utils.JSON.parseSafe] Parse error:', e);
                return null;
            }
        },
        clone: (v) => JSON.parse(JSON.stringify(v)),
        stableStringify: JSON.stringify
    };

    console.log('🔵 [Mishkah] Utils initialized');

    // 2. Simple State & Renderer
    class MishkahApp {
        constructor() {
            console.log('🟢 [MishkahApp] Constructor called');
            this.state = { data: {} };
            this.template = null;
            this.mountPoint = null;
        }

        init() {
            console.log('🟢 [MishkahApp.init] Starting initialization...');

            // Load initial state from script tag
            const dataScript = document.querySelector('script[data-m-data]');
            console.log('🟢 [MishkahApp.init] Looking for data-m-data script:', dataScript);

            if (dataScript) {
                try {
                    const dataText = dataScript.textContent;
                    console.log('🟢 [MishkahApp.init] Script content:', dataText.substring(0, 100) + '...');
                    this.state.data = JSON.parse(dataText);
                    console.log('✅ [MishkahApp.init] State loaded:', this.state.data);
                } catch (e) {
                    console.error('❌ [MishkahApp.init] State parse error:', e);
                }
            } else {
                console.warn('⚠️ [MishkahApp.init] No data-m-data script found');
            }

            // Find template
            const templateEl = document.getElementById('dna-dashboard');
            console.log('🟢 [MishkahApp.init] Looking for template #dna-dashboard:', templateEl);

            if (templateEl) {
                this.template = templateEl.innerHTML;
                console.log('✅ [MishkahApp.init] Template loaded, length:', this.template.length);
            } else {
                console.error('❌ [MishkahApp.init] Template #dna-dashboard NOT FOUND');
            }

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
            console.log('🟡 [MishkahApp.setState] Called with callback:', callback);
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

        // Basic Template Engine
        render() {
            console.log('🔷 [MishkahApp.render] Starting render...');

            if (!this.template) {
                console.error('❌ [MishkahApp.render] No template available');
                return;
            }

            if (!this.mountPoint) {
                console.error('❌ [MishkahApp.render] No mount point available');
                return;
            }

            let html = this.template;
            console.log('🔷 [MishkahApp.render] Template HTML length:', html.length);

            // 1. Interpolation {state.data.xxx}
            console.log('🔷 [MishkahApp.render] Starting interpolation...');
            let replacements = 0;
            html = html.replace(/\{([^}]+)\}/g, (match, key) => {
                const path = key.trim().split('.');
                let val = this.state;
                for (let p of path) {
                    if (p === 'state') continue;
                    val = val ? val[p] : '';
                }
                replacements++;
                if (replacements <= 5) { // Log first 5 replacements
                    console.log(`  🔹 Replace "${match}" with "${val}"`);
                }
                return val !== undefined ? val : '';
            });
            console.log(`✅ [MishkahApp.render] Interpolation complete. ${replacements} replacements made.`);

            // 2. Inject
            console.log('🔷 [MishkahApp.render] Injecting HTML into mount point...');
            try {
                this.mountPoint.innerHTML = html;
                console.log('✅ [MishkahApp.render] HTML injected successfully');
            } catch (e) {
                console.error('❌ [MishkahApp.render] Injection error:', e);
                return;
            }

            // 3. Post-process directives (x-if, x-for)
            console.log('🔷 [MishkahApp.render] Processing directives...');
            try {
                this.processDirectives(this.mountPoint);
                console.log('✅ [MishkahApp.render] Directives processed');
            } catch (e) {
                console.error('❌ [MishkahApp.render] Directive processing error:', e);
            }

            // 4. Hydrate Plotly if needed
            console.log('🔷 [MishkahApp.render] Checking for Plotly hydration...');
            if (M.UI.Plotly && M.UI.Plotly.hydrate) {
                console.log('🔷 [MishkahApp.render] Calling Plotly.hydrate()...');
                M.UI.Plotly.hydrate(this.mountPoint);
            } else {
                console.log('ℹ️ [MishkahApp.render] Plotly not available for hydration');
            }

            console.log('✅ [MishkahApp.render] Render complete!');
        }

        processDirectives(root) {
            console.log('🔶 [processDirectives] Starting...');

            // Handle x-if
            const xIfElements = root.querySelectorAll('[x-if]');
            console.log(`🔶 [processDirectives] Found ${xIfElements.length} x-if elements`);

            xIfElements.forEach((el, i) => {
                const cond = el.getAttribute('x-if');
                console.log(`  🔸 Processing x-if[${i}]: "${cond}"`);
                try {
                    const val = new Function('state', 'return ' + cond)(this.state);
                    console.log(`    Result: ${val}`);
                    if (!val) {
                        console.log(`    Removing element (condition false)`);
                        el.remove();
                    }
                } catch (e) {
                    console.warn(`  ⚠️ x-if error for "${cond}":`, e);
                }
            });

            // Handle x-for
            const xForElements = root.querySelectorAll('[x-for]');
            console.log(`🔶 [processDirectives] Found ${xForElements.length} x-for elements`);

            xForElements.forEach((el, i) => {
                const expr = el.getAttribute('x-for');
                console.log(`  🔸 Processing x-for[${i}]: "${expr}"`);

                try {
                    const [item, source] = expr.split(' in ').map(s => s.trim());
                    const listPath = source.replace('state.', '').split('.');
                    let list = this.state;
                    for (let p of listPath) list = list ? list[p] : [];

                    console.log(`    List length: ${Array.isArray(list) ? list.length : 'NOT AN ARRAY'}`);

                    if (!Array.isArray(list) || list.length === 0) {
                        console.log(`    Removing element (empty list)`);
                        el.remove();
                    }
                } catch (e) {
                    console.warn(`  ⚠️ x-for error for "${expr}":`, e);
                }
            });

            // Handle x-class
            const xClassElements = root.querySelectorAll('[x-class]');
            console.log(`🔶 [processDirectives] Found ${xClassElements.length} x-class elements`);

            xClassElements.forEach((el, i) => {
                const expr = el.getAttribute('x-class');
                try {
                    const val = new Function('state', 'exp', 'return ' + expr)(this.state, {});
                    if (val) {
                        el.className = val;
                        if (i < 3) console.log(`  🔸 Applied x-class[${i}]: "${val}"`);
                    }
                } catch (e) {
                    if (i < 3) console.warn(`  ⚠️ x-class error:`, e);
                }
            });

            console.log('✅ [processDirectives] Complete');
        }

        rebuild() {
            console.log('🔄 [MishkahApp.rebuild] Called');
            this.render();
        }
    }

    // 3. App Factory
    M.app = {
        make: function () {
            console.log('🟣 [Mishkah.app.make] Creating new app instance...');
            const app = new MishkahApp();

            // Defer init to ensure DOM is ready
            console.log('🟣 [Mishkah.app.make] Scheduling init() for next tick...');
            setTimeout(() => {
                console.log('⏰ [Mishkah.app.make] Timeout fired, calling app.init()...');
                try {
                    app.init();
                } catch (e) {
                    console.error('❌ [Mishkah.app.make] Init error:', e);
                    console.error('Stack trace:', e.stack);
                }
            }, 0);

            console.log('🟣 [Mishkah.app.make] Returning app instance');
            return app;
        }
    };

    console.log('🔵 [Mishkah] App factory created');

    // 4. Auto Loader
    window.MishkahAuto = {
        ready: function (cb) {
            console.log('🟤 [MishkahAuto.ready] Called, readyState:', document.readyState);
            if (document.readyState === 'complete') {
                console.log('🟤 [MishkahAuto.ready] Document already complete, calling callback immediately');
                try {
                    cb(M);
                } catch (e) {
                    console.error('❌ [MishkahAuto.ready] Callback error:', e);
                    console.error('Stack trace:', e.stack);
                }
            } else {
                console.log('🟤 [MishkahAuto.ready] Waiting for window.load event...');
                window.addEventListener('load', () => {
                    console.log('⏰ [MishkahAuto.ready] Load event fired, calling callback');
                    try {
                        cb(M);
                    } catch (e) {
                        console.error('❌ [MishkahAuto.ready] Callback error:', e);
                        console.error('Stack trace:', e.stack);
                    }
                });
            }
        }
    };

    console.log('✅ [Mishkah] Initialization complete! MishkahAuto.ready available.');
    console.log('📊 [Mishkah] Final structure:', {
        utils: !!M.utils,
        UI: !!M.UI,
        app: !!M.app,
        DSL: !!M.DSL
    });

})(window);
