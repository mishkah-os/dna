/**
 * ═══════════════════════════════════════════════════════════════════════════
 * Mishkah Scaffold System - نظام السقالات
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * نظام ذكي لتحميل مكتبات Mishkah بشكل مشروط حسب الحاجة
 * يدعم سيناريوهات التطوير والإنتاج ويحافظ على صغر حجم الكود
 * 
 * @version 1.0.1
 * @author Mishkah Team
 */

(function (window) {
    'use strict';

    // ═══════════════════════════════════════════════════════════════════════════
    // كشف المسار التلقائي من مكان scaffold.js
    // ═════════════════════════════════════════════════════════════════════════

    function _detectBasePath() {
        // محاولة كشف المسار من script tag الحالي
        var scripts = document.getElementsByTagName('script');
        for (var i = scripts.length - 1; i >= 0; i--) {
            var src = scripts[i].src;
            if (src && src.indexOf('mishkah.scaffold.js') !== -1) {
                // استخراج المسار الأساسي
                var path = src.substring(0, src.lastIndexOf('/') + 1);
                return path;
            }
        }
        return '/lib/'; // افتراضي بسيط
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // الإعدادات الافتراضية
    // ═══════════════════════════════════════════════════════════════════════════

    var DEFAULT_CONFIG = {
        mode: 'dev',              // dev | prod | debug | minimal
        basePath: null,           // سيتم حسابه تلقائياً

        // الطبقات التشخيصية
        diagnostics: {
            div: true,              // mishkah.div.js - نظام القواعد والتحقق
            help: true,             // mishkah.help.js - نظام المساعدة للمطورين
            performance: false,     // مراقبة الأداء (سيتم إضافته لاحقاً)
            security: false         // فحوصات أمنية إضافية
        },

        // المكتبات الأساسية
        features: {
            core: true,             // mishkah.core.js
            utils: true,            // mishkah-utils.js
            ui: true,               // mishkah-ui.js
            htmlx: true,            // mishkah-htmlx.js
            store: false,           // mishkah.store.js
            crud: false,            // mishkah.crud.js
            pages: false            // mishkah.pages.js
        },

        // خيارات التحميل
        loading: {
            async: false,           // تحميل متزامن لضمان الترتيب
            defer: false,           // تأجيل التحميل
            timeout: 10000,         // وقت الانتظار الأقصى (مللي ثانية)
            retry: 2                // عدد المحاولات عند الفشل
        },

        // الـ CDN (اختياري)
        cdn: {
            enabled: false,
            baseUrl: 'https://cdn.example.com/mishkah/'
        },

        // callbacks
        onReady: null,            // يُنفذ عند اكتمال التحميل
        onError: null,            // يُنفذ عند حدوث خطأ
        onProgress: null          // يُنفذ أثناء التحميل
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // قراءة الإعدادات من مصادر متعددة
    // ═══════════════════════════════════════════════════════════════════════════

    function _readConfig() {
        var config = _deepClone(DEFAULT_CONFIG);

        // 1. قراءة من window.__MISHKAH_CONFIG__
        if (window.__MISHKAH_CONFIG__) {
            _deepMerge(config, window.__MISHKAH_CONFIG__);
        }

        // 2. كشف المسار تلقائياً إذا لم يتم تحديده
        if (!config.basePath) {
            config.basePath = _detectBasePath();
            _log('🔍 Auto-detected basePath: ' + config.basePath);
        }

        // 3. قراءة من URL parameters (للتطوير)
        var urlParams = _parseUrlParams();
        if (urlParams.mishkah_mode) {
            config.mode = urlParams.mishkah_mode;
        }
        if (urlParams.mishkah_debug === 'true') {
            config.diagnostics.div = true;
            config.diagnostics.help = true;
            config.diagnostics.performance = true;
        }

        // 4. تطبيق سيناريوهات الأوضاع المعرّفة مسبقاً
        _applyModePreset(config);

        return config;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // تطبيق السيناريوهات المعرّفة مسبقاً
    // ═══════════════════════════════════════════════════════════════════════════

    function _applyModePreset(config) {
        var mode = config.mode.toLowerCase();

        switch (mode) {
            case 'minimal':
                // الحد الأدنى: فقط الأساسيات
                config.diagnostics.div = false;
                config.diagnostics.help = false;
                config.diagnostics.performance = false;
                config.features.store = false;
                config.features.crud = false;
                config.features.pages = false;
                break;

            case 'prod':
            case 'production':
                // الإنتاج: بدون طبقات تشخيصية
                config.diagnostics.div = false;
                config.diagnostics.help = false;
                config.diagnostics.performance = false;
                break;

            case 'debug':
                // التشخيص الكامل: كل شيء مفعّل
                config.diagnostics.div = true;
                config.diagnostics.help = true;
                config.diagnostics.performance = true;
                config.diagnostics.security = true;
                break;

            case 'dev':
            case 'development':
            default:
                // التطوير: طبقات تشخيصية أساسية
                config.diagnostics.div = true;
                config.diagnostics.help = true;
                break;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // بناء قائمة الملفات للتحميل
    // ══════════════════════════════════════════════════════════════════════════

    function _buildLoadingQueue(config) {
        var queue = [];
        var f = config.features;
        var d = config.diagnostics;

        // ترتيب التحميل مهم جداً!

        // 1. Core (دائماً أولاً)
        if (f.core) {
            queue.push({
                name: 'core',
                path: 'mishkah.core.js',
                required: true,
                diagnostic: false
            });
        }

        // 2. Utils (ثانياً)
        if (f.utils) {
            queue.push({
                name: 'utils',
                path: 'mishkah-utils.js',
                required: true,
                diagnostic: false
            });
        }

        // 3. UI Components
        if (f.ui) {
            queue.push({
                name: 'ui',
                path: 'mishkah-ui.js',
                required: false,
                diagnostic: false
            });
        }

        // 4. HTMLx (يعتمد على utils و ui)
        if (f.htmlx) {
            queue.push({
                name: 'htmlx',
                path: 'mishkah-htmlx.js',
                required: false,
                diagnostic: false
            });
        }

        // 5. Store
        if (f.store) {
            queue.push({
                name: 'store',
                path: 'mishkah.store.js',
                required: false,
                diagnostic: false
            });
        }

        // 6. CRUD
        if (f.crud) {
            queue.push({
                name: 'crud',
                path: 'mishkah.crud.js',
                required: false,
                diagnostic: false
            });
        }

        // 7. Pages
        if (f.pages) {
            queue.push({
                name: 'pages',
                path: 'mishkah.pages.js',
                required: false,
                diagnostic: false
            });
        }

        // ═══════════════════════════════════════════════════════════════════════
        // الطبقات التشخيصية (تُحمّل بعد المكتبات الأساسية)
        // ═══════════════════════════════════════════════════════════════════════

        // 8. Div (RuleCenter - نظام القواعد)
        if (d.div) {
            queue.push({
                name: 'div',
                path: 'mishkah.div.js',
                required: false,
                diagnostic: true
            });
        }

        // 9. Help (نظام المساعدة)
        if (d.help) {
            queue.push({
                name: 'help',
                path: 'mishkah.help.js',
                required: false,
                diagnostic: true
            });
        }

        // 10. Performance Monitor (مستقبلاً)
        if (d.performance) {
            queue.push({
                name: 'performance',
                path: 'mishkah.perf.js',
                required: false,
                diagnostic: true
            });
        }

        return queue;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // تحميل السكريبتات بالترتيب
    // ═══════════════════════════════════════════════════════════════════════════

    function _loadScripts(queue, config, callback) {
        var loaded = [];
        var failed = [];
        var index = 0;
        var basePath = config.cdn.enabled ? config.cdn.baseUrl : config.basePath;

        function loadNext() {
            if (index >= queue.length) {
                // اكتمل التحميل
                _onComplete(loaded, failed, config);
                if (callback) callback(null, { loaded: loaded, failed: failed });
                return;
            }

            var item = queue[index];
            var url = basePath + item.path;

            _log('📦 Loading: ' + item.name + ' (' + item.path + ')');

            // تقدم التحميل
            if (config.onProgress) {
                config.onProgress({
                    current: index + 1,
                    total: queue.length,
                    item: item
                });
            }

            _loadScript(url, config.loading.timeout, config.loading.retry, function (err) {
                if (err) {
                    _warn('❌ Failed to load: ' + item.name + ' - ' + err);
                    failed.push({ item: item, error: err });

                    // إذا كان مطلوباً، نتوقف
                    if (item.required) {
                        _error('🛑 Required library failed: ' + item.name);
                        if (callback) callback(err);
                        if (config.onError) config.onError(err, item);
                        return;
                    }
                } else {
                    _log('✅ Loaded: ' + item.name);
                    loaded.push(item);
                }

                index++;
                loadNext();
            });
        }

        loadNext();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // تحميل سكريبت واحد مع إعادة المحاولة
    // ═══════════════════════════════════════════════════════════════════════════

    function _loadScript(url, timeout, retries, callback) {
        var attempts = 0;

        function attempt() {
            attempts++;

            var script = document.createElement('script');
            script.src = url;
            script.type = 'text/javascript';

            var timeoutId = setTimeout(function () {
                script.onerror = null;
                script.onload = null;
                if (attempts <= retries) {
                    _warn('⏱️ Timeout loading ' + url + ', retrying... (' + attempts + '/' + retries + ')');
                    attempt();
                } else {
                    callback(new Error('Timeout after ' + retries + ' retries'));
                }
            }, timeout);

            script.onload = function () {
                clearTimeout(timeoutId);
                callback(null);
            };

            script.onerror = function (e) {
                clearTimeout(timeoutId);
                if (attempts <= retries) {
                    _warn('❌ Error loading ' + url + ', retrying... (' + attempts + '/' + retries + ')');
                    attempt();
                } else {
                    callback(new Error('Failed to load script: ' + url));
                }
            };

            document.head.appendChild(script);
        }

        attempt();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // عند اكتمال التحميل
    // ═══════════════════════════════════════════════════════════════════════════

    function _onComplete(loaded, failed, config) {
        _log('🎉 Mishkah scaffolding complete!');
        _log('   Loaded: ' + loaded.length + ' modules');

        if (failed.length > 0) {
            _warn('   Failed: ' + failed.length + ' modules');
            failed.forEach(function (f) {
                _warn('     - ' + f.item.name);
            });
        }

        // حفظ معلومات التحميل
        window.__MISHKAH_SCAFFOLD__ = {
            config: config,
            loaded: loaded,
            failed: failed,
            timestamp: new Date().toISOString()
        };

        // تنفيذ callback
        if (config.onReady) {
            config.onReady({
                loaded: loaded,
                failed: failed,
                config: config
            });
        }

        // إطلاق حدث مخصص
        if (window.dispatchEvent) {
            window.dispatchEvent(new CustomEvent('mishkah:ready', {
                detail: {
                    loaded: loaded,
                    failed: failed,
                    config: config
                }
            }));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Utilities
    // ═══════════════════════════════════════════════════════════════════════════

    function _parseUrlParams() {
        var params = {};
        var search = window.location.search.substring(1);
        if (!search) return params;

        search.split('&').forEach(function (pair) {
            var parts = pair.split('=');
            if (parts.length === 2) {
                params[decodeURIComponent(parts[0])] = decodeURIComponent(parts[1]);
            }
        });

        return params;
    }

    function _deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    }

    function _deepMerge(target, source) {
        for (var key in source) {
            if (source.hasOwnProperty(key)) {
                if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                    target[key] = target[key] || {};
                    _deepMerge(target[key], source[key]);
                } else {
                    target[key] = source[key];
                }
            }
        }
        return target;
    }

    function _log(msg) {
        if (console && console.log) {
            console.log('[Mishkah Scaffold] ' + msg);
        }
    }

    function _warn(msg) {
        if (console && console.warn) {
            console.warn('[Mishkah Scaffold] ' + msg);
        }
    }

    function _error(msg) {
        if (console && console.error) {
            console.error('[Mishkah Scaffold] ' + msg);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Public API
    // ═══════════════════════════════════════════════════════════════════════════

    var MishkahScaffold = {
        version: '1.0.1',

        /**
         * بدء التحميل (يُنفذ تلقائياً)
         */
        boot: function (customConfig, callback) {
            _log('🚀 Starting Mishkah scaffolding...');

            // دمج الإعدادات المخصصة
            if (customConfig) {
                window.__MISHKAH_CONFIG__ = window.__MISHKAH_CONFIG__ || {};
                _deepMerge(window.__MISHKAH_CONFIG__, customConfig);
            }

            var config = _readConfig();
            var queue = _buildLoadingQueue(config);

            _log('📋 Loading queue: ' + queue.map(function (q) { return q.name; }).join(', '));
            _log('🔧 Mode: ' + config.mode);

            _loadScripts(queue, config, callback);
        },

        /**
         * الحصول على الإعدادات الحالية
         */
        getConfig: function () {
            return window.__MISHKAH_SCAFFOLD__ ? window.__MISHKAH_SCAFFOLD__.config : null;
        },

        /**
         * الحصول على حالة التحميل
         */
        getStatus: function () {
            return window.__MISHKAH_SCAFFOLD__ || null;
        },

        /**
         * إعادة التحميل
         */
        reload: function (customConfig, callback) {
            _log('🔄 Reloading Mishkah...');
            this.boot(customConfig, callback);
        }
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Auto-boot عند تحميل الملف (إلا إذا تم التعطيل)
    // ═══════════════════════════════════════════════════════════════════════════

    if (typeof window !== 'undefined') {
        window.MishkahScaffold = MishkahScaffold;

        // Auto-boot (إلا إذا كان __MISHKAH_MANUAL_BOOT__ = true)
        if (!window.__MISHKAH_MANUAL_BOOT__) {
            // انتظار DOM
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', function () {
                    MishkahScaffold.boot();
                });
            } else {
                // DOM جاهز
                MishkahScaffold.boot();
            }
        }
    }

})(window);
