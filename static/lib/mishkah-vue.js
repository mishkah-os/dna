!function (root, factory) { if (typeof define === 'function' && define.amd) { define(['mishkah'], function (M) { return factory(root, M) }) } else if (typeof module === 'object' && module.exports) { module.exports = factory(root, require('mishkah')) } else { root.Mishkah = root.Mishkah || {}; root.Mishkah.Vue = factory(root, root.Mishkah) } }(typeof window !== 'undefined' ? window : this, function (global, M) {
    'use strict';
    var targetMap = new WeakMap;
    var activeEffect = null;
    var shouldTrack = false;
    function track(target, key) { if (!shouldTrack || activeEffect === undefined) return; var depsMap = targetMap.get(target); if (!depsMap) { targetMap.set(target, depsMap = new Map) } var dep = depsMap.get(key); if (!dep) { depsMap.set(key, dep = new Set) } if (!dep.has(activeEffect)) { dep.add(activeEffect); activeEffect.deps.push(dep) } }
    function trigger(target, key) { var depsMap = targetMap.get(target); if (!depsMap) return; var effects = new Set; var deps = depsMap.get(key); if (deps) { deps.forEach(function (effect) { if (effect !== activeEffect) { effects.add(effect) } }) } effects.forEach(function (effect) { if (effect.scheduler) { effect.scheduler() } else { effect.run() } }) }
    function effect(fn, options) { var _effect = new ReactiveEffect(fn); if (options) { Object.assign(_effect, options) } if (!options || !options.lazy) { _effect.run() } var runner = _effect.run.bind(_effect); runner.effect = _effect; return runner }
    var ReactiveEffect = function (fn, scheduler) { this.deps = []; this.fn = fn; this.scheduler = scheduler; this.active = true }; ReactiveEffect.prototype.run = function () { if (!this.active) { return this.fn() } if (!effectStack.includes(this)) { try { effectStack.push(this); activeEffect = this; shouldTrack = true; return this.fn() } finally { effectStack.pop(); activeEffect = effectStack[effectStack.length - 1]; shouldTrack = effectStack.length > 0 } } }; ReactiveEffect.prototype.stop = function () { if (this.active) { cleanupEffect(this); if (this.onStop) { this.onStop() } this.active = false } };
    function cleanupEffect(effect) { effect.deps.forEach(function (dep) { dep.delete(effect) }); effect.deps.length = 0 }
    var effectStack = [];
    function reactive(raw) { if (raw && raw.__v_isReactive) { return raw } return new Proxy(raw, { get: function (target, key, receiver) { if (key === '__v_isReactive') return true; var res = Reflect.get(target, key, receiver); track(target, key); if (res != null && typeof res === 'object') { return reactive(res) } return res }, set: function (target, key, value, receiver) { var oldValue = target[key]; var res = Reflect.set(target, key, value, receiver); if (value !== oldValue) { trigger(target, key) } return res } }) }
    function ref(value) { return new RefImpl(value) }
    var RefImpl = function (value) { this._rawValue = value; this._value = convert(value); this.__v_isRef = true; this.dep = new Set };
    RefImpl.prototype.toString = function () { trackRefValue(this); return String(this._value); };
    Object.defineProperty(RefImpl.prototype, "value", { get: function () { trackRefValue(this); return this._value }, set: function (newVal) { if (newVal !== this._rawValue) { this._rawValue = newVal; this._value = convert(newVal); triggerRefValue(this) } }, enumerable: true, configurable: true });
    function convert(val) { return val != null && typeof val === 'object' ? reactive(val) : val }
    function trackRefValue(ref) { if (shouldTrack && activeEffect) { if (!ref.dep) { ref.dep = new Set } if (!ref.dep.has(activeEffect)) { ref.dep.add(activeEffect); activeEffect.deps.push(ref.dep) } } }
    function triggerRefValue(ref) { if (ref.dep) { var effects = new Set(ref.dep); effects.forEach(function (effect) { if (effect !== activeEffect) { if (effect.scheduler) { effect.scheduler() } else { effect.run() } } }) } }
    function computed(getter) { var _value; var _dirty = true; var _effect = new ReactiveEffect(getter, function () { if (!_dirty) { _dirty = true; triggerRefValue(cRef) } }); var cRef = { __v_isRef: true, get value() { if (_dirty) { _value = _effect.run(); _dirty = false } trackRefValue(cRef); return _value }, toString: function () { return String(this.value); } }; return cRef }
    var queue = []; var isFlushing = false; var resolvedPromise = Promise.resolve();
    function queueJob(job) { if (!queue.includes(job)) { queue.push(job); queueFlush() } }
    function queueFlush() { if (!isFlushing) { isFlushing = true; resolvedPromise.then(flushJobs) } }
    function flushJobs() { try { for (var i = 0; i < queue.length; i++) { var job = queue[i]; if (job && job.active !== false) { job() } } } finally { isFlushing = false; queue.length = 0 } }
    function nextTick(fn) { return fn ? resolvedPromise.then(fn) : resolvedPromise }
    var currentInstance = null;
    function createComponentInstance(vnode) { var instance = { vnode: vnode, type: vnode.type, setupState: {}, isMounted: false, subTree: null, update: null, effects: [], provides: Object.create(null), ctx: {} }; instance.ctx._ = instance; return instance }
    function setupComponent(instance) { var setup = instance.type.setup; if (setup) { currentInstance = instance; var setupResult = setup(instance.type.props || {}, { emit: function () { } }); currentInstance = null; handleSetupResult(instance, setupResult) } else { finishComponentSetup(instance) } }
    function handleSetupResult(instance, setupResult) { if (typeof setupResult === 'function') { instance.render = setupResult } else if (typeof setupResult === 'object') { instance.setupState = setupResult } finishComponentSetup(instance) }
    function finishComponentSetup(instance) { if (!instance.render) { instance.render = instance.type.render || instance.type.template } }
    function mountComponent(vnode, container, anchor) { var instance = vnode.component = createComponentInstance(vnode); setupComponent(instance); setupRenderEffect(instance, vnode, container) }
    function attachEvents(vnode) { if (!vnode) return; if (vnode._vueEvents && vnode._dom) { for (var ev in vnode._vueEvents) { vnode._dom.addEventListener(ev, vnode._vueEvents[ev]) } } if (vnode.children) { for (var i = 0; i < vnode.children.length; i++) { attachEvents(vnode.children[i]) } } }
    function setupRenderEffect(instance, initialVNode, container) { instance.update = effect(function componentEffect() { if (!instance.isMounted) { var subTree = instance.subTree = renderComponentRoot(instance); M.VDOM.patch(container, subTree, null, {}, {}, ""); attachEvents(subTree); initialVNode.el = subTree.el ? subTree.el : container.firstChild; instance.isMounted = true; if (instance.mounted) { instance.mounted.forEach(function (hook) { hook() }) } } else { var nextTree = renderComponentRoot(instance); var prevTree = instance.subTree; instance.subTree = nextTree; var el = prevTree.el; M.VDOM.patch(el ? el.parentNode : container, nextTree, prevTree, {}, {}, ""); attachEvents(nextTree); instance.vnode.el = nextTree.el } }, { scheduler: function () { queueJob(instance.update) } }) }
    function renderComponentRoot(instance) { var render = instance.render; if (!render) return null; var proxy = new Proxy(instance.setupState, { get: function (target, key) { var val; if (key in target) val = target[key]; else if (key in instance.ctx) val = instance.ctx[key]; else return undefined; return (val && val.__v_isRef) ? val.value : val; }, set: function (target, key, value) { if (key in target) { var curr = target[key]; if (curr && curr.__v_isRef && !value.__v_isRef) { curr.value = value; return true } target[key] = value; return true } return false } }); return render.call(proxy, proxy) }
    var gkeyCounter = 0;
    function h(type, props, children) { if (typeof type === 'string') { var cfg = { attrs: {} }; var events = {}; if (props) { for (var k in props) { if (k.indexOf('on') === 0 && typeof props[k] === 'function') { events[k.toLowerCase().substring(2)] = props[k] } else if (k === 'class' || k === 'className') { cfg.attrs['class'] = props[k] } else { cfg.attrs[k] = props[k] } } } if (Object.keys(events).length > 0) { cfg.attrs.gkey = 'vue:' + (++gkeyCounter) } var vnode = M.h(type, 'Vue', cfg, children != null ? children : []); vnode._vueEvents = events; return vnode } return { tag: type, type: type, props: props || {}, children: children || [], category: 'Vue', key: props ? props.key : null } }
    function createApp(rootComponent) { return { mount: function (selector) { var container = typeof selector === 'string' ? document.querySelector(selector) : selector; var vnode = { tag: rootComponent, type: rootComponent, props: {}, children: [], category: 'Vue', appContext: {} }; mountComponent(vnode, container) } } }
    function inject(key, defaultValue) { if (currentInstance) { if (key in currentInstance.provides) { return currentInstance.provides[key] } else if (currentInstance.parent) { return currentInstance.parent.provides[key] } } return defaultValue }
    function provide(key, value) { if (currentInstance) { currentInstance.provides[key] = value } }
    function onMounted(fn) { if (currentInstance) { if (!currentInstance.mounted) currentInstance.mounted = []; currentInstance.mounted.push(fn) } }
    function onUnmounted(fn) { if (currentInstance) { if (!currentInstance.unmounted) currentInstance.unmounted = []; currentInstance.unmounted.push(fn) } }
    function onUpdated(fn) { if (currentInstance) { if (!currentInstance.updated) currentInstance.updated = []; currentInstance.updated.push(fn) } }

    // GLOBAL INIT & STATE
    var globalState = reactive({ db: { data: {}, env: { lang: 'ar', dir: 'rtl', theme: 'dark' }, i18n: { dict: {} } }, theme: { tokens: {}, vars: {}, current: 'dark' } });
    function initMishkah(config) {
        if (config) {
            if (config.env) Object.assign(globalState.db.env, config.env);
            if (config.i18n) globalState.db.i18n = config.i18n;
            // Merge arbitrary data
            for (var k in config) {
                if (k !== 'env' && k !== 'i18n') globalState.db.data[k] = config[k];
            }
        }
        // Sync with Core if available
        if (global.Mishkah && global.Mishkah.Database) {
            global.Mishkah.Database = globalState.db; // Overwrite or Sync? Best to sync
        }
        updateThemeVariables();
    }
    function useDatabase() { return { data: computed(function () { return globalState.db.data }), get: function (path) { var parts = path.split('.'); var val = globalState.db.data; for (var i = 0; i < parts.length; i++) { val = val ? val[parts[i]] : null } return val }, set: function (path, value) { var parts = path.split('.'); var obj = globalState.db.data; for (var i = 0; i < parts.length - 1; i++) { if (!obj[parts[i]]) obj[parts[i]] = {}; obj = obj[parts[i]] } obj[parts[parts.length - 1]] = value } } }
    function useI18n() {
        var t = function (key) {
            var lang = globalState.db.env.lang || 'en';
            var dict = globalState.db.i18n.dict || {};
            // 1. Try Key-based (User Preference): dict.title.ar
            if (dict[key] && dict[key][lang]) return dict[key][lang];
            // 2. Try Lang-based (Legacy/Provided Data): dict.ar.title
            if (dict[lang] && dict[lang][key]) return dict[lang][key];
            return key;
        };
        return {
            t: t,
            lang: computed(function () { return globalState.db.env.lang }),
            dir: computed(function () { return globalState.db.env.dir }),
            toggleLang: function () {
                var newLang = globalState.db.env.lang === 'ar' ? 'en' : 'ar';
                globalState.db.env.lang = newLang;
                globalState.db.env.dir = newLang === 'ar' ? 'rtl' : 'ltr';
                document.documentElement.lang = newLang;
                document.documentElement.dir = globalState.db.env.dir;
            }
        };
    }
    function useTheme() {
        return {
            theme: computed(function () { return globalState.db.env.theme }),
            tokens: globalState.theme.tokens,
            vars: globalState.theme.vars,
            toggleTheme: function () {
                var newTheme = globalState.db.env.theme === 'dark' ? 'light' : 'dark';
                globalState.db.env.theme = newTheme;
                updateThemeVariables();
            }
        };
    }
    function updateThemeVariables() {
        var isDark = globalState.db.env.theme === 'dark';
        var tokens = {
            colors: {
                background: isDark ? '#0a0a0a' : '#ffffff',
                foreground: isDark ? '#ffffff' : '#000000',
                primary: '#3b82f6',
                secondary: '#6366f1',
                card: isDark ? '#1a1a1a' : '#f3f4f6',
                border: isDark ? '#333333' : '#e5e7eb',
                muted: isDark ? '#888888' : '#6b7280'
            },
            radius: { sm: '4px', md: '8px', lg: '16px', full: '9999px' },
            spacing: { sm: '8px', md: '16px', lg: '24px' }
        };
        globalState.theme.tokens = tokens;
        var vars = {};
        function flatten(obj, prefix) {
            for (var k in obj) {
                if (typeof obj[k] === 'object') { flatten(obj[k], prefix + '-' + k) }
                else { vars[prefix + '-' + k] = obj[k] }
            }
        }
        flatten(tokens, '--color'); // Simplified mapping
        // Fix mapping manually for better control
        vars['--color-background'] = tokens.colors.background;
        vars['--color-foreground'] = tokens.colors.foreground;
        vars['--color-primary'] = tokens.colors.primary;
        vars['--color-card'] = tokens.colors.card;
        vars['--color-border'] = tokens.colors.border;
        vars['--color-muted'] = tokens.colors.muted;

        globalState.theme.vars = vars;

        // Inject into root
        if (typeof document !== 'undefined') {
            var root = document.documentElement;
            for (var v in vars) { root.style.setProperty(v, vars[v]) }
        }
    }
    function html(strings) {
        var values = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            values[_i - 1] = arguments[_i];
        }
        // This is a placeholder for standard tagged template handling
        // In a real scenario, this would need complex parsing or be handled by the transform
        // But since we use mishkah-vue-jsx transformer, this function might just receive processed values
        // For the sake of the complete example which uses html`...`, we might need a runtime parser or the transformer
        // should handle it. Given the transformer 'mishkah-vue-jsx.js' is active, it likely transforms JSX, 
        // but if the user uses html`...` explicitly, that needs a runtime parser.
        // HOWEVER, the user asked for JSX transformer. The complete example uses `html` tagged template.
        // Let's implement a basic one or assume the transformer converts it?
        // Actually the transformer MishkahVueJSX converts <tags>.
        // If the user code has html`...`, that's valid JS, transformer won't touch it unless we tell it.
        // We'll stick to the requested "Mishkah.Vue.html" being available.
        // For now, let's just warn or handle simple string interpolation if used as a cheap way?
        // No, better to stick to h() or JSX. But to support the example:
        return "HTML Tagged Template not fully supported in Standalone Mode yet. Use JSX <script type='text/mishkah-vue'>";
    }

    return { createApp: createApp, h: h, ref: ref, reactive: reactive, computed: computed, watchEffect: effect, watch: function (src, cb) { var getter = typeof src === 'function' ? src : function () { return src.value }; var oldVal; effect(function () { var newVal = getter(); if (oldVal !== undefined && newVal !== oldVal) { cb(newVal, oldVal) } oldVal = newVal }) }, provide: provide, inject: inject, onMounted: onMounted, onUnmounted: onUnmounted, onUpdated: onUpdated, nextTick: nextTick, useDatabase: useDatabase, useI18n: useI18n, useTheme: useTheme, initMishkah: initMishkah, html: html, version: '3.2.0-standalone' }
})