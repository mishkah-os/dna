/*!
 * mishkah-svelte.js — Svelte 5 (Runes) Layer for Mishkah
 * Provides: mount, state, derived, effect
 * 2025-12-03 (standalone edition)
 */
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(['mishkah'], function (M) { return factory(root, M); });
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory(root, require('mishkah'));
    } else {
        root.Mishkah = root.Mishkah || {};
        root.Mishkah.Svelte = factory(root, root.Mishkah);
    }
}(typeof window !== 'undefined' ? window : this, function (global, M) {
    "use strict";

    var activeSubscriber = null;
    var htmlId = 0;

    // -------------------------------------------------------------------
    // Tiny Template Helper (supports events + inline expressions)
    // -------------------------------------------------------------------
    function html(strings) {
        var values = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            values[_i - 1] = arguments[_i];
        }

        var fns = [];
        var out = '';
        for (var i = 0; i < strings.length; i++) {
            var seg = strings[i];
            out += seg;
            if (i >= values.length) continue;
            var val = values[i];
            var isEventSlot = /on[a-zA-Z]+=["']?$/.test(seg);

            if (typeof val === 'function' && isEventSlot) {
                var marker = '__mk_sv_fn_' + (++htmlId) + '__';
                fns.push({ id: marker, fn: val });
                out += marker;
            } else {
                out += typeof val === 'function' ? val() : (val == null ? '' : val);
            }
        }

        return { __mishkah_html: true, html: out, fns: fns };
    }

    function renderTemplate(tpl, ctx, target) {
        var templateHTML = '';
        var fnList = [];

        if (tpl && tpl.__mishkah_html) {
            templateHTML = tpl.html;
            fnList = tpl.fns || [];
        } else {
            templateHTML = tpl == null ? '' : String(tpl);
        }

        if (!target) return;
        target.innerHTML = templateHTML;

        if (!fnList.length) return;

        var all = target.querySelectorAll('*');
        for (var i = 0; i < all.length; i++) {
            var el = all[i];
            var attrs = Array.from(el.attributes);
            for (var j = 0; j < attrs.length; j++) {
                var attr = attrs[j];
                for (var k = 0; k < fnList.length; k++) {
                    var entry = fnList[k];
                    if (attr.value === entry.id && attr.name.indexOf('on') === 0) {
                        el.addEventListener(attr.name.substring(2), entry.fn.bind(ctx));
                        el.removeAttribute(attr.name);
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Signals-style runtime
    // -------------------------------------------------------------------
    function createNotifier() {
        var listeners = new Set();
        return {
            subscribe: function (fn) { listeners.add(fn); },
            notify: function () { listeners.forEach(function (fn) { return fn(); }); }
        };
    }

    function state(initialValue) {
        var notifier = createNotifier();

        function createProxy(target) {
            if (typeof target !== 'object' || target === null) return target;
            return new Proxy(target, {
                get: function (obj, prop) {
                    if (prop === '$subscribe') return function (fn) { notifier.subscribe(fn); };
                    if (activeSubscriber) notifier.subscribe(activeSubscriber);
                    return createProxy(obj[prop]);
                },
                set: function (obj, prop, value) {
                    obj[prop] = value;
                    notifier.notify();
                    return true;
                }
            });
        }

        return createProxy(initialValue);
    }

    function derived(fn) {
        return {
            get value() { return fn(); }
        };
    }

    function effect(fn) {
        fn();
    }

    function mount(ComponentFn, target) {
        var container = typeof target === 'string' ? document.querySelector(target) : target;
        if (!container) return;

        function render() {
            activeSubscriber = render;
            var result = ComponentFn();
            if (typeof result === 'function') result = result();
            renderTemplate(result, {}, container);
            activeSubscriber = null;
        }

        render();
    }

    return {
        state: state,
        derived: derived,
        effect: effect,
        mount: mount,
        html: html
    };

}));
