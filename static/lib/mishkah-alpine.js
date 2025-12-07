/*!
 * mishkah-alpine.js — Alpine.js-like (Direct DOM) Layer for Mishkah
 * Provides: start (auto-starts on DOMContentLoaded)
 * 2025-12-12
 */
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(['mishkah'], function (M) { return factory(root, M); });
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory(root, require('mishkah'));
    } else {
        root.Mishkah = root.Mishkah || {};
        root.Mishkah.Alpine = factory(root, root.Mishkah);
    }
}(typeof window !== 'undefined' ? window : this, function (global, M) {
    "use strict";

    // Lightweight reactive core (property-level tracking)
    var bucket = new WeakMap();
    var activeEffect = null;

    function track(target, key) {
        if (!activeEffect) return;
        var depsMap = bucket.get(target);
        if (!depsMap) {
            depsMap = new Map();
            bucket.set(target, depsMap);
        }
        var dep = depsMap.get(key);
        if (!dep) {
            dep = new Set();
            depsMap.set(key, dep);
        }
        dep.add(activeEffect);
    }

    function trigger(target, key) {
        var depsMap = bucket.get(target);
        if (!depsMap) return;
        var dep = depsMap.get(key);
        if (!dep) return;
        dep.forEach(function (effectFn) { effectFn(); });
    }

    function effect(fn) {
        var runner = function () {
            activeEffect = runner;
            try { fn(); } finally { activeEffect = null; }
        };
        runner();
        return runner;
    }

    function reactive(obj) {
        return new Proxy(obj, {
            get: function (target, prop) {
                track(target, prop);
                var value = target[prop];
                return (typeof value === 'object' && value !== null) ? reactive(value) : value;
            },
            set: function (target, prop, value) {
                target[prop] = value;
                trigger(target, prop);
                return true;
            }
        });
    }

    // -------------------------------------------------------------------
    // Directive bindings
    // -------------------------------------------------------------------
    function bindText(el, state, expr) {
        effect(function () {
            el.textContent = evaluate(expr, state);
        });
    }

    function bindHtml(el, state, expr) {
        effect(function () {
            el.innerHTML = evaluate(expr, state);
        });
    }

    function bindShow(el, state, expr) {
        effect(function () {
            el.style.display = evaluate(expr, state) ? '' : 'none';
        });
    }

    function bindModel(el, state, expr) {
        effect(function () { el.value = evaluate(expr, state); });
        el.addEventListener('input', function (e) {
            assign(expr, state, e.target.value);
        });
    }

    function bindAttr(el, state, prop, expr) {
        effect(function () {
            var val = evaluate(expr, state);
            if (prop === 'class' && typeof val === 'object') {
                Object.keys(val).forEach(function (cls) {
                    if (val[cls]) el.classList.add(cls); else el.classList.remove(cls);
                });
            } else {
                el.setAttribute(prop, val);
            }
        });
    }

    function bindOn(el, state, eventName, expr) {
        el.addEventListener(eventName, function (e) {
            evaluate(expr, state, { $event: e });
        });
    }

    // -------------------------------------------------------------------
    // Expression helpers
    // -------------------------------------------------------------------
    function evaluate(expr, state, extra) {
        try {
            var scopeKeys = Object.keys(state);
            var scopeVals = scopeKeys.map(function (k) { return state[k]; });
            var extraKeys = extra ? Object.keys(extra) : [];
            var extraVals = extra ? Object.values(extra) : [];
            var fn = new Function(scopeKeys.concat(extraKeys).join(','), 'return ' + expr);
            return fn.apply(state, scopeVals.concat(extraVals));
        } catch (e) {
            console.warn('Mishkah Alpine: Eval error', e);
            return '';
        }
    }

    function assign(expr, state, value) {
        try {
            var fn = new Function('state', 'value', 'with(state){ ' + expr + ' = value; }');
            fn(state, value);
        } catch (e) {
            console.warn('Mishkah Alpine: assign error', e);
        }
    }

    // -------------------------------------------------------------------
    // Walker
    // -------------------------------------------------------------------
    function walk(el, state) {
        if (el.nodeType !== 1) return;

        // x-text
        if (el.hasAttribute('x-text')) bindText(el, state, el.getAttribute('x-text'));
        // x-html
        if (el.hasAttribute('x-html')) bindHtml(el, state, el.getAttribute('x-html'));
        // x-show
        if (el.hasAttribute('x-show')) bindShow(el, state, el.getAttribute('x-show'));

        // Attributes
        Array.from(el.attributes).forEach(function (attr) {
            var name = attr.name;
            var value = attr.value;

            // x-model
            if (name === 'x-model') {
                bindModel(el, state, value);
            }

            // x-on / @
            if (name.startsWith('x-on:')) {
                bindOn(el, state, name.slice(5), value);
            } else if (name.startsWith('@')) {
                bindOn(el, state, name.slice(1), value);
            }

            // x-bind / :
            if (name.startsWith('x-bind:')) {
                bindAttr(el, state, name.slice(7), value);
            } else if (name.startsWith(':')) {
                bindAttr(el, state, name.slice(1), value);
            }
        });

        // Walk children (skip nested x-data scopes)
        var child = el.firstElementChild;
        while (child) {
            if (!child.hasAttribute('x-data')) walk(child, state);
            child = child.nextElementSibling;
        }
    }

    // -------------------------------------------------------------------
    // Bootstrapper
    // -------------------------------------------------------------------
    function start() {
        var roots = document.querySelectorAll('[x-data]');
        roots.forEach(function (rootEl) {
            if (rootEl.__mishkah_alpine_inited) return;
            rootEl.__mishkah_alpine_inited = true;

            var dataExpr = rootEl.getAttribute('x-data');
            var initialState = {};
            try {
                initialState = new Function('return ' + dataExpr)();
            } catch (e) {
                console.error('Mishkah Alpine: Error evaluating x-data', e);
            }

            var state = reactive(initialState);
            walk(rootEl, state);
        });
    }

    if (typeof document !== 'undefined') {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', start);
        } else {
            start();
        }
    }

    return { start: start };

}));
