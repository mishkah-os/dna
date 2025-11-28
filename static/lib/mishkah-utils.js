// Placeholder for mishkah-utils.js
window.Mishkah = window.Mishkah || {};
window.Mishkah.utils = {
    // Add minimal utils if needed by plotly adapter
    extend: function (target, source) {
        for (var key in source) {
            if (source.hasOwnProperty(key)) {
                target[key] = source[key];
            }
        }
        return target;
    }
};
