(function () {
    const LOGIN_PATH = '/login.html';

    function ensureToken() {
        const token = localStorage.getItem('token');
        if (!token) {
            window.location.href = LOGIN_PATH;
            throw new Error('Missing auth token');
        }
        return token;
    }

    function buildAuthHeaders(token) {
        return {
            'Authorization': `Bearer ${token}`
        };
    }

    function handleUnauthorized(response) {
        if (response.status === 401) {
            localStorage.removeItem('token');
            window.location.href = LOGIN_PATH;
            return true;
        }
        return false;
    }

    async function fetchWithAuth(url, options = {}) {
        const token = ensureToken();
        const mergedOptions = {
            ...options,
            headers: {
                ...(options.headers || {}),
                ...buildAuthHeaders(token)
            }
        };

        const response = await fetch(url, mergedOptions);

        if (handleUnauthorized(response)) {
            throw new Error('Unauthorized');
        }

        if (!response.ok) {
            throw new Error(`Request failed with status ${response.status}`);
        }

        return response;
    }

    window.auth = {
        ensureToken,
        buildAuthHeaders,
        fetchWithAuth,
    };
})();
